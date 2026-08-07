#!/usr/bin/env python3
"""Generate the public synthetic corpus.

Every value is drawn from a fixed list of invented names and a public procedure
vocabulary. **No document, image, name or identifier from any production system
is used, referenced or derived from.** The generator is seeded, so the corpus is
reproducible from this file alone.

The point is not to look pretty. It is to reproduce the conditions that actually
break document extraction, one tag at a time:

``handwritten``  a jittered, slanted font in "ink"
``low_light``    reduced contrast and a lighting gradient
``phone_photo``  perspective skew, JPEG artefacts
``rotated``      90/180/270 — where a VLM confidently returns a coherent,
                 wrong reading instead of admitting it cannot read
``stamped``      a rotated stamp overlapping the licence number
``partial``      a corner of the page missing
``faded``        low-toner print

Usage::

    python tools/generate_synthetic_corpus.py --out corpora/synthetic-forms -n 60
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
except ImportError:  # pragma: no cover
    sys.exit("pillow is required: pip install 'docvlm-eval[corpus]'")

W, H = 1240, 1754  # A4 at 150 dpi

FIRST = [
    "Marina",
    "Otavio",
    "Bianca",
    "Rafael",
    "Leticia",
    "Gustavo",
    "Camila",
    "Henrique",
    "Priscila",
    "Eduardo",
    "Fernanda",
    "Rodrigo",
    "Juliana",
    "Thiago",
    "Vanessa",
    "Leonardo",
    "Patricia",
    "Marcelo",
    "Simone",
    "Anderson",
    "Carolina",
    "Vinicius",
]
LAST = [
    "Ferrazzo",
    "Quintanilha",
    "Marchetti",
    "Sobral",
    "Bittencourt",
    "Vasconcelos",
    "Andrade",
    "Rezende",
    "Camargo",
    "Peixoto",
    "Bandeira",
    "Tavares",
    "Nogueira",
    "Cavalcanti",
    "Bezerra",
    "Monteiro",
    "Rabelo",
    "Siqueira",
]
DOCTORS = [
    "Alceu Wenzel",
    "Marta Kolinski",
    "Sergio Andrioli",
    "Regina Bastos",
    "Ivan Poletto",
    "Celia Marques",
    "Nelson Grazziotin",
    "Beatriz Sanches",
]
EXAMS = [
    "Ecocardiograma transtoracico",
    "Holter 24h",
    "Ressonancia de joelho direito",
    "Tomografia de torax",
    "Ultrassom abdominal total",
    "Raio-X de torax PA e perfil",
    "Mamografia bilateral",
    "Densitometria ossea",
    "Doppler de carotidas",
    "Ressonancia de coluna lombar",
    "Ultrassom de tireoide",
    "Teste ergometrico",
]
UF = ["RS", "SC", "PR", "SP", "MG", "BA"]
CLINICS = ["Clinica Aurora", "Instituto Vale Verde", "Centro Medico Ipiranga"]
# Invented insurers. Real trade names would be recognisable on a fake medical
# form, and "every value is fabricated" has to be true without an asterisk.
INSURERS = [
    "Vitalis Saude",
    "Meridiano Saude",
    "Plano Aurora",
    "Boavida Assistencia",
    "Interclin",
    "Particular",
]

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]
HAND_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
]


def _font(paths: list[str], size: int):
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _hand_text(draw, xy, text, font, fill=(20, 30, 90), jitter=2.4, rng=None):
    """Draw text character by character with jitter and baseline drift.

    Not real handwriting — but it reproduces the property that matters: locally
    ambiguous glyphs the model has to commit to, on a value (a licence number)
    where committing wrongly is invisible downstream.
    """
    rng = rng or random
    x, y = xy
    for ch in text:
        dy = rng.uniform(-jitter, jitter)
        draw.text((x, y + dy), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + rng.uniform(-0.6, 1.4)


def build_form(rec: dict, tags: list[str], rng: random.Random) -> Image.Image:
    handwritten = "handwritten" in tags
    img = Image.new("RGB", (W, H), (252, 251, 248))
    d = ImageDraw.Draw(img)

    f_title = _font(FONT_CANDIDATES, 40)
    f_label = _font(FONT_CANDIDATES, 25)
    f_value = _font(HAND_CANDIDATES if handwritten else FONT_CANDIDATES, 30)
    f_small = _font(FONT_CANDIDATES, 20)

    # Header
    d.rectangle([60, 60, W - 60, 190], outline=(120, 120, 120), width=2)
    d.text((85, 80), rec["_clinic"], font=f_title, fill=(35, 35, 35))
    d.text((85, 135), "REQUISICAO DE EXAME / SOLICITACAO MEDICA", font=f_small, fill=(90, 90, 90))
    d.text((W - 320, 80), f"No {rec['_form_no']}", font=f_label, fill=(90, 90, 90))

    y = 250
    rows = [
        ("PACIENTE", rec.get("patient_name") or ""),
        ("DATA DE NASCIMENTO", rec.get("_birth") or ""),
        ("CONVENIO", rec.get("insurer") or ""),
        ("CARTEIRA", rec.get("member_id") or ""),
    ]
    for label, value in rows:
        d.text((85, y), label, font=f_label, fill=(110, 110, 110))
        d.line([85, y + 62, W - 85, y + 62], fill=(170, 170, 170), width=1)
        if value:
            if handwritten:
                _hand_text(d, (95, y + 26), str(value), f_value, rng=rng)
            else:
                d.text((95, y + 28), str(value), font=f_value, fill=(25, 25, 25))
        y += 100

    # Exams block
    y += 20
    d.text((85, y), "EXAMES SOLICITADOS", font=f_label, fill=(110, 110, 110))
    y += 45
    d.rectangle([85, y, W - 85, y + 230], outline=(170, 170, 170), width=1)
    yy = y + 20
    for exam in rec.get("exams", []):
        d.text((105, yy), "X", font=f_label, fill=(20, 30, 90))
        if handwritten:
            _hand_text(d, (145, yy - 4), exam, f_value, rng=rng)
        else:
            d.text((145, yy), exam, font=f_value, fill=(25, 25, 25))
        yy += 52
    y += 270

    # Urgency
    d.text((85, y), "CARATER:", font=f_label, fill=(110, 110, 110))
    urgent = bool(rec.get("urgent"))
    for i, (label, is_it) in enumerate([("ROTINA", not urgent), ("URGENTE", urgent)]):
        bx = 260 + i * 240
        d.rectangle([bx, y - 4, bx + 30, y + 26], outline=(90, 90, 90), width=2)
        if is_it:
            d.line([bx + 4, y + 10, bx + 13, y + 21], fill=(20, 30, 90), width=4)
            d.line([bx + 13, y + 21, bx + 27, y], fill=(20, 30, 90), width=4)
        d.text((bx + 45, y), label, font=f_label, fill=(60, 60, 60))
    y += 90

    # Date
    d.text((85, y), "DATA", font=f_label, fill=(110, 110, 110))
    d.line([85, y + 62, 520, y + 62], fill=(170, 170, 170), width=1)
    if rec.get("request_date"):
        shown = _fmt_date(rec["request_date"])
        if handwritten:
            _hand_text(d, (95, y + 26), shown, f_value, rng=rng)
        else:
            d.text((95, y + 28), shown, font=f_value, fill=(25, 25, 25))
    y += 130

    # Doctor stamp block
    d.line([700, y + 40, W - 85, y + 40], fill=(120, 120, 120), width=1)
    d.text((700, y + 50), "Assinatura e carimbo do medico", font=f_small, fill=(140, 140, 140))
    stamp = Image.new("RGBA", (520, 190), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stamp)
    ink = (30, 45, 120, 235)
    sd.rectangle([2, 2, 516, 186], outline=ink, width=3)
    sd.text((22, 26), f"Dr(a). {rec['_doctor']}", font=f_label, fill=ink)
    crm_text = f"CRM {rec.get('doctor_crm') or ''}"
    if handwritten:
        _hand_text(sd, (22, 78), crm_text, f_value, fill=ink, jitter=1.6, rng=rng)
    else:
        sd.text((22, 78), crm_text, font=f_value, fill=ink)
    sd.text((22, 130), rec["_specialty"], font=f_small, fill=ink)
    stamp = stamp.rotate(rng.uniform(-7, 7), expand=True, resample=Image.BICUBIC)
    img.paste(stamp, (690, y - 130), stamp)

    d.text(
        (85, H - 90),
        "SYNTHETIC DOCUMENT — generated for evaluation. Not a real request.",
        font=f_small,
        fill=(190, 190, 190),
    )
    return img


def _fmt_date(iso: str) -> str:
    y, m, dd = iso.split("-")
    return f"{dd}/{m}/{y}"


# --------------------------------------------------------------------------- #
# Degradations — one per tag
# --------------------------------------------------------------------------- #


def apply_tags(img: Image.Image, tags: list[str], rng: random.Random) -> Image.Image:
    if "faded" in tags:
        img = ImageEnhance.Contrast(img).enhance(0.55)
        img = ImageEnhance.Brightness(img).enhance(1.12)

    if "low_light" in tags:
        img = ImageEnhance.Brightness(img).enhance(0.52)
        img = ImageEnhance.Contrast(img).enhance(0.72)
        gradient = Image.linear_gradient("L").resize(img.size).rotate(rng.choice([0, 90, 180, 270]))
        img = Image.composite(
            img, Image.new("RGB", img.size, (18, 18, 22)), gradient.point(lambda p: 90 + p // 2)
        )

    if "phone_photo" in tags:
        dx = rng.uniform(0.02, 0.05)
        w, h = img.size
        img = img.transform(
            (w, h),
            Image.QUAD,
            data=(
                w * dx,
                h * dx * 0.4,
                w * dx * 0.5,
                h * (1 - dx * 0.5),
                w * (1 - dx * 0.3),
                h * (1 - dx * 0.2),
                w * (1 - dx),
                h * dx * 0.8,
            ),
            resample=Image.BICUBIC,
            fillcolor=(70, 70, 74),
        )
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.1)))

    if "partial" in tags:
        d = ImageDraw.Draw(img)
        w, h = img.size
        d.polygon([(w, h), (w - int(w * 0.28), h), (w, h - int(h * 0.22))], fill=(30, 30, 34))

    if "stamped" in tags:
        d = ImageDraw.Draw(img, "RGBA")
        w, h = img.size
        overlay = Image.new("RGBA", (420, 160), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.ellipse([2, 2, 418, 158], outline=(150, 30, 30, 190), width=6)
        od.text(
            (70, 60), "RECEBIDO 14:02", font=_font(FONT_CANDIDATES, 32), fill=(150, 30, 30, 200)
        )
        overlay = overlay.rotate(rng.uniform(-25, 25), expand=True, resample=Image.BICUBIC)
        img.paste(overlay, (int(w * 0.52), int(h * 0.70)), overlay)

    if "rotated" in tags:
        img = img.rotate(rng.choice([90, 180, 270]), expand=True)
    elif "skewed" in tags:
        img = img.rotate(rng.uniform(-4, 4), expand=True, fillcolor=(240, 239, 236))

    return img


def apply_hard(img: Image.Image, tags: list[str], rng: random.Random) -> Image.Image:
    """Extra degradations for the ``hard`` corpus.

    The clean synthetic corpus saturates: current vision models score at or near
    1.000 on it, and a benchmark everybody passes measures nothing. What
    separates a solvable document from an unsolvable one in practice is mostly
    **effective resolution on the glyph** — not artistic noise. So the main lever
    here is a genuine round trip through a lower resolution, plus the specific
    occlusions that hurt: a stamp over the licence number, and glare.
    """
    w, h = img.size

    # Resolution round trip — the dominant real-world factor.
    scale = rng.uniform(0.26, 0.38)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    if "glare" in tags:
        glare = Image.new("L", img.size, 0)
        gd = ImageDraw.Draw(glare)
        gw, gh = img.size
        cx, cy = rng.uniform(0.3, 0.7) * gw, rng.uniform(0.2, 0.6) * gh
        gd.ellipse([cx - gw * 0.32, cy - gh * 0.16, cx + gw * 0.32, cy + gh * 0.16], fill=190)
        glare = glare.filter(ImageFilter.GaussianBlur(gw * 0.06))
        img = Image.composite(Image.new("RGB", img.size, (255, 255, 252)), img, glare)

    if "stamped" in tags:
        # Directly over the licence number: the value where a confident
        # misread is most expensive downstream.
        d = ImageDraw.Draw(img, "RGBA")
        gw, gh = img.size
        d.ellipse(
            [gw * 0.55, gh * 0.60, gw * 0.98, gh * 0.72],
            outline=(140, 25, 25, 170),
            width=max(2, int(gw * 0.006)),
        )

    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.0)))
    if "motion_blur" in tags:
        k = 5  # PIL supports 3x3 and 5x5 kernels only
        kernel = [0] * (k * k)
        for i in range(k):
            kernel[i * k + i] = 1
        img = img.filter(ImageFilter.Kernel((k, k), kernel, scale=k))

    img = img.resize((int(w * 0.55), int(h * 0.55)), Image.BICUBIC)
    return img


TAG_PLAN = [
    (["printed", "scanned"], 0.22),
    (["printed", "phone_photo"], 0.12),
    (["handwritten", "scanned"], 0.18),
    (["handwritten", "phone_photo"], 0.14),
    (["handwritten", "phone_photo", "low_light"], 0.10),
    (["printed", "scanned", "stamped"], 0.08),
    (["handwritten", "scanned", "faded"], 0.06),
    (["printed", "phone_photo", "rotated"], 0.05),
    (["handwritten", "phone_photo", "partial"], 0.05),
]


HARD_EXTRA = [(["glare"], 0.30), (["motion_blur"], 0.30), ([], 0.40)]


def make_record(i: int, rng: random.Random, hard: bool = False) -> tuple[dict, list[str]]:
    tags = list(rng.choices([t for t, _ in TAG_PLAN], weights=[w for _, w in TAG_PLAN])[0])
    if "phone_photo" in tags and rng.random() < 0.3:
        tags.append("skewed")
    if hard:
        tags += rng.choices([t for t, _ in HARD_EXTRA], weights=[w for _, w in HARD_EXTRA])[0]

    n_exams = rng.choices([1, 1, 1, 2, 3], weights=[0.5, 0.15, 0.1, 0.18, 0.07])[0]
    exams = rng.sample(EXAMS, n_exams)
    doctor = rng.choice(DOCTORS)
    request_date = date(2026, 1, 1) + timedelta(days=rng.randrange(0, 210))

    # ~12% of documents have no readable member id; the ground truth is empty.
    # These are the cases that expose hallucination — a model that always fills
    # the field scores well on the other 88% and is dangerous on these.
    has_member_id = rng.random() > 0.12
    has_insurer = rng.random() > 0.08

    record = {
        "patient_name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        "doctor_crm": f"{rng.randrange(10000, 99999)}-{rng.choice(UF)}",
        "insurer": rng.choice(INSURERS) if has_insurer else None,
        "member_id": "".join(str(rng.randrange(10)) for _ in range(rng.choice([9, 12, 15])))
        if has_member_id
        else None,
        "exams": exams,
        "request_date": request_date.isoformat(),
        "urgent": rng.random() < 0.22,
        "_doctor": doctor,
        "_clinic": rng.choice(CLINICS),
        "_form_no": f"{rng.randrange(100000, 999999)}",
        "_birth": f"{rng.randrange(1, 29):02d}/{rng.randrange(1, 13):02d}/"
        f"{rng.randrange(1945, 2010)}",
        "_specialty": rng.choice(
            ["Cardiologia", "Ortopedia", "Clinica Geral", "Radiologia", "Endocrinologia"]
        ),
    }
    return record, tags


TRUTH_KEYS = (
    "patient_name",
    "doctor_crm",
    "insurer",
    "member_id",
    "exams",
    "request_date",
    "urgent",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="corpora/synthetic-forms")
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--quality", type=int, default=82, help="JPEG quality.")
    ap.add_argument(
        "--hard",
        action="store_true",
        help="Generate the hard variant: low effective resolution, glare, motion blur, "
        "a stamp over the licence number. The clean corpus saturates.",
    )
    args = ap.parse_args()
    if args.hard and args.quality == 82:
        args.quality = 34

    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    lines = []
    for i in range(1, args.n + 1):
        record, tags = make_record(i, rng, hard=args.hard)
        img = build_form(record, tags, rng)
        img = apply_tags(img, tags, rng)
        if args.hard:
            img = apply_hard(img, tags, rng)
        case_id = f"{i:04d}"
        rel = f"images/{case_id}.jpg"
        img.convert("RGB").save(out / rel, quality=args.quality, optimize=True)
        lines.append(
            json.dumps(
                {
                    "id": case_id,
                    "image": rel,
                    "truth": {k: record[k] for k in TRUTH_KEYS},
                    "tags": tags,
                },
                ensure_ascii=False,
            )
        )
        print(f"  {case_id}  {','.join(tags)}", file=sys.stderr)

    (out / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.n} cases to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
