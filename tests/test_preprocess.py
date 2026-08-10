"""Image preparation profiles."""

from __future__ import annotations

import asyncio
import io

import pytest

from docvlm_eval.config import Config
from docvlm_eval.corpus import load_corpus
from docvlm_eval.engine import run_config
from docvlm_eval.preprocess import PROFILES, available, describe, prepare

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def make_jpeg(size=(1200, 1600), colour=(200, 210, 190)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def test_none_is_byte_identical():
    """The default has to be a true no-op, not a re-encode. Otherwise every
    corpus is silently transcoded before anyone opts into anything."""
    raw = make_jpeg()
    assert prepare(raw, "none") is raw
    assert prepare(raw, "") is raw


@pytest.mark.parametrize("name", [n for n in PROFILES if n != "none"])
def test_every_profile_produces_a_readable_image(name):
    out = prepare(make_jpeg(), name)
    img = Image.open(io.BytesIO(out))
    img.load()
    assert max(img.size) == PROFILES[name].max_side


def test_profiles_produce_different_bytes():
    """If two profiles produced identical bytes the comparison would be
    measuring nothing, and the cache would happily serve one for the other."""
    raw = make_jpeg()
    outputs = {name: prepare(raw, name) for name in available() if name != "none"}
    assert len(set(outputs.values())) == len(outputs)


def test_greyscale_profiles_are_greyscale():
    out = prepare(make_jpeg(), "maxima")
    assert (
        Image.open(io.BytesIO(out)).convert("RGB").getpixel((5, 5))[0]
        == Image.open(io.BytesIO(out)).convert("RGB").getpixel((5, 5))[1]
    )


def test_maxima_is_lossless():
    assert Image.open(io.BytesIO(prepare(make_jpeg(), "maxima"))).format == "PNG"


def test_preprocessing_is_deterministic():
    """A benchmark cannot have a nondeterministic input stage."""
    raw = make_jpeg()
    assert prepare(raw, "alta") == prepare(raw, "alta")


def test_unknown_profile_fails_loudly():
    with pytest.raises(ValueError, match="unknown preprocess profile"):
        prepare(make_jpeg(), "nao_existe")


def test_describe_goes_into_provenance():
    d = describe("alta_pb")
    assert d["grayscale"] is True and d["max_side"] == 2048


# --------------------------------------------------------------------------- #
# Integration: the profile has to reach the cache key and the provenance
# --------------------------------------------------------------------------- #


def build_corpus(tmp_path):
    from tests.test_end_to_end import SCHEMA_SRC

    root = tmp_path / "corp"
    (root / "images").mkdir(parents=True)
    (root / "schema.py").write_text(SCHEMA_SRC)
    import json

    lines = []
    for i in range(6):
        cid = f"{i:03d}"
        (root / "images" / f"{cid}.jpg").write_bytes(make_jpeg((400, 500)))
        lines.append(
            json.dumps(
                {
                    "id": cid,
                    "image": f"images/{cid}.jpg",
                    "truth": {"name": f"P{cid}", "crm": f"{10000 + i}"},
                    "tags": [],
                }
            )
        )
    (root / "manifest.jsonl").write_text("\n".join(lines))
    return root


def test_changing_the_profile_changes_the_config_hash(tmp_path):
    a = Config(name="x", runner="mock", model="m", preprocess="padrao")
    b = Config(name="x", runner="mock", model="m", preprocess="alta")
    assert a.hash != b.hash  # so the cache cannot serve one for the other


def test_profile_is_recorded_in_provenance(tmp_path):
    corpus = load_corpus(build_corpus(tmp_path))
    cfg = Config(name="p", runner="mock", model="mock", preprocess="alta_pb")
    run = asyncio.run(run_config(corpus, cfg, concurrency=2))
    assert run.provenance["preprocess"]["name"] == "alta_pb"
    assert run.provenance["preprocess"]["grayscale"] is True
