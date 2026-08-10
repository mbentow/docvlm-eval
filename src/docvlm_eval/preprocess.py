"""Image preparation profiles.

Preprocessing is a *variable under test*, not a fixed step. Resizing, converting
to greyscale or sharpening changes what the model sees, and the effect is rarely
uniform across fields — a profile that helps a printed header can destroy a
handwritten stamp.

Each profile is named, deterministic, and part of the config hash, so a run made
with preprocessing is never silently compared to one made without, and the cache
invalidates when the profile changes.

The profiles below mirror the shape of a real production ladder: a default, a
high-resolution variant, a greyscale variant, a maximum-fidelity variant, and a
channel-dropout variant for pre-printed forms. Tune the numbers to your own
documents — these are a starting point, not a recommendation.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

_PIL_ERROR = (
    "image preprocessing needs pillow: pip install 'docvlm-eval[images]' (or use preprocess: none)"
)


@dataclass(frozen=True)
class Profile:
    """One named image preparation."""

    name: str
    max_side: int = 0
    """Longest edge, in pixels. 0 leaves the size alone."""
    fmt: str = "JPEG"
    quality: int = 88
    grayscale: bool = False
    contrast: float = 1.0
    sharpness: float = 1.0
    drop_green: bool = False
    """Discard the green channel. Pre-printed forms are often green; dropping
    that channel can leave the handwriting and erase the template."""
    flat_field: bool = False
    """Divide out a heavy blur of the page to flatten uneven lighting."""
    description: str = ""


PROFILES: dict[str, Profile] = {
    "none": Profile("none", description="Send the file exactly as it is on disk."),
    "padrao": Profile(
        "padrao",
        max_side=1440,
        quality=88,
        description="Default: modest downscale, keeps colour.",
    ),
    "alta": Profile(
        "alta",
        max_side=2048,
        quality=92,
        contrast=1.25,
        sharpness=1.6,
        description="Higher resolution with contrast and sharpening.",
    ),
    "alta_pb": Profile(
        "alta_pb",
        max_side=2048,
        quality=92,
        grayscale=True,
        contrast=1.35,
        sharpness=1.6,
        description="Same as alta, in greyscale.",
    ),
    "maxima": Profile(
        "maxima",
        max_side=2560,
        fmt="PNG",
        grayscale=True,
        contrast=1.3,
        sharpness=1.4,
        description="Maximum fidelity: lossless, greyscale, large.",
    ),
    "formulario": Profile(
        "formulario",
        max_side=2560,
        quality=92,
        drop_green=True,
        flat_field=True,
        contrast=1.4,
        sharpness=1.5,
        description="For pre-printed forms: green-channel dropout plus flat field.",
    ),
}


def available() -> list[str]:
    return sorted(PROFILES)


def prepare(image: bytes, profile_name: str) -> bytes:
    """Apply a named profile. ``none`` is a no-op and needs no dependencies."""
    if not profile_name or profile_name == "none":
        return image
    try:
        profile = PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(
            f"unknown preprocess profile {profile_name!r}; available: {available()}"
        ) from exc
    return _apply(image, profile)


def _apply(image: bytes, profile: Profile) -> bytes:
    try:
        from PIL import Image, ImageEnhance, ImageFilter
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(_PIL_ERROR) from exc

    img = Image.open(io.BytesIO(image))
    img = img.convert("RGB")

    if profile.drop_green:
        # Keep red and blue, rebuild green from them. Green ink and green
        # pre-printed rules fade; black and blue handwriting survives.
        r, _g, b = img.split()
        img = Image.merge("RGB", (r, Image.blend(r, b, 0.5), b))

    if profile.flat_field:
        blurred = img.filter(ImageFilter.GaussianBlur(max(img.size) / 24))
        img = Image.blend(img, _invert(blurred), 0.35)

    if profile.grayscale:
        img = img.convert("L")

    if profile.max_side:
        longest = max(img.size)
        if longest != profile.max_side:
            scale = profile.max_side / longest
            img = img.resize(
                (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                Image.LANCZOS,
            )

    if profile.contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(profile.contrast)
    if profile.sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(profile.sharpness)

    buffer = io.BytesIO()
    if profile.fmt == "PNG":
        img.save(buffer, format="PNG", optimize=True)
    else:
        img.convert("RGB").save(buffer, format="JPEG", quality=profile.quality, optimize=True)
    return buffer.getvalue()


def _invert(img):
    from PIL import ImageOps

    return ImageOps.invert(img.convert("RGB"))


def describe(profile_name: str) -> dict[str, Any]:
    """Profile settings, for the run provenance."""
    profile = PROFILES.get(profile_name or "none")
    if profile is None:
        return {"name": profile_name, "unknown": True}
    return {
        "name": profile.name,
        "max_side": profile.max_side,
        "format": profile.fmt,
        "quality": profile.quality,
        "grayscale": profile.grayscale,
        "contrast": profile.contrast,
        "sharpness": profile.sharpness,
        "drop_green": profile.drop_green,
        "flat_field": profile.flat_field,
    }
