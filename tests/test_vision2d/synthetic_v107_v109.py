"""Original, deterministic in-memory fixtures for V1-07 through V1-09."""

from __future__ import annotations

import cv2
import numpy as np


def _transform_gray(
    image: np.ndarray,
    *,
    rotate_deg: float = 0.0,
    scale: float = 1.0,
    noise_sigma: float = 0.0,
    brightness: float = 0.0,
    seed: int = 20260802,
) -> np.ndarray:
    """Apply bounded deterministic image perturbations without file I/O."""
    output = image
    if scale != 1.0:
        interpolation = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_AREA
        output = cv2.resize(output, None, fx=scale, fy=scale, interpolation=interpolation)
    if rotate_deg:
        height, width = output.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), rotate_deg, 1.0)
        output = cv2.warpAffine(
            output,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
    if brightness:
        output = np.clip(output.astype(np.float32) + brightness, 0.0, 255.0).astype(
            np.uint8
        )
    if noise_sigma:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, noise_sigma, output.shape)
        output = np.clip(output.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    return output


def make_qr(
    payload: str,
    *,
    scale: int = 4,
    rotate_deg: float = 0.0,
    noise_sigma: float = 0.0,
    brightness: float = 0.0,
) -> np.ndarray:
    """Create an OpenCV-encoded QR image with a deterministic white quiet zone."""
    encoded = cv2.QRCodeEncoder_create().encode(payload)
    scaled = cv2.resize(encoded, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    quiet = max(12, 4 * scale)
    canvas = cv2.copyMakeBorder(
        scaled,
        quiet,
        quiet,
        quiet,
        quiet,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    transformed = _transform_gray(
        canvas,
        rotate_deg=rotate_deg,
        noise_sigma=noise_sigma,
        brightness=brightness,
        seed=11,
    )
    return cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)


def _rs1d_bits(payload: str) -> list[int]:
    raw = payload.encode("utf-8")
    if len(raw) > 64:
        raise ValueError("fixture payload is limited to 64 bytes")
    frame = bytes([0xA7, len(raw)]) + raw + bytes([sum(raw) & 0xFF])
    bits: list[int] = []
    for value in frame:
        bits.extend((value >> shift) & 1 for shift in range(7, -1, -1))
    return bits


def make_rs1d(
    payload: str,
    *,
    module: int = 3,
    bar_height: int = 72,
    rotate_deg: float = 0.0,
    scale: float = 1.0,
    noise_sigma: float = 0.0,
    brightness: float = 0.0,
    damaged: bool = False,
) -> np.ndarray:
    """Create the project's original RS1D teaching barcode.

    A bit is a black bar of one module for 0 or two modules for 1, followed by
    one white module. Four-module black guards delimit the frame.
    """
    if module < 2 or bar_height < 20:
        raise ValueError("module and bar_height are too small")
    bits = _rs1d_bits(payload)
    run_widths = [4 * module] + [module * (1 + bit) for bit in bits] + [4 * module]
    quiet = 12 * module
    width = 2 * quiet + sum(run_widths) + module * len(run_widths)
    height = bar_height + 2 * quiet
    image = np.full((height, width), 255, dtype=np.uint8)
    x = quiet
    for index, bar_width in enumerate(run_widths):
        image[quiet : quiet + bar_height, x : x + bar_width] = 0
        if damaged and index == len(run_widths) // 2:
            image[quiet : quiet + bar_height, x + bar_width // 3 : x + (2 * bar_width) // 3] = 255
        x += bar_width + module
    transformed = _transform_gray(
        image,
        rotate_deg=rotate_deg,
        scale=scale,
        noise_sigma=noise_sigma,
        brightness=brightness,
        seed=17,
    )
    return cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)


_EAN_L_PATTERNS = (
    "0001101",
    "0011001",
    "0010011",
    "0111101",
    "0100011",
    "0110001",
    "0101111",
    "0111011",
    "0110111",
    "0001011",
)
_EAN_G_PATTERNS = (
    "0100111",
    "0110011",
    "0011011",
    "0100001",
    "0011101",
    "0111001",
    "0000101",
    "0010001",
    "0001001",
    "0010111",
)
_EAN_R_PATTERNS = (
    "1110010",
    "1100110",
    "1101100",
    "1000010",
    "1011100",
    "1001110",
    "1010000",
    "1000100",
    "1001000",
    "1110100",
)
_EAN_PARITY = (
    "AAAAAA",
    "AABABB",
    "AABBAB",
    "AABBBA",
    "ABAABB",
    "ABBAAB",
    "ABBBAA",
    "ABABAB",
    "ABABBA",
    "ABBABA",
)


def _ean13_checksum(first_twelve: str) -> str:
    total = sum((3 if index % 2 else 1) * int(value) for index, value in enumerate(first_twelve))
    return str((-total) % 10)


def make_ean13(
    digits: str = "590123412345",
    *,
    module: int = 3,
    bar_height: int = 72,
    rotate_deg: float = 0.0,
    scale: float = 1.0,
    noise_sigma: float = 0.0,
    brightness: float = 0.0,
) -> np.ndarray:
    """Create a standards-shaped EAN-13 symbol without external generators."""
    if len(digits) == 12:
        digits = digits + _ean13_checksum(digits)
    if len(digits) != 13 or not digits.isdigit():
        raise ValueError("EAN-13 fixture requires 12 or 13 decimal digits")
    if digits[-1] != _ean13_checksum(digits[:12]):
        raise ValueError("EAN-13 checksum is invalid")
    bits = "101"
    parity = _EAN_PARITY[int(digits[0])]
    for digit, side in zip(digits[1:7], parity):
        table = _EAN_L_PATTERNS if side == "A" else _EAN_G_PATTERNS
        bits += table[int(digit)]
    bits += "01010"
    for digit in digits[7:]:
        bits += _EAN_R_PATTERNS[int(digit)]
    bits += "101"
    quiet = 10 * module
    image = np.full((bar_height + 2 * quiet, len(bits) * module + 2 * quiet), 255, dtype=np.uint8)
    for index, bit in enumerate(bits):
        if bit == "1":
            image[quiet : quiet + bar_height, quiet + index * module : quiet + (index + 1) * module] = 0
    transformed = _transform_gray(
        image,
        rotate_deg=rotate_deg,
        scale=scale,
        noise_sigma=noise_sigma,
        brightness=brightness,
        seed=19,
    )
    return cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)


def compose_side_by_side(*images: np.ndarray, gap: int = 36) -> np.ndarray:
    """Place same-height BGR fixtures on a white canvas."""
    if not images:
        raise ValueError("at least one image is required")
    height = max(image.shape[0] for image in images)
    width = sum(image.shape[1] for image in images) + gap * (len(images) - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x = 0
    for image in images:
        y = (height - image.shape[0]) // 2
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
        x += image.shape[1] + gap
    return canvas


# Original compact 5x7 glyphs.  These are test assets, not a system font.
GLYPH_BITMAPS: dict[str, tuple[str, ...]] = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
}


def make_glyph(
    label: str,
    *,
    scale: int = 4,
    rotate_deg: float = 0.0,
    noise_sigma: float = 0.0,
    brightness: float = 0.0,
    broken: bool = False,
) -> np.ndarray:
    """Render one original 5x7 glyph to a BGR uint8 image."""
    bitmap = GLYPH_BITMAPS[label]
    margin = 2 * scale
    glyph = np.full((7 * scale + 2 * margin, 5 * scale + 2 * margin), 255, dtype=np.uint8)
    for row, bits in enumerate(bitmap):
        for column, bit in enumerate(bits):
            if bit == "1":
                glyph[
                    margin + row * scale : margin + (row + 1) * scale,
                    margin + column * scale : margin + (column + 1) * scale,
                ] = 0
    if broken:
        middle = glyph.shape[0] // 2
        glyph[middle : middle + max(1, scale // 2), margin:-margin] = 255
    transformed = _transform_gray(
        glyph,
        rotate_deg=rotate_deg,
        noise_sigma=noise_sigma,
        brightness=brightness,
        seed=31 + ord(label),
    )
    return cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)


def make_text(
    text: str,
    *,
    scale: int = 4,
    spacing: int | None = None,
    rotate_deg: float = 0.0,
    noise_sigma: float = 0.0,
    brightness: float = 0.0,
    broken_index: int | None = None,
) -> np.ndarray:
    """Render a text line from the original bitmap glyphs."""
    if spacing is None:
        spacing = scale * 2
    glyphs = [
        make_glyph(
            label,
            scale=scale,
            broken=index == broken_index,
        )
        for index, label in enumerate(text)
    ]
    height = max(glyph.shape[0] for glyph in glyphs)
    width = sum(glyph.shape[1] for glyph in glyphs) + spacing * max(0, len(glyphs) - 1)
    canvas = np.full((height, width), 255, dtype=np.uint8)
    x = 0
    for glyph in glyphs:
        y = (height - glyph.shape[0]) // 2
        gray = cv2.cvtColor(glyph, cv2.COLOR_BGR2GRAY)
        canvas[y : y + gray.shape[0], x : x + gray.shape[1]] = np.minimum(
            canvas[y : y + gray.shape[0], x : x + gray.shape[1]], gray
        )
        x += gray.shape[1] + spacing
    transformed = _transform_gray(
        canvas,
        rotate_deg=rotate_deg,
        noise_sigma=noise_sigma,
        brightness=brightness,
        seed=41,
    )
    return cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)


def make_glyph_samples(
    labels: str = "AB12",
    *,
    count: int = 8,
    seed: int = 20260802,
) -> dict[str, list[np.ndarray]]:
    """Make separate deterministic training samples for every label."""
    rng = np.random.default_rng(seed)
    samples: dict[str, list[np.ndarray]] = {}
    for label in labels:
        samples[label] = []
        for _index in range(count):
            samples[label].append(
                make_glyph(
                    label,
                    scale=int(rng.choice([3, 4, 5])),
                    rotate_deg=float(rng.choice([-3.0, 0.0, 3.0])),
                    noise_sigma=float(rng.choice([0.0, 0.5, 1.0])),
                    brightness=float(rng.choice([-4.0, 0.0, 4.0])),
                )
            )
    return samples


def make_surface_pair(
    kind: str = "pass",
    *,
    brightness: int = 0,
    noise_sigma: float = 0.0,
    size: tuple[int, int] = (220, 220),
) -> tuple[np.ndarray, np.ndarray]:
    """Create a reference rectangle and one deterministic surface variant."""
    height, width = size
    reference = np.full((height, width), 235, dtype=np.uint8)
    cv2.rectangle(reference, (55, 42), (155, 166), 48, thickness=-1)
    candidate = reference.copy()
    if kind in {"multi_component_pass", "multi_component_broken"}:
        reference = np.full((height, width), 235, dtype=np.uint8)
        cv2.rectangle(reference, (25, 42), (85, 166), 48, thickness=-1)
        cv2.rectangle(reference, (135, 42), (195, 166), 48, thickness=-1)
        candidate = reference.copy()
        if kind == "multi_component_broken":
            candidate[100:108, 25:86] = 235
    elif kind == "same_hole":
        cv2.circle(reference, (105, 103), 18, 235, thickness=-1)
        candidate = reference.copy()
    elif kind == "filled_hole":
        # A legal reference hole is completely filled by candidate material.
        # The radius is intentionally about 10 px so the extra foreground is
        # large enough to exercise the configured foreign-area threshold.
        cv2.circle(reference, (105, 103), 10, 235, thickness=-1)
        candidate = reference.copy()
        cv2.circle(candidate, (105, 103), 10, 48, thickness=-1)
    elif kind == "shifted":
        matrix = np.asarray([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]], dtype=np.float32)
        candidate = cv2.warpAffine(candidate, matrix, (width, height), borderValue=235)
    elif kind == "missing":
        candidate[92:137, 55:105] = 235
    elif kind == "hole":
        cv2.circle(candidate, (105, 103), 18, 235, thickness=-1)
    elif kind == "foreign":
        cv2.circle(candidate, (185, 175), 12, 48, thickness=-1)
    elif kind == "broken":
        candidate[100:108, 55:155] = 235
    elif kind == "dimension":
        candidate = np.full((height, width), 235, dtype=np.uint8)
        cv2.rectangle(candidate, (45, 34), (170, 178), 48, thickness=-1)
    elif kind != "pass":
        raise ValueError(f"unknown surface fixture kind: {kind}")
    if brightness:
        reference = np.clip(reference.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
        candidate = np.clip(candidate.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    if noise_sigma:
        rng = np.random.default_rng(101)
        reference = np.clip(
            reference.astype(np.float32) + rng.normal(0.0, noise_sigma, reference.shape),
            0,
            255,
        ).astype(np.uint8)
        candidate = np.clip(
            candidate.astype(np.float32) + rng.normal(0.0, noise_sigma, candidate.shape),
            0,
            255,
        ).astype(np.uint8)
    return cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR), cv2.cvtColor(candidate, cv2.COLOR_GRAY2BGR)
