from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageStat
import csv
import random
import math
import hashlib


# =========================================================
# KONFIGURACJA
# =========================================================

SOURCE_ROOT = Path("Zdjecia")
MANIFEST = Path("experiment_manifest_final.csv")

OUTPUT_ROOT = Path("final_dataset")
MASK_ROOT = Path("final_masks")

OUTPUT_MANIFEST = Path("final_dataset_manifest.csv")

MASTER_SEED = 2026

JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0


# =========================================================
# POMOCNICZE
# =========================================================

def get_rng_for_source(source_id):
    """
    Tworzy deterministyczny generator losowy dla konkretnego obrazu.
    Ten sam source_id + MASTER_SEED zawsze da tę samą anomalię.
    """

    seed_text = f"{MASTER_SEED}:{source_id}"

    digest = hashlib.sha256(
        seed_text.encode("utf-8")
    ).digest()

    seed = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False
    )

    return random.Random(seed), seed


# =========================================================
# MASKA ANOMALII
# =========================================================

def create_irregular_mask(width, height, rng):

    min_dim = min(width, height)

    radius_fraction = rng.uniform(
        0.045,
        0.10
    )

    base_radius = int(
        min_dim * radius_fraction
    )

    x_scale = rng.uniform(
        0.75,
        1.30
    )

    y_scale = rng.uniform(
        0.75,
        1.30
    )

    margin = int(
        base_radius * 1.8
    )

    center_x = rng.randint(
        margin,
        width - margin
    )

    center_y = rng.randint(
        margin,
        height - margin
    )

    number_of_points = rng.randint(
        12,
        20
    )

    points = []

    for i in range(number_of_points):

        angle = (
            2 * math.pi
            * i
            / number_of_points
        )

        radius_noise = rng.uniform(
            0.60,
            1.35
        )

        radius = (
            base_radius
            * radius_noise
        )

        x = (
            center_x
            + math.cos(angle)
            * radius
            * x_scale
        )

        y = (
            center_y
            + math.sin(angle)
            * radius
            * y_scale
        )

        points.append(
            (x, y)
        )


    mask = Image.new(
        "L",
        (width, height),
        0
    )

    draw = ImageDraw.Draw(mask)

    draw.polygon(
        points,
        fill=255
    )


    satellite_count = rng.randint(
        1,
        3
    )

    for _ in range(satellite_count):

        offset_x = rng.randint(
            -base_radius,
            base_radius
        )

        offset_y = rng.randint(
            -base_radius,
            base_radius
        )

        satellite_radius = int(
            base_radius
            * rng.uniform(
                0.25,
                0.55
            )
        )

        sx = center_x + offset_x
        sy = center_y + offset_y

        draw.ellipse(
            [
                sx - satellite_radius,
                sy - satellite_radius,
                sx + satellite_radius,
                sy + satellite_radius
            ],
            fill=rng.randint(
                160,
                255
            )
        )


    blur_px = max(
        1,
        int(
            min_dim
            * rng.uniform(
                0.010,
                0.022
            )
        )
    )

    mask = mask.filter(
        ImageFilter.GaussianBlur(
            radius=blur_px
        )
    )


    parameters = {
        "center_x": center_x,
        "center_y": center_y,
        "radius_px": base_radius,
        "radius_fraction": radius_fraction,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "number_of_points": number_of_points,
        "satellite_count": satellite_count,
        "blur_px": blur_px
    }


    return mask, parameters


# =========================================================
# GENEROWANIE ANOMALII
# =========================================================

def apply_anomaly(image, rng):

    image = image.convert("RGB")

    width, height = image.size

    mask, params = create_irregular_mask(
        width,
        height,
        rng
    )


    # -----------------------------------------------------
    # Lokalna jasność
    # -----------------------------------------------------

    grayscale = image.convert("L")

    statistics = ImageStat.Stat(
        grayscale,
        mask=mask
    )

    local_brightness = (
        statistics.mean[0]
    )


    # -----------------------------------------------------
    # Adaptacyjna siła przyciemnienia
    # -----------------------------------------------------

    darkening_strength = (
        0.68
        - 0.0015
        * local_brightness
    )

    darkening_strength += rng.uniform(
        -0.04,
        0.04
    )

    darkening_strength = max(
        0.35,
        min(
            0.65,
            darkening_strength
        )
    )


    brightness_factor = (
        1.0
        - darkening_strength
    )


    darkened = image.point(
        lambda value:
            int(
                value
                * brightness_factor
            )
    )


    anomaly = Image.composite(
        darkened,
        image,
        mask
    )


    params["local_brightness"] = (
        local_brightness
    )

    params["darkening_strength"] = (
        darkening_strength
    )


    return anomaly, mask, params


# =========================================================
# WCZYTANIE MANIFESTU
# =========================================================

with open(
    MANIFEST,
    "r",
    encoding="utf-8-sig"
) as f:

    rows = list(
        csv.DictReader(f)
    )


print("=" * 80)
print("GENEROWANIE FINALNEGO ZBIORU")
print("=" * 80)

print(
    f"\nLiczba obrazów źródłowych: "
    f"{len(rows)}"
)


# =========================================================
# UTWORZENIE KATALOGÓW
# =========================================================

splits = [
    "train",
    "validation",
    "test"
]

classes = [
    "clean",
    "anomaly"
]


for split in splits:

    for class_name in classes:

        (
            OUTPUT_ROOT
            / split
            / class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    (
        MASK_ROOT
        / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# GENEROWANIE ZBIORU
# =========================================================

final_rows = []


for index, row in enumerate(
    rows,
    start=1
):

    source_id = row["source_id"]
    split = row["split"]

    source_path = (
        SOURCE_ROOT
        / Path(
            row["source_path"]
        )
    )


    if not source_path.exists():

        raise FileNotFoundError(
            f"Nie znaleziono pliku: "
            f"{source_path}"
        )


    rng, image_seed = (
        get_rng_for_source(
            source_id
        )
    )


    with Image.open(
        source_path
    ) as img:

        original = img.convert(
            "RGB"
        )

        anomaly, mask, params = (
            apply_anomaly(
                original,
                rng
            )
        )


    # =====================================================
    # ŚCIEŻKI WYJŚCIOWE
    # =====================================================

    clean_filename = (
        f"{source_id}"
        f"__clean.jpg"
    )

    anomaly_filename = (
        f"{source_id}"
        f"__anomaly.jpg"
    )

    mask_filename = (
        f"{source_id}"
        f"__mask.png"
    )


    clean_path = (
        OUTPUT_ROOT
        / split
        / "clean"
        / clean_filename
    )

    anomaly_path = (
        OUTPUT_ROOT
        / split
        / "anomaly"
        / anomaly_filename
    )

    mask_path = (
        MASK_ROOT
        / split
        / mask_filename
    )


    # =====================================================
    # IDENTYCZNY SPOSÓB ZAPISU CLEAN I ANOMALY
    # =====================================================

    original.save(
        clean_path,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=JPEG_SUBSAMPLING,
        optimize=False
    )

    anomaly.save(
        anomaly_path,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=JPEG_SUBSAMPLING,
        optimize=False
    )

    mask.save(
        mask_path,
        format="PNG"
    )


    # =====================================================
    # REKORD CLEAN
    # =====================================================

    clean_row = {
        "sample_id":
            f"{source_id}__clean",

        "source_id":
            source_id,

        "label":
            0,

        "class_name":
            "clean",

        "split":
            split,

        "material_group":
            row["material_group"],

        "thickness_mm":
            row["thickness_mm"],

        "magnification":
            row["magnification"],

        "source_path":
            row["source_path"],

        "output_path":
            str(clean_path),

        "mask_path":
            "",

        "source_sha256":
            row.get(
                "sha256",
                ""
            ),

        "generator_seed":
            "",

        "center_x":
            "",

        "center_y":
            "",

        "radius_px":
            "",

        "radius_fraction":
            "",

        "x_scale":
            "",

        "y_scale":
            "",

        "number_of_points":
            "",

        "satellite_count":
            "",

        "blur_px":
            "",

        "local_brightness":
            "",

        "darkening_strength":
            ""
    }


    # =====================================================
    # REKORD ANOMALY
    # =====================================================

    anomaly_row = {
        "sample_id":
            f"{source_id}__anomaly",

        "source_id":
            source_id,

        "label":
            1,

        "class_name":
            "anomaly",

        "split":
            split,

        "material_group":
            row["material_group"],

        "thickness_mm":
            row["thickness_mm"],

        "magnification":
            row["magnification"],

        "source_path":
            row["source_path"],

        "output_path":
            str(anomaly_path),

        "mask_path":
            str(mask_path),

        "source_sha256":
            row.get(
                "sha256",
                ""
            ),

        "generator_seed":
            image_seed,

        "center_x":
            params["center_x"],

        "center_y":
            params["center_y"],

        "radius_px":
            params["radius_px"],

        "radius_fraction":
            round(
                params[
                    "radius_fraction"
                ],
                6
            ),

        "x_scale":
            round(
                params["x_scale"],
                6
            ),

        "y_scale":
            round(
                params["y_scale"],
                6
            ),

        "number_of_points":
            params[
                "number_of_points"
            ],

        "satellite_count":
            params[
                "satellite_count"
            ],

        "blur_px":
            params["blur_px"],

        "local_brightness":
            round(
                params[
                    "local_brightness"
                ],
                4
            ),

        "darkening_strength":
            round(
                params[
                    "darkening_strength"
                ],
                6
            )
    }


    final_rows.append(
        clean_row
    )

    final_rows.append(
        anomaly_row
    )


    print(
        f"[{index:03d}/{len(rows)}] "
        f"{source_id} | "
        f"{split}"
    )


# =========================================================
# ZAPIS FINALNEGO MANIFESTU
# =========================================================

with open(
    OUTPUT_MANIFEST,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            final_rows[0].keys()
    )

    writer.writeheader()

    writer.writerows(
        final_rows
    )


# =========================================================
# WALIDACJA
# =========================================================

print("\n" + "=" * 80)
print("KONTROLA FINALNEGO ZBIORU")
print("=" * 80)


expected_source_count = len(rows)

unique_sources = {
    row["source_id"]
    for row in final_rows
}

if len(unique_sources) != expected_source_count:

    raise RuntimeError(
        "Błąd liczby unikalnych "
        "source_id."
    )


# Każdy obraz źródłowy musi mieć:
# clean + anomaly

for source_id in unique_sources:

    source_records = [
        row
        for row in final_rows
        if row["source_id"]
        == source_id
    ]

    labels = {
        row["class_name"]
        for row
        in source_records
    }

    record_splits = {
        row["split"]
        for row
        in source_records
    }

    if len(source_records) != 2:

        raise RuntimeError(
            f"{source_id}: "
            f"nie ma dokładnie "
            f"2 rekordów."
        )

    if labels != {
        "clean",
        "anomaly"
    }:

        raise RuntimeError(
            f"{source_id}: "
            f"nieprawidłowe klasy."
        )

    if len(record_splits) != 1:

        raise RuntimeError(
            f"{source_id}: "
            f"clean i anomaly "
            f"są w różnych splitach."
        )


# =========================================================
# STATYSTYKI
# =========================================================

for split in splits:

    clean_count = sum(
        1
        for row in final_rows
        if (
            row["split"] == split
            and
            row["class_name"]
            == "clean"
        )
    )

    anomaly_count = sum(
        1
        for row in final_rows
        if (
            row["split"] == split
            and
            row["class_name"]
            == "anomaly"
        )
    )

    print(
        f"{split:12s}: "
        f"{clean_count:3d} clean | "
        f"{anomaly_count:3d} anomaly | "
        f"{clean_count + anomaly_count:3d} razem"
    )


print("\n" + "-" * 80)

print(
    f"Obrazy źródłowe: "
    f"{len(rows)}"
)

print(
    f"Rekordy finalne: "
    f"{len(final_rows)}"
)

print(
    f"Manifest: "
    f"{OUTPUT_MANIFEST}"
)

print(
    f"Dataset: "
    f"{OUTPUT_ROOT}"
)

print(
    f"Maski: "
    f"{MASK_ROOT}"
)

print(
    f"MASTER_SEED: "
    f"{MASTER_SEED}"
)

print("\nWALIDACJA ZAKOŃCZONA POPRAWNIE.")