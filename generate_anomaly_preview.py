from pathlib import Path
from PIL import (
    Image,
    ImageDraw,
    ImageFilter,
    ImageOps,
    ImageFont,
    ImageStat
)
import csv
import random
import math


# =========================================================
# KONFIGURACJA
# =========================================================

SOURCE_ROOT = Path("Zdjecia")
MANIFEST = Path("experiment_manifest_grouped.csv")

OUTPUT_ROOT = Path("anomaly_preview")

ANOMALY_DIR = OUTPUT_ROOT / "anomalies"
MASK_DIR = OUTPUT_ROOT / "masks"
COMPARISON_DIR = OUTPUT_ROOT / "comparisons"

OUTPUT_CSV = OUTPUT_ROOT / "preview_anomalies.csv"

RANDOM_SEED = 2026
PREVIEW_COUNT = 12


ANOMALY_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


rng = random.Random(RANDOM_SEED)


# =========================================================
# WCZYTANIE MANIFESTU
# =========================================================

with open(MANIFEST, "r", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))


train_rows = [
    row for row in rows
    if row["split"] == "train"
]


print("=" * 80)
print("GENEROWANIE PODGLĄDOWYCH ANOMALII")
print("=" * 80)

print(f"\nObrazów train: {len(train_rows)}")


# =========================================================
# WYBÓR 12 OBRAZÓW
#
# Najpierw po jednym z każdej grupy materiałowej,
# żeby podgląd był możliwie różnorodny.
# =========================================================

selected = []

for material_group in range(1, 12):

    candidates = [
        row for row in train_rows
        if int(row["material_group"]) == material_group
    ]

    if candidates:
        selected.append(rng.choice(candidates))


remaining = [
    row for row in train_rows
    if row["source_id"] not in {
        item["source_id"]
        for item in selected
    }
]


while len(selected) < PREVIEW_COUNT and remaining:

    chosen = rng.choice(remaining)

    selected.append(chosen)
    remaining.remove(chosen)


# =========================================================
# GENEROWANIE NIEREGULARNEJ MASKI
# =========================================================

def create_irregular_mask(width, height):

    min_dim = min(width, height)

    # Wielkość anomalii:
    # około 4.5–10% krótszego wymiaru obrazu
    radius_fraction = rng.uniform(0.045, 0.10)

    base_radius = int(
        min_dim * radius_fraction
    )

    # Anomalia nie jest idealnym kołem
    x_scale = rng.uniform(0.75, 1.30)
    y_scale = rng.uniform(0.75, 1.30)

    # Unikamy samej krawędzi obrazu
    margin = int(base_radius * 1.8)

    center_x = rng.randint(
        margin,
        width - margin
    )

    center_y = rng.randint(
        margin,
        height - margin
    )

    number_of_points = rng.randint(12, 20)

    points = []

    for i in range(number_of_points):

        angle = (
            2 * math.pi
            * i
            / number_of_points
        )

        # Losowa nieregularność brzegu
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

        points.append((x, y))


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


    # Dodajemy 1–3 dodatkowe nieregularne fragmenty,
    # żeby kształt nie wyglądał jak pojedynczy stempel.
    satellite_count = rng.randint(1, 3)

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
            * rng.uniform(0.25, 0.55)
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
            fill=rng.randint(160, 255)
        )


    # Miękka granica anomalii
    blur_px = int(
        min_dim
        * rng.uniform(0.010, 0.022)
    )

    mask = mask.filter(
        ImageFilter.GaussianBlur(
            radius=blur_px
        )
    )


    return (
        mask,
        center_x,
        center_y,
        base_radius,
        radius_fraction,
        blur_px
    )


# =========================================================
# NAKŁADANIE ANOMALII
# =========================================================

def apply_anomaly(image):

    image = image.convert("RGB")

    width, height = image.size


    (
        mask,
        center_x,
        center_y,
        base_radius,
        radius_fraction,
        blur_px
    ) = create_irregular_mask(
        width,
        height
    )


    # =====================================================
    # LOKALNA JASNOŚĆ OBSZARU ANOMALII
    # =====================================================

    grayscale = image.convert("L")

    statistics = ImageStat.Stat(
        grayscale,
        mask=mask
    )

    local_brightness = statistics.mean[0]


    # =====================================================
    # ADAPTACYJNA SIŁA ANOMALII
    #
    # ciemniejsze miejsce -> mocniejsza modyfikacja
    # jaśniejsze miejsce  -> słabsza modyfikacja
    # =====================================================

    darkening_strength = (
        0.68
        - 0.0015 * local_brightness
    )


    # Mała losowa zmienność, żeby anomalie
    # nie miały identycznej intensywności
    darkening_strength += rng.uniform(
        -0.04,
        0.04
    )


    # Bezpieczne ograniczenie
    darkening_strength = max(
        0.35,
        min(
            0.65,
            darkening_strength
        )
    )


    brightness_factor = (
        1.0 - darkening_strength
    )


    darkened = image.point(
        lambda value:
            int(value * brightness_factor)
    )


    anomaly = Image.composite(
        darkened,
        image,
        mask
    )


    parameters = {
        "center_x": center_x,
        "center_y": center_y,
        "radius_px": base_radius,
        "radius_fraction": round(
            radius_fraction,
            5
        ),
        "blur_px": blur_px,

        "local_brightness": round(
            local_brightness,
            2
        ),

        "darkening_strength": round(
            darkening_strength,
            5
        )
    }


    return anomaly, mask, parameters


# =========================================================
# TWORZENIE PLANSZY PORÓWNAWCZEJ
# =========================================================

def create_comparison(
    original,
    anomaly,
    mask,
    source_id
):

    preview_size = (600, 450)

    original_preview = ImageOps.fit(
        original.convert("RGB"),
        preview_size
    )

    anomaly_preview = ImageOps.fit(
        anomaly.convert("RGB"),
        preview_size
    )

    mask_rgb = ImageOps.fit(
        mask.convert("RGB"),
        preview_size
    )


    label_height = 45

    canvas = Image.new(
        "RGB",
        (
            preview_size[0] * 3,
            preview_size[1] + label_height
        ),
        "white"
    )


    canvas.paste(
        original_preview,
        (0, label_height)
    )

    canvas.paste(
        anomaly_preview,
        (
            preview_size[0],
            label_height
        )
    )

    canvas.paste(
        mask_rgb,
        (
            preview_size[0] * 2,
            label_height
        )
    )


    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()


    draw.text(
        (10, 15),
        f"ORYGINAL | {source_id}",
        fill="black",
        font=font
    )

    draw.text(
        (preview_size[0] + 10, 15),
        "SYNTHETIC ANOMALY",
        fill="black",
        font=font
    )

    draw.text(
        (
            preview_size[0] * 2 + 10,
            15
        ),
        "MASK",
        fill="black",
        font=font
    )


    return canvas


# =========================================================
# GENEROWANIE
# =========================================================

metadata = []


for index, row in enumerate(
    selected,
    start=1
):

    source_path = (
        SOURCE_ROOT
        / Path(row["source_path"])
    )


    with Image.open(source_path) as img:

        original = img.convert("RGB")

        anomaly, mask, params = (
            apply_anomaly(original)
        )


    anomaly_filename = (
        f"{row['source_id']}"
        f"__anomaly.jpg"
    )

    mask_filename = (
        f"{row['source_id']}"
        f"__mask.png"
    )

    comparison_filename = (
        f"{row['source_id']}"
        f"__comparison.jpg"
    )


    anomaly_path = (
        ANOMALY_DIR
        / anomaly_filename
    )

    mask_path = (
        MASK_DIR
        / mask_filename
    )

    comparison_path = (
        COMPARISON_DIR
        / comparison_filename
    )


    anomaly.save(
        anomaly_path,
        quality=95
    )

    mask.save(
        mask_path
    )


    comparison = create_comparison(
        original,
        anomaly,
        mask,
        row["source_id"]
    )

    comparison.save(
        comparison_path,
        quality=95
    )


    metadata_row = {
        "preview_id": index,
        "source_id": row["source_id"],
        "material_group": row["material_group"],
        "thickness_mm": row["thickness_mm"],
        "local_brightness": params["local_brightness"],
        "split": row["split"],
        "source_path": row["source_path"],
        "anomaly_path": str(anomaly_path),
        "mask_path": str(mask_path),
        "comparison_path": str(
            comparison_path
        ),
        "center_x": params["center_x"],
        "center_y": params["center_y"],
        "radius_px": params["radius_px"],
        "radius_fraction":
            params["radius_fraction"],
        "blur_px": params["blur_px"],
        "darkening_strength":
            params["darkening_strength"],
        "generator_seed": RANDOM_SEED
    }


    metadata.append(metadata_row)


    print(
        f"[{index:02d}/{len(selected)}] "
        f"{row['source_id']} | "
        f"grupa {row['material_group']} | "
        f"{row['thickness_mm']} mm | "
        f"radius={params['radius_px']} px | "
        f"strength="
        f"{params['darkening_strength']}"
    )


# =========================================================
# ZAPIS PARAMETRÓW
# =========================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=metadata[0].keys()
    )

    writer.writeheader()
    writer.writerows(metadata)


print("\n" + "=" * 80)
print("GOTOWE")
print("=" * 80)

print(
    f"\nWygenerowano: "
    f"{len(metadata)} anomalii"
)

print(
    f"Plansze porównawcze: "
    f"{COMPARISON_DIR}"
)

print(
    f"Parametry: "
    f"{OUTPUT_CSV}"
)

print(
    f"Seed generatora: "
    f"{RANDOM_SEED}"
)