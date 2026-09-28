from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont
import csv


SOURCE_ROOT = Path("Zdjecia")
MANIFEST = Path("master_manifest.csv")
OUTPUT_DIR = Path("magnification_review")

GROUPS_TO_REVIEW = [1, 3, 4]

CELL_W = 500
IMAGE_H = 360
LABEL_H = 70
CELL_H = IMAGE_H + LABEL_H
COLUMNS = 3


OUTPUT_DIR.mkdir(exist_ok=True)


with open(MANIFEST, "r", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))


def mag_label(row):
    if row["magnification"] == "":
        return "BRAK INFO"

    return f"x{row['magnification']}"


def sort_key(row):
    thickness = int(row["thickness_mm"])

    if row["magnification"] == "":
        mag_order = 0
    elif row["magnification"] == "500":
        mag_order = 1
    else:
        mag_order = 2

    return (
        thickness,
        mag_order,
        row["filename"]
    )


font = ImageFont.load_default()


for group_number in GROUPS_TO_REVIEW:

    group_rows = [
        row for row in rows
        if (
            int(row["material_group"]) == group_number
            and row["cross_group_duplicate"].lower() != "true"
        )
    ]

    group_rows.sort(key=sort_key)

    row_count = (
        len(group_rows) + COLUMNS - 1
    ) // COLUMNS

    sheet = Image.new(
        "RGB",
        (
            COLUMNS * CELL_W,
            row_count * CELL_H
        ),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for i, row in enumerate(group_rows):

        col = i % COLUMNS
        r = i // COLUMNS

        x = col * CELL_W
        y = r * CELL_H

        image_path = SOURCE_ROOT / row["source_path"]

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            preview = ImageOps.contain(
                image,
                (CELL_W - 20, IMAGE_H - 20)
            )

        px = (
            x
            + (CELL_W - preview.width) // 2
        )

        py = (
            y
            + (IMAGE_H - preview.height) // 2
        )

        sheet.paste(
            preview,
            (px, py)
        )

        label = (
            f"{row['source_id']} | "
            f"{row['thickness_mm']} mm | "
            f"{mag_label(row)}\n"
            f"{row['filename']}"
        )

        draw.text(
            (x + 10, y + IMAGE_H + 5),
            label,
            fill="black",
            font=font
        )

        draw.rectangle(
            [
                x,
                y,
                x + CELL_W - 1,
                y + CELL_H - 1
            ],
            outline="black"
        )

    output_path = (
        OUTPUT_DIR
        / f"group_{group_number}_review.jpg"
    )

    sheet.save(
        output_path,
        quality=95
    )

    print(
        f"Zapisano: {output_path}"
    )


print("\nGotowe.")
print(
    "Sprawdź plansze dla grup 1, 3 i 4."
)