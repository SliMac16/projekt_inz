from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter
import hashlib
import csv
import re


SOURCE_ROOT = Path("Zdjecia")
OUTPUT_CSV = Path("master_manifest.csv")

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
}


def calculate_sha256(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            block = f.read(8192)

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


def extract_thickness(filename):

    match = re.search(
        r"(?<!\d)(3|6|12|24)\s*mm",
        filename,
        flags=re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return ""


def extract_magnification(filename):

    match = re.search(
        r"[x×]\s*(\d+)",
        filename,
        flags=re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return ""


print("=" * 70)
print("TWORZENIE GŁÓWNEGO MANIFESTU DATASETU")
print("=" * 70)


rows = []
hash_groups = defaultdict(list)

source_counter = 1


for folder_number in range(1, 12):

    folder_path = SOURCE_ROOT / str(folder_number)

    if not folder_path.exists():

        print(f"UWAGA: brak folderu {folder_path}")
        continue

    files = sorted([
        path
        for path in folder_path.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        )
    ])

    print(
        f"Folder {folder_number}: "
        f"{len(files)} obrazów"
    )

    for file_path in files:

        file_hash = calculate_sha256(file_path)

        with Image.open(file_path) as image:

            width, height = image.size
            mode = image.mode
            channels = len(image.getbands())

        source_id = f"src_{source_counter:04d}"

        row = {
            "source_id": source_id,
            "material_group": folder_number,
            "filename": file_path.name,
            "source_path": str(
                file_path.relative_to(SOURCE_ROOT)
            ),
            "width": width,
            "height": height,
            "mode": mode,
            "channels": channels,
            "thickness_mm": extract_thickness(
                file_path.name
            ),
            "magnification": extract_magnification(
                file_path.name
            ),
            "sha256": file_hash
        }

        rows.append(row)

        hash_groups[file_hash].append(row)

        source_counter += 1


# ------------------------------------------------
# Analiza duplikatów
# ------------------------------------------------

for row in rows:

    duplicates = hash_groups[row["sha256"]]

    duplicate_groups = sorted({
        int(item["material_group"])
        for item in duplicates
    })

    row["duplicate_count"] = len(duplicates)

    row["duplicate_groups"] = "|".join(
        map(str, duplicate_groups)
    )

    row["cross_group_duplicate"] = (
        len(duplicate_groups) > 1
    )

    if row["cross_group_duplicate"]:

        row["use_for_experiment"] = False
        row["exclude_reason"] = (
            "duplicate_across_material_groups"
        )

    else:

        row["use_for_experiment"] = True
        row["exclude_reason"] = ""


# ------------------------------------------------
# Statystyki
# ------------------------------------------------

folder_counts = Counter()
thickness_counts = Counter()
magnification_counts = Counter()

cross_group_duplicates = []

for row in rows:

    folder_counts[row["material_group"]] += 1

    thickness_counts[row["thickness_mm"]] += 1

    magnification_counts[row["magnification"]] += 1

    if row["cross_group_duplicate"]:
        cross_group_duplicates.append(row)


# ------------------------------------------------
# Zapis CSV
# ------------------------------------------------

fieldnames = [
    "source_id",
    "material_group",
    "filename",
    "source_path",
    "width",
    "height",
    "mode",
    "channels",
    "thickness_mm",
    "magnification",
    "sha256",
    "duplicate_count",
    "duplicate_groups",
    "cross_group_duplicate",
    "use_for_experiment",
    "exclude_reason"
]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(rows)


# ------------------------------------------------
# Raport
# ------------------------------------------------

print("\n" + "=" * 70)
print("PODSUMOWANIE")
print("=" * 70)

print(f"\nWszystkie obrazy: {len(rows)}")

usable = sum(
    row["use_for_experiment"]
    for row in rows
)

excluded = len(rows) - usable

print(f"Do wykorzystania: {usable}")
print(f"Wykluczone: {excluded}")


print("\n--- GRUPY MATERIAŁOWE ---")

for group in range(1, 12):

    print(
        f"Folder {group}: "
        f"{folder_counts[group]}"
    )


print("\n--- GRUBOŚCI ---")

for thickness in sorted(
    thickness_counts,
    key=lambda x: (x == "", x)
):

    label = (
        f"{thickness} mm"
        if thickness != ""
        else "brak"
    )

    print(
        f"{label}: "
        f"{thickness_counts[thickness]}"
    )


print("\n--- POWIĘKSZENIA ---")

for magnification in sorted(
    magnification_counts,
    key=lambda x: (x == "", x)
):

    label = (
        f"x{magnification}"
        if magnification != ""
        else "brak informacji"
    )

    print(
        f"{label}: "
        f"{magnification_counts[magnification]}"
    )


print("\n--- DUPLIKATY MIĘDZY GRUPAMI ---")

if cross_group_duplicates:

    displayed_hashes = set()

    for row in cross_group_duplicates:

        file_hash = row["sha256"]

        if file_hash in displayed_hashes:
            continue

        displayed_hashes.add(file_hash)

        group = hash_groups[file_hash]

        print("\nDuplikat:")

        for item in group:

            print(
                f"  folder "
                f"{item['material_group']}: "
                f"{item['filename']}"
            )

else:

    print("Brak.")


print("\n" + "=" * 70)
print(f"Zapisano: {OUTPUT_CSV}")
print("=" * 70)