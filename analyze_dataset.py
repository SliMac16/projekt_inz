from pathlib import Path
from PIL import Image
from collections import Counter, defaultdict
import hashlib
import csv


DATASET_DIR = Path("dataset")
OUTPUT_CSV = Path("dataset_inventory.csv")

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff"
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


files = sorted([
    path for path in DATASET_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
])


print("=" * 60)
print("ANALIZA DATASETU")
print("=" * 60)

print(f"\nLiczba znalezionych obrazów: {len(files)}")


rows = []

extensions = Counter()
dimensions = Counter()
modes = Counter()
formats = Counter()

hash_groups = defaultdict(list)

errors = []


for index, file_path in enumerate(files, start=1):

    source_id = f"img_{index:04d}"

    try:
        # Sprawdzenie, czy plik można poprawnie otworzyć
        with Image.open(file_path) as image:
            image.verify()

        # Ponowne otwarcie w celu pobrania parametrów
        with Image.open(file_path) as image:
            width, height = image.size
            mode = image.mode
            image_format = image.format
            channels = len(image.getbands())

        file_hash = calculate_sha256(file_path)

        file_size = file_path.stat().st_size

        extensions[file_path.suffix.lower()] += 1
        dimensions[(width, height)] += 1
        modes[mode] += 1
        formats[image_format] += 1

        hash_groups[file_hash].append(file_path.name)

        rows.append({
            "source_id": source_id,
            "filename": file_path.name,
            "relative_path": str(file_path.relative_to(DATASET_DIR)),
            "extension": file_path.suffix.lower(),
            "width": width,
            "height": height,
            "mode": mode,
            "channels": channels,
            "format": image_format,
            "file_size_bytes": file_size,
            "sha256": file_hash,
            "status": "OK",
            "error": "",
            "material_group": "",
            "specimen_id": "",
            "thickness_mm": "",
            "split": ""
        })

    except Exception as error:

        errors.append((file_path.name, str(error)))

        rows.append({
            "source_id": source_id,
            "filename": file_path.name,
            "relative_path": str(file_path.relative_to(DATASET_DIR)),
            "extension": file_path.suffix.lower(),
            "width": "",
            "height": "",
            "mode": "",
            "channels": "",
            "format": "",
            "file_size_bytes": file_path.stat().st_size,
            "sha256": "",
            "status": "ERROR",
            "error": str(error),
            "material_group": "",
            "specimen_id": "",
            "thickness_mm": "",
            "split": ""
        })


fieldnames = [
    "source_id",
    "filename",
    "relative_path",
    "extension",
    "width",
    "height",
    "mode",
    "channels",
    "format",
    "file_size_bytes",
    "sha256",
    "status",
    "error",
    "material_group",
    "specimen_id",
    "thickness_mm",
    "split"
]


with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as csv_file:

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(rows)


print("\n--- ROZSZERZENIA ---")

for extension, count in extensions.items():
    print(f"{extension}: {count}")


print("\n--- ROZMIARY OBRAZÓW ---")

for dimension, count in dimensions.most_common():
    print(f"{dimension[0]} x {dimension[1]}: {count}")


print("\n--- TRYBY KOLORÓW ---")

for mode, count in modes.items():
    print(f"{mode}: {count}")


print("\n--- FORMATY ---")

for image_format, count in formats.items():
    print(f"{image_format}: {count}")


print("\n--- USZKODZONE / NIEPOPRAWNE PLIKI ---")

if errors:
    for filename, error in errors:
        print(f"{filename}: {error}")
else:
    print("Brak.")


print("\n--- IDENTYCZNE PLIKI ---")

duplicate_groups = [
    filenames
    for filenames in hash_groups.values()
    if len(filenames) > 1
]

if duplicate_groups:

    for group_number, filenames in enumerate(duplicate_groups, start=1):

        print(f"\nGrupa {group_number}:")

        for filename in filenames:
            print(f"  {filename}")

else:
    print("Nie znaleziono identycznych plików.")


print("\n--- PIERWSZE 40 NAZW PLIKÓW ---")

for file_path in files[:40]:
    print(file_path.name)


print("\n" + "=" * 60)
print(f"Zapisano: {OUTPUT_CSV}")
print("=" * 60)