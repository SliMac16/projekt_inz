from pathlib import Path
import csv
import hashlib
from collections import defaultdict, Counter


DATASET_DIR = Path("dataset")
SOURCE_ROOT = Path("Zdjecia")
INPUT_CSV = Path("dataset_inventory.csv")
OUTPUT_CSV = Path("dataset_inventory_mapped.csv")

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


print("=" * 70)
print("MAPOWANIE OBRAZÓW DO ORYGINALNYCH FOLDERÓW")
print("=" * 70)

# hash -> lista folderów źródłowych
source_hashes = defaultdict(list)

for folder_number in range(1, 12):

    folder_path = SOURCE_ROOT / str(folder_number)

    if not folder_path.exists():
        print(f"UWAGA: brak folderu {folder_path}")
        continue

    files = [
        path for path in folder_path.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]

    print(f"Folder {folder_number}: {len(files)} obrazów")

    for file_path in files:

        file_hash = calculate_sha256(file_path)

        source_hashes[file_hash].append({
            "folder": folder_number,
            "filename": file_path.name,
            "path": str(file_path)
        })


with open(INPUT_CSV, "r", encoding="utf-8-sig") as csv_file:
    reader = csv.DictReader(csv_file)
    rows = list(reader)


mapped_count = 0
unmapped_count = 0
ambiguous_count = 0

folder_counts = Counter()

unmapped_files = []
ambiguous_files = []


for row in rows:

    file_hash = row["sha256"]

    matches = source_hashes.get(file_hash, [])

    if len(matches) == 1:

        match = matches[0]

        row["material_group"] = match["folder"]
        row["source_original_filename"] = match["filename"]
        row["mapping_status"] = "MATCHED"

        mapped_count += 1
        folder_counts[match["folder"]] += 1

    elif len(matches) == 0:

        row["material_group"] = ""
        row["source_original_filename"] = ""
        row["mapping_status"] = "NOT_FOUND"

        unmapped_count += 1
        unmapped_files.append(row["filename"])

    else:

        folders = sorted(set(match["folder"] for match in matches))

        row["material_group"] = ",".join(map(str, folders))
        row["source_original_filename"] = "|".join(
            match["filename"] for match in matches
        )
        row["mapping_status"] = "AMBIGUOUS"

        ambiguous_count += 1

        ambiguous_files.append({
            "filename": row["filename"],
            "folders": folders,
            "matches": matches
        })


fieldnames = list(rows[0].keys())

if "source_original_filename" not in fieldnames:
    fieldnames.append("source_original_filename")

if "mapping_status" not in fieldnames:
    fieldnames.append("mapping_status")


with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as csv_file:

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(rows)


print("\n" + "=" * 70)
print("WYNIKI MAPOWANIA")
print("=" * 70)

print(f"\nPoprawnie przypisane: {mapped_count}")
print(f"Nieznalezione: {unmapped_count}")
print(f"Niejednoznaczne: {ambiguous_count}")


print("\n--- LICZBA OBRAZÓW W POSZCZEGÓLNYCH FOLDERACH ---")

for folder_number in range(1, 12):
    print(
        f"Folder {folder_number}: "
        f"{folder_counts.get(folder_number, 0)}"
    )


if unmapped_files:

    print("\n--- NIEZNALEZIONE OBRAZY ---")

    for filename in unmapped_files:
        print(filename)


if ambiguous_files:

    print("\n--- NIEJEDNOZNACZNE DOPASOWANIA ---")

    for item in ambiguous_files:

        print(f"\nPlik: {item['filename']}")
        print(f"Foldery: {item['folders']}")

        for match in item["matches"]:
            print(
                f"  folder {match['folder']} -> "
                f"{match['filename']}"
            )


print("\n" + "=" * 70)
print(f"Zapisano: {OUTPUT_CSV}")
print("=" * 70)