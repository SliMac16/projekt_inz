from pathlib import Path
from PIL import Image
import csv

SOURCE_ROOT = Path("Zdjecia")
MANIFEST = Path("experiment_manifest_grouped.csv")

with open(
    MANIFEST,
    "r",
    encoding="utf-8-sig"
) as f:
    rows = list(csv.DictReader(f))


print("=" * 80)
print("PEŁNA KONTROLA ODCZYTU OBRAZÓW")
print("=" * 80)

errors = []


for index, row in enumerate(rows, start=1):

    source_id = row["source_id"]

    source_path = (
        SOURCE_ROOT
        / Path(row["source_path"])
    )

    try:

        with Image.open(source_path) as img:

            img.load()

            converted = img.convert("RGB")

            width, height = converted.size

        print(
            f"[OK] {index:03d}/"
            f"{len(rows)} | "
            f"{source_id} | "
            f"{width}x{height}"
        )

    except Exception as e:

        size_bytes = (
            source_path.stat().st_size
            if source_path.exists()
            else None
        )

        errors.append({
            "source_id": source_id,
            "source_path": str(source_path),
            "size_bytes": size_bytes,
            "error": str(e)
        })

        print(
            f"[BŁĄD] "
            f"{source_id} | "
            f"{source_path}"
        )

        print(
            f"        {e}"
        )


print("\n" + "=" * 80)
print("PODSUMOWANIE")
print("=" * 80)

print(
    f"Sprawdzono: {len(rows)}"
)

print(
    f"Błędne:     {len(errors)}"
)


if errors:

    print("\nPROBLEMATYCZNE PLIKI:")

    for item in errors:

        print()
        print(
            f"source_id:   "
            f"{item['source_id']}"
        )

        print(
            f"source_path: "
            f"{item['source_path']}"
        )

        print(
            f"rozmiar:     "
            f"{item['size_bytes']} B"
        )

        print(
            f"błąd:        "
            f"{item['error']}"
        )

else:

    print(
        "\nWszystkie obrazy "
        "zostały odczytane poprawnie."
    )