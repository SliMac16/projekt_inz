import csv
from collections import Counter, defaultdict


INPUT_CSV = "master_manifest.csv"


with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)


def is_true(value):
    return str(value).lower() == "true"


# Tylko obrazy, które przeszły dotychczasowy filtr
usable = [
    row for row in rows
    if is_true(row["use_for_experiment"])
]


explicit_x500 = [
    row for row in usable
    if row["magnification"] == "500"
]

unknown_magnification = [
    row for row in usable
    if row["magnification"] == ""
]

other_magnification = [
    row for row in usable
    if row["magnification"] not in ("", "500")
]


print("=" * 75)
print("AUDYT KANDYDATÓW DO EKSPERYMENTU")
print("=" * 75)

print(f"\nWszystkie rekordy: {len(rows)}")
print(f"Po usunięciu duplikatów między grupami: {len(usable)}")

print(f"\nJawne x500: {len(explicit_x500)}")
print(f"Brak informacji o powiększeniu: {len(unknown_magnification)}")
print(f"Inne powiększenie: {len(other_magnification)}")


print("\n" + "=" * 75)
print("OBRAZY BEZ INFORMACJI O POWIĘKSZENIU")
print("=" * 75)

for row in unknown_magnification:
    print(
        f"grupa={row['material_group']:>2} | "
        f"grubość={row['thickness_mm']:>2} mm | "
        f"{row['filename']}"
    )


print("\n" + "=" * 75)
print("OBRAZY O INNYM POWIĘKSZENIU")
print("=" * 75)

for row in other_magnification:
    print(
        f"grupa={row['material_group']:>2} | "
        f"grubość={row['thickness_mm']:>2} mm | "
        f"x{row['magnification']} | "
        f"{row['filename']}"
    )


# ---------------------------------------------------------
# Rozkład grup materiałowych
# ---------------------------------------------------------

group_counts_all = Counter(
    int(row["material_group"])
    for row in usable
)

group_counts_x500 = Counter(
    int(row["material_group"])
    for row in explicit_x500
)

group_counts_unknown = Counter(
    int(row["material_group"])
    for row in unknown_magnification
)


print("\n" + "=" * 75)
print("ROZKŁAD WEDŁUG GRUP MATERIAŁOWYCH")
print("=" * 75)

print(
    f"{'Grupa':<8}"
    f"{'Po duplikatach':<18}"
    f"{'x500':<10}"
    f"{'brak x':<10}"
)

for group in range(1, 12):

    print(
        f"{group:<8}"
        f"{group_counts_all[group]:<18}"
        f"{group_counts_x500[group]:<10}"
        f"{group_counts_unknown[group]:<10}"
    )


# ---------------------------------------------------------
# Rozkład grubości
# ---------------------------------------------------------

thickness_x500 = Counter(
    int(row["thickness_mm"])
    for row in explicit_x500
)

thickness_unknown = Counter(
    int(row["thickness_mm"])
    for row in unknown_magnification
)


print("\n" + "=" * 75)
print("ROZKŁAD WEDŁUG GRUBOŚCI")
print("=" * 75)

print(
    f"{'Grubość':<12}"
    f"{'x500':<10}"
    f"{'brak x':<10}"
)

for thickness in [3, 6, 12, 24]:

    print(
        f"{str(thickness) + ' mm':<12}"
        f"{thickness_x500[thickness]:<10}"
        f"{thickness_unknown[thickness]:<10}"
    )


# ---------------------------------------------------------
# Macierz grupa × grubość dla x500
# ---------------------------------------------------------

matrix = defaultdict(Counter)

for row in explicit_x500:

    group = int(row["material_group"])
    thickness = int(row["thickness_mm"])

    matrix[group][thickness] += 1


print("\n" + "=" * 75)
print("MACIERZ: GRUPA MATERIAŁOWA x GRUBOŚĆ - TYLKO x500")
print("=" * 75)

print(
    f"{'Grupa':<8}"
    f"{'3 mm':<8}"
    f"{'6 mm':<8}"
    f"{'12 mm':<8}"
    f"{'24 mm':<8}"
    f"{'SUMA':<8}"
)

for group in range(1, 12):

    total = sum(
        matrix[group][thickness]
        for thickness in [3, 6, 12, 24]
    )

    print(
        f"{group:<8}"
        f"{matrix[group][3]:<8}"
        f"{matrix[group][6]:<8}"
        f"{matrix[group][12]:<8}"
        f"{matrix[group][24]:<8}"
        f"{total:<8}"
    )


print("\n" + "=" * 75)
print("WARIANTY DATASETU")
print("=" * 75)

print(
    f"Wariant A - tylko jawne x500: "
    f"{len(explicit_x500)} obrazów"
)

print(
    f"Wariant B - x500 + brak informacji: "
    f"{len(explicit_x500) + len(unknown_magnification)} obrazów"
)

print("=" * 75)