import csv
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split


INPUT_CSV = "master_manifest.csv"
OUTPUT_CSV = "experiment_manifest.csv"

RANDOM_SEED = 42


def is_true(value):
    return str(value).lower() == "true"


# ---------------------------------------------------------
# Wczytanie manifestu
# ---------------------------------------------------------

with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))


# ---------------------------------------------------------
# Finalny filtr eksperymentu:
#
# 1. brak duplikatów między grupami
# 2. tylko jawne x500
# ---------------------------------------------------------

eligible = [
    row
    for row in rows
    if (
        is_true(row["use_for_experiment"])
        and row["magnification"] == "500"
    )
]


print("=" * 75)
print("TWORZENIE PODZIAŁU TRAIN / VALIDATION / TEST")
print("=" * 75)

print(f"\nLiczba obrazów wejściowych: {len(rows)}")
print(f"Liczba obrazów wykorzystanych: {len(eligible)}")


# ---------------------------------------------------------
# Pierwszy podział:
#
# 70% train
# 30% temp
# ---------------------------------------------------------

groups = [
    row["material_group"]
    for row in eligible
]

train_rows, temp_rows = train_test_split(
    eligible,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=groups
)


# ---------------------------------------------------------
# Drugi podział:
#
# temp dzielimy 50/50
# czyli około 15% validation i 15% test
# ---------------------------------------------------------

temp_groups = [
    row["material_group"]
    for row in temp_rows
]

val_rows, test_rows = train_test_split(
    temp_rows,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_groups
)


# ---------------------------------------------------------
# Przypisanie splitu
# ---------------------------------------------------------

for row in train_rows:
    row["split"] = "train"

for row in val_rows:
    row["split"] = "validation"

for row in test_rows:
    row["split"] = "test"


all_experiment_rows = (
    train_rows
    + val_rows
    + test_rows
)


# Sortowanie po source_id
all_experiment_rows.sort(
    key=lambda row: row["source_id"]
)


# ---------------------------------------------------------
# Zapis
# ---------------------------------------------------------

fieldnames = list(all_experiment_rows[0].keys())

if "split" not in fieldnames:
    fieldnames.append("split")


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_experiment_rows)


# ---------------------------------------------------------
# Raport
# ---------------------------------------------------------

splits = {
    "train": train_rows,
    "validation": val_rows,
    "test": test_rows
}


print("\n" + "=" * 75)
print("LICZEBNOŚĆ SPLITÓW")
print("=" * 75)

for name, split_rows in splits.items():

    percentage = (
        len(split_rows)
        / len(eligible)
        * 100
    )

    print(
        f"{name:<12}: "
        f"{len(split_rows):>3} "
        f"({percentage:.2f}%)"
    )


# ---------------------------------------------------------
# Rozkład grup
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("ROZKŁAD GRUP MATERIAŁOWYCH")
print("=" * 75)

print(
    f"{'Grupa':<8}"
    f"{'Train':<10}"
    f"{'Val':<10}"
    f"{'Test':<10}"
    f"{'Razem':<10}"
)

for group in range(1, 12):

    values = {}

    for split_name, split_rows in splits.items():

        values[split_name] = sum(
            1
            for row in split_rows
            if int(row["material_group"]) == group
        )

    total = sum(values.values())

    print(
        f"{group:<8}"
        f"{values['train']:<10}"
        f"{values['validation']:<10}"
        f"{values['test']:<10}"
        f"{total:<10}"
    )


# ---------------------------------------------------------
# Rozkład grubości
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("ROZKŁAD GRUBOŚCI")
print("=" * 75)

print(
    f"{'Grubość':<12}"
    f"{'Train':<10}"
    f"{'Val':<10}"
    f"{'Test':<10}"
    f"{'Razem':<10}"
)

for thickness in [3, 6, 12, 24]:

    counts = {}

    for split_name, split_rows in splits.items():

        counts[split_name] = sum(
            1
            for row in split_rows
            if int(row["thickness_mm"]) == thickness
        )

    total = sum(counts.values())

    print(
        f"{str(thickness) + ' mm':<12}"
        f"{counts['train']:<10}"
        f"{counts['validation']:<10}"
        f"{counts['test']:<10}"
        f"{total:<10}"
    )


# ---------------------------------------------------------
# Grupa x grubość x split
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("SZCZEGÓŁOWY ROZKŁAD: GRUPA / GRUBOŚĆ / SPLIT")
print("=" * 75)

for group in range(1, 12):

    print(f"\nGrupa {group}")

    for thickness in [3, 6, 12, 24]:

        train_count = sum(
            1 for row in train_rows
            if (
                int(row["material_group"]) == group
                and int(row["thickness_mm"]) == thickness
            )
        )

        val_count = sum(
            1 for row in val_rows
            if (
                int(row["material_group"]) == group
                and int(row["thickness_mm"]) == thickness
            )
        )

        test_count = sum(
            1 for row in test_rows
            if (
                int(row["material_group"]) == group
                and int(row["thickness_mm"]) == thickness
            )
        )

        total = (
            train_count
            + val_count
            + test_count
        )

        if total > 0:

            print(
                f"  {thickness:>2} mm | "
                f"train={train_count:<2} "
                f"val={val_count:<2} "
                f"test={test_count:<2} "
                f"total={total}"
            )


print("\n" + "=" * 75)
print(f"Zapisano: {OUTPUT_CSV}")
print(f"Seed losowania: {RANDOM_SEED}")
print("=" * 75)