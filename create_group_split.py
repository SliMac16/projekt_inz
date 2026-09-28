import csv
import random
from collections import Counter, defaultdict


INPUT_CSV = "master_manifest.csv"
OUTPUT_CSV = "experiment_manifest_grouped.csv"

RANDOM_SEED = 42
ITERATIONS = 100000

TARGET_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15
}


def is_true(value):
    return str(value).lower() == "true"


# ---------------------------------------------------------
# Wczytanie danych
# ---------------------------------------------------------

with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))


# Finalne obrazy eksperymentalne:
# - brak duplikatów między grupami
# - tylko jawne x500
eligible = [
    row for row in rows
    if (
        is_true(row["use_for_experiment"])
        and row["magnification"] == "500"
    )
]


# ---------------------------------------------------------
# Tworzenie grup potencjalnie powiązanych
# material_group + thickness
# ---------------------------------------------------------

groups = defaultdict(list)

for row in eligible:

    material_group = int(row["material_group"])
    thickness = int(row["thickness_mm"])

    specimen_group = f"G{material_group:02d}_T{thickness:02d}"

    row["specimen_group"] = specimen_group

    groups[specimen_group].append(row)


group_ids = sorted(groups.keys())


print("=" * 80)
print("GRUPOWY PODZIAŁ TRAIN / VALIDATION / TEST")
print("=" * 80)

print(f"\nLiczba obrazów: {len(eligible)}")
print(f"Liczba grup źródłowych: {len(group_ids)}")


# ---------------------------------------------------------
# Informacje o całym zbiorze
# ---------------------------------------------------------

total_images = len(eligible)

all_material_counts = Counter(
    int(row["material_group"])
    for row in eligible
)

all_thickness_counts = Counter(
    int(row["thickness_mm"])
    for row in eligible
)


# Docelowa liczba grup
# 42 -> około 29 / 6 / 7
n_groups = len(group_ids)

train_group_count = round(n_groups * 0.70)
val_group_count = round(n_groups * 0.15)

test_group_count = (
    n_groups
    - train_group_count
    - val_group_count
)

print(
    f"\nDocelowa liczba grup:"
    f"\ntrain      = {train_group_count}"
    f"\nvalidation = {val_group_count}"
    f"\ntest       = {test_group_count}"
)


# ---------------------------------------------------------
# Funkcje pomocnicze
# ---------------------------------------------------------

def get_rows(group_names):

    result = []

    for group_name in group_names:
        result.extend(groups[group_name])

    return result


def calculate_score(assignments):

    score = 0.0

    split_rows = {
        split: get_rows(group_names)
        for split, group_names in assignments.items()
    }

    # -----------------------------------------------------
    # 1. Liczebność całych splitów
    # -----------------------------------------------------

    for split_name, split_data in split_rows.items():

        actual_ratio = len(split_data) / total_images
        target_ratio = TARGET_RATIOS[split_name]

        difference = actual_ratio - target_ratio

        score += 1000 * (difference ** 2)


    # -----------------------------------------------------
    # 2. Rozkład grubości
    # -----------------------------------------------------

    for split_name, split_data in split_rows.items():

        split_size = len(split_data)

        if split_size == 0:
            return float("inf")

        counts = Counter(
            int(row["thickness_mm"])
            for row in split_data
        )

        for thickness in [3, 6, 12, 24]:

            global_ratio = (
                all_thickness_counts[thickness]
                / total_images
            )

            split_ratio = (
                counts[thickness]
                / split_size
            )

            score += 50 * (
                split_ratio - global_ratio
            ) ** 2


    # -----------------------------------------------------
    # 3. Rozkład grup materiałowych
    # -----------------------------------------------------

    for split_name, split_data in split_rows.items():

        split_size = len(split_data)

        counts = Counter(
            int(row["material_group"])
            for row in split_data
        )

        for material_group in range(1, 12):

            global_ratio = (
                all_material_counts[material_group]
                / total_images
            )

            split_ratio = (
                counts[material_group]
                / split_size
            )

            score += 15 * (
                split_ratio - global_ratio
            ) ** 2


    # -----------------------------------------------------
    # 4. Twarde wymagania
    # -----------------------------------------------------

    train_materials = {
        int(row["material_group"])
        for row in split_rows["train"]
    }

    train_thicknesses = {
        int(row["thickness_mm"])
        for row in split_rows["train"]
    }

    val_thicknesses = {
        int(row["thickness_mm"])
        for row in split_rows["validation"]
    }

    test_thicknesses = {
        int(row["thickness_mm"])
        for row in split_rows["test"]
    }

    # Train powinien znać wszystkie 11 materiałów
    if len(train_materials) < 11:
        score += 10000

    # Train powinien mieć wszystkie grubości
    if len(train_thicknesses) < 4:
        score += 10000

    # Validation i test powinny zawierać wszystkie grubości
    if len(val_thicknesses) < 4:
        score += 5000

    if len(test_thicknesses) < 4:
        score += 5000


    # Validation/test powinny obejmować rozsądną liczbę
    # różnych grup materiałowych.
    val_materials = {
        int(row["material_group"])
        for row in split_rows["validation"]
    }

    test_materials = {
        int(row["material_group"])
        for row in split_rows["test"]
    }

    if len(val_materials) < 5:
        score += 2000

    if len(test_materials) < 5:
        score += 2000


    return score


# ---------------------------------------------------------
# Losowe wyszukiwanie dobrego podziału
# ---------------------------------------------------------

rng = random.Random(RANDOM_SEED)

best_score = float("inf")
best_assignment = None


for iteration in range(ITERATIONS):

    shuffled = group_ids.copy()
    rng.shuffle(shuffled)

    train_groups = shuffled[:train_group_count]

    val_groups = shuffled[
        train_group_count:
        train_group_count + val_group_count
    ]

    test_groups = shuffled[
        train_group_count + val_group_count:
    ]

    assignment = {
        "train": train_groups,
        "validation": val_groups,
        "test": test_groups
    }

    score = calculate_score(assignment)

    if score < best_score:

        best_score = score
        best_assignment = assignment


# ---------------------------------------------------------
# Finalne rekordy
# ---------------------------------------------------------

final_rows = []

for split_name, assigned_groups in best_assignment.items():

    for group_name in assigned_groups:

        for row in groups[group_name]:

            row["split"] = split_name

            final_rows.append(row)


final_rows.sort(
    key=lambda row: row["source_id"]
)


# ---------------------------------------------------------
# Kontrola przecieku grup
# ---------------------------------------------------------

group_to_splits = defaultdict(set)

for row in final_rows:

    group_to_splits[
        row["specimen_group"]
    ].add(row["split"])


leaking_groups = {
    group_name: splits
    for group_name, splits in group_to_splits.items()
    if len(splits) > 1
}


# ---------------------------------------------------------
# Kontrola hashy
# ---------------------------------------------------------

hash_to_splits = defaultdict(set)

for row in final_rows:

    hash_to_splits[
        row["sha256"]
    ].add(row["split"])


leaking_hashes = {
    file_hash: splits
    for file_hash, splits in hash_to_splits.items()
    if len(splits) > 1
}


# ---------------------------------------------------------
# Zapis CSV
# ---------------------------------------------------------

fieldnames = list(final_rows[0].keys())


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
    writer.writerows(final_rows)


# ---------------------------------------------------------
# Raport
# ---------------------------------------------------------

split_rows = {
    split_name: [
        row
        for row in final_rows
        if row["split"] == split_name
    ]
    for split_name in [
        "train",
        "validation",
        "test"
    ]
}


print("\n" + "=" * 80)
print("FINALNY PODZIAŁ")
print("=" * 80)

for split_name in [
    "train",
    "validation",
    "test"
]:

    data = split_rows[split_name]

    group_count = len({
        row["specimen_group"]
        for row in data
    })

    print(
        f"{split_name:<12}: "
        f"{len(data):>3} obrazów | "
        f"{group_count:>2} grup | "
        f"{len(data) / total_images * 100:>6.2f}%"
    )


print("\n" + "=" * 80)
print("GRUPY ŹRÓDŁOWE W SPLITACH")
print("=" * 80)

for split_name in [
    "train",
    "validation",
    "test"
]:

    print(f"\n{split_name.upper()}:")

    assigned = sorted(
        best_assignment[split_name]
    )

    for group_name in assigned:

        print(
            f"  {group_name}: "
            f"{len(groups[group_name])} obrazów"
        )


print("\n" + "=" * 80)
print("ROZKŁAD GRUP MATERIAŁOWYCH")
print("=" * 80)

print(
    f"{'Grupa':<8}"
    f"{'Train':<10}"
    f"{'Val':<10}"
    f"{'Test':<10}"
    f"{'Razem':<10}"
)

for material_group in range(1, 12):

    values = {}

    for split_name, data in split_rows.items():

        values[split_name] = sum(
            1
            for row in data
            if int(row["material_group"])
            == material_group
        )

    total = sum(values.values())

    print(
        f"{material_group:<8}"
        f"{values['train']:<10}"
        f"{values['validation']:<10}"
        f"{values['test']:<10}"
        f"{total:<10}"
    )


print("\n" + "=" * 80)
print("ROZKŁAD GRUBOŚCI")
print("=" * 80)

print(
    f"{'Grubość':<12}"
    f"{'Train':<10}"
    f"{'Val':<10}"
    f"{'Test':<10}"
    f"{'Razem':<10}"
)

for thickness in [3, 6, 12, 24]:

    values = {}

    for split_name, data in split_rows.items():

        values[split_name] = sum(
            1
            for row in data
            if int(row["thickness_mm"])
            == thickness
        )

    total = sum(values.values())

    print(
        f"{str(thickness) + ' mm':<12}"
        f"{values['train']:<10}"
        f"{values['validation']:<10}"
        f"{values['test']:<10}"
        f"{total:<10}"
    )


print("\n" + "=" * 80)
print("KONTROLA DATA LEAKAGE")
print("=" * 80)

print(
    f"Grupy występujące w >1 splicie: "
    f"{len(leaking_groups)}"
)

print(
    f"Identyczne hashe występujące w >1 splicie: "
    f"{len(leaking_hashes)}"
)


if leaking_groups:
    print("\nUWAGA - wykryto przeciek grup:")

    for group_name, splits in leaking_groups.items():
        print(group_name, splits)


if leaking_hashes:
    print("\nUWAGA - wykryto przeciek hashy:")

    for file_hash, splits in leaking_hashes.items():
        print(file_hash, splits)


print("\nNajlepszy score:", round(best_score, 6))

print("\n" + "=" * 80)
print(f"Zapisano: {OUTPUT_CSV}")
print(f"Seed: {RANDOM_SEED}")
print(f"Liczba testowanych podziałów: {ITERATIONS}")
print("=" * 80)