import csv

INPUT_MANIFEST = "experiment_manifest_grouped.csv"
OUTPUT_MANIFEST = "experiment_manifest_final.csv"

EXCLUDED_SOURCE_IDS = {
    "src_0153",
    "src_0157"
}


with open(
    INPUT_MANIFEST,
    "r",
    encoding="utf-8-sig"
) as f:

    rows = list(csv.DictReader(f))


filtered_rows = [
    row for row in rows
    if row["source_id"]
    not in EXCLUDED_SOURCE_IDS
]


with open(
    OUTPUT_MANIFEST,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(filtered_rows)


print("=" * 70)
print("FINALNY MANIFEST EKSPERYMENTU")
print("=" * 70)

print(f"Wejściowo:     {len(rows)}")
print(f"Wykluczono:    {len(rows) - len(filtered_rows)}")
print(f"Finalnie:      {len(filtered_rows)}")

print("\nLiczebność splitów:")

for split in [
    "train",
    "validation",
    "test"
]:

    count = sum(
        1
        for row in filtered_rows
        if row["split"] == split
    )

    print(
        f"{split:12s}: {count}"
    )


print(
    f"\nZapisano: "
    f"{OUTPUT_MANIFEST}"
)