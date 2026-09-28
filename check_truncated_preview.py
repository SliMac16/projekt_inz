from pathlib import Path
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

FILES = [
    Path(r"Zdjecia\8\Wytop II 12mm x500 - 3.jpg"),
    Path(r"Zdjecia\8\Wytop II 24mm x500 - 3.jpg"),
]

OUTPUT = Path("truncated_preview")
OUTPUT.mkdir(exist_ok=True)

for path in FILES:

    with Image.open(path) as img:
        img.load()
        img = img.convert("RGB")

        output_path = OUTPUT / (
            path.stem + "__decoded.png"
        )

        img.save(output_path)

        print(
            f"{path.name} -> "
            f"{img.size} -> "
            f"{output_path}"
        )

print("\nGotowe.")