import os
import shutil
from pathlib import Path
import yaml

# === CONFIG ===
# Ruta a la carpeta del dataset (la que contiene train/ valid/ test/)
DATASET_ROOT = Path("dataset")
# Ruta a tu data.yaml para leer id->nombre de clases
DATA_YAML = DATASET_ROOT / "data.yaml"
# Si True no mueve nada, solo imprime lo que haría
DRY_RUN = True
# Si una imagen tiene objetos de varias clases:
#   "mixed" => manda la imagen/etiqueta a subcarpeta 'mixed'
#   "first" => usa la primera clase encontrada en la etiqueta
MULTI_CLASS_POLICY = "mixed"  # "mixed" o "first"

def load_names(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    # En YAML puede venir dict o list; normalizamos a dict {id: name}
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}

def parse_label_file(label_path):
    """Devuelve el conjunto de class_ids presentes en la etiqueta."""
    ids = set()
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cid = int(float(parts[0]))
                ids.add(cid)
            except Exception:
                pass
    return ids

def move_pair(img_path, lbl_path, split_root, class_folder, dry_run=True):
    # p.ej. split_root/images/lemon/xxx.jpg
    dst_img = split_root / "images" / class_folder / img_path.name
    dst_lbl = split_root / "labels" / class_folder / lbl_path.name
    for p in [dst_img.parent, dst_lbl.parent]:
        p.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[DRY] {img_path}  ->  {dst_img}")
        print(f"[DRY] {lbl_path}  ->  {dst_lbl}")
    else:
        shutil.move(str(img_path), str(dst_img))
        shutil.move(str(lbl_path), str(dst_lbl))

def process_split(split_dir, id2name):
    split_dir = Path(split_dir)
    img_root = split_dir / "images"
    lbl_root = split_dir / "labels"

    # Recorremos recursivo por si ya hay subcarpetas
    img_files = list(img_root.rglob("*.*"))
    moved, missing, multi = 0, 0, 0

    for img in img_files:
        if not img.is_file():
            continue
        if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
            continue
        # etiqueta con mismo nombre relativa a labels/**
        rel = img.relative_to(img_root)        # p.ej lemon/xxx.jpg o xxx.jpg
        lbl = (lbl_root / rel).with_suffix(".txt")
        if not lbl.exists():
            # Puede que la etiqueta esté en labels/ plano; probamos por nombre
            lbl_alt = lbl_root / img.stem
            lbl_alt = lbl_alt.with_suffix(".txt")
            if lbl_alt.exists():
                lbl = lbl_alt
            else:
                print(f"[WARN] Falta etiqueta: {lbl}")
                missing += 1
                continue

        class_ids = parse_label_file(lbl)
        if not class_ids:
            print(f"[WARN] Etiqueta vacía (sin objetos): {lbl.name}")
            class_folder = "background"  # opcional, o saltar
        elif len(class_ids) == 1:
            cid = next(iter(class_ids))
            class_folder = id2name.get(cid, f"class_{cid}")
        else:
            multi += 1
            if MULTI_CLASS_POLICY == "mixed":
                class_folder = "mixed"
            else:
                cid = sorted(class_ids)[0]
                class_folder = id2name.get(cid, f"class_{cid}")

        move_pair(img, lbl, split_dir, class_folder, dry_run=DRY_RUN)
        moved += 1

    print(f"\n[{split_dir.name}] movidos={moved}, sin_etiqueta={missing}, multi_clase={multi}")

def main():
    id2name = load_names(DATA_YAML)
    print("Clases:", id2name)

    for split in ["train", "valid", "test"]:
        split_dir = DATASET_ROOT / split
        if split_dir.exists():
            process_split(split_dir, id2name)
        else:
            print(f"[INFO] split no existe: {split_dir}")

    if DRY_RUN:
        print("\n🧪 DRY_RUN=True → NO se movió nada. Cambia DRY_RUN=False para aplicar cambios.")
    else:
        # borrar caches para que YOLO recalcule
        for cache in DATASET_ROOT.rglob("*.cache"):
            try:
                cache.unlink()
            except:
                pass
        print("\n✅ Reorganización hecha. Se eliminaron *.cache.")

if __name__ == "__main__":
    main()
