# train_dataset.py
# Entrena YOLOv5 sin usar la CLI. Funciona en Windows/Linux/Raspberry.
import os
import sys
import platform
from pathlib import Path

# ---------- Config por defecto (puedes cambiar aquí o por línea de comandos) ----------
DEFAULT_IMG      = 416
DEFAULT_BATCH    = 16           # En Raspberry 5 prueba 8 o 4 si falta RAM
DEFAULT_EPOCHS   = 50
DEFAULT_DATA     = "dataset/data.yaml"
DEFAULT_WEIGHTS  = "yolov5s.pt" # O pon tu best.pt para transfer learning
DEFAULT_DEVICE   = "cpu"        # "cpu" o "0" si hay GPU compatible
DEFAULT_WORKERS  = 2            # En Windows suele ser 0-2; en Raspberry 2-4
DEFAULT_CACHE    = True         # Cachea dataset para acelerar (requiere RAM)
DEFAULT_PROJECT  = None         # None => yolov5/runs/train
DEFAULT_NAME     = "exp"        # Nombre de la carpeta de salida

# ---------- Util: buscar carpeta hacia arriba ----------
def find_upwards(start: Path, name: str) -> Path | None:
    start = start.resolve()
    for base in [start] + list(start.parents):
        candidate = base / name
        if candidate.exists():
            return candidate
    return None

def main():
    import argparse

    # Punto de partida = carpeta desde donde ejecutas este script
    HERE = Path(__file__).resolve().parent

    # Descubre yolov5 y dataset automáticamente
    yolov5_dir = find_upwards(HERE, "yolov5")
    if yolov5_dir is None:
        raise FileNotFoundError("No se encontró la carpeta 'yolov5/'. Coloca el repo en la raíz del proyecto.")

    dataset_dir = find_upwards(HERE, "dataset")  # opcional, solo para resolver DEFAULT_DATA
    if dataset_dir is None:
        # no es obligatorio, pero avisamos
        print("⚠️  No se encontró la carpeta 'dataset/'. Me basaré en la ruta que indiques en --data.")

    parser = argparse.ArgumentParser(description="Entrenamiento YOLOv5 (sin CLI).")
    parser.add_argument("--img", type=int, default=DEFAULT_IMG)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA,
                        help="Ruta a data.yaml (p.ej. dataset/data.yaml)")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                        help="yolov5s.pt | path a best.pt")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--cache", action="store_true", default=DEFAULT_CACHE)
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT,
                        help="Carpeta de salida. Por defecto yolov5/runs/train")
    parser.add_argument("--name", type=str, default=DEFAULT_NAME,
                        help="Nombre del experimento (carpeta)")

    args = parser.parse_args()

    # Ajustes “amigables” por plataforma
    if platform.system() == "Windows" and args.workers > 2:
        print("ℹ️ Windows detectado: ajustando workers a 2 para evitar problemas.")
        args.workers = 2

    # Variables de entorno para que Hub no intente usar git ni moleste con espacios
    os.environ["YOLOV5_NO_GIT"] = "1"
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"

    # Inserta yolov5 en sys.path para importar el módulo
    yolov5_dir = yolov5_dir.resolve()
    if str(yolov5_dir) not in sys.path:
        sys.path.insert(0, str(yolov5_dir))

    # Resuelve rutas relativas
    data_path = Path(args.data)
    if not data_path.is_absolute():
        # si diste algo como "dataset/data.yaml", resuélvelo respecto a la raíz del proyecto
        root = yolov5_dir.parent
        data_path = (root / data_path).resolve()

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        # permite "yolov5s.pt" (del repo) o un best.pt relativo a la raíz
        wp1 = (yolov5_dir / args.weights).resolve()
        wp2 = (yolov5_dir.parent / args.weights).resolve()
        weights_path = wp1 if wp1.exists() else wp2

    if not data_path.exists():
        raise FileNotFoundError(f"No existe el data.yaml: {data_path}")
    if not weights_path.exists():
        print(f"⚠️  No encontré {weights_path}. Entrenaré desde cero (weights='').")
        weights_path = Path("")  # entrenamiento desde cero

    # Directorio de salida
    project_dir = Path(args.project).resolve() if args.project else (yolov5_dir / "runs" / "train").resolve()

    # Muestra la config efectiva
    print("\n=========== CONFIG ENTRENAMIENTO ===========")
    print(f"yolov5_dir : {yolov5_dir}")
    print(f"data       : {data_path}")
    print(f"weights    : {weights_path if str(weights_path) else '(desde cero)'}")
    print(f"img        : {args.img}")
    print(f"batch      : {args.batch}")
    print(f"epochs     : {args.epochs}")
    print(f"device     : {args.device}")
    print(f"workers    : {args.workers}")
    print(f"cache      : {args.cache}")
    print(f"project    : {project_dir}")
    print(f"name       : {args.name}")
    print("===========================================\n")

    # Importa la función oficial de entrenamiento
    # (evita usar la CLI y no necesitas escribir el comando)
    from train import run as yv5_train

    # En Windows conviene spawn
    if platform.system() == "Windows":
        import torch.multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # Llama directamente a la función de entrenamiento
    # Los argumentos son los mismos que usarías en la CLI
    yv5_train(
        imgsz=args.img,
        batch_size=args.batch,
        epochs=args.epochs,
        data=str(data_path),
        weights=str(weights_path),
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,   # permitimos reutilizar nombre
        # Puedes añadir más flags si los usas normalmente:
        # hyp='data/hyps/hyp.scratch-low.yaml',
        # optimizer='SGD',
        # patience=50, ...
    )

if __name__ == "__main__":
    main()

# python yolov5/train.py --img 416 --batch 16 --epochs 50 --data dataset/data.yaml --weights yolov5s.pt --cache --device cpu --workers 2