import os
import sys
from pathlib import Path


def project_root() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


ROOT = project_root()

YOLOV5_DIR = ROOT / "yolov5"
YOLOV5S_PT = ROOT / "yolov5s.pt"

def find_latest_best(weights_root: Path) -> Path | None:
    candidates = list((weights_root / "runs" / "train").glob("**/weights/best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

WEIGHTS_PT = find_latest_best(YOLOV5_DIR)
if WEIGHTS_PT is None and YOLOV5S_PT.exists():
    WEIGHTS_PT = YOLOV5S_PT
    USING_PRETRAINED = True
else:
    USING_PRETRAINED = False

CAM_INDEX   = 0

CSV_PATH    = ROOT / "prueba" / "coordenadas.csv"
OUTPUT_HTML = ROOT / "prueba" / "mapa_citricos_csv.html"

CLASS_NAMES = ["lemon", "mandarin", "orange", "grapefruit"]
INFER_SIZE  = 416
CONF_THRES  = 0.25
IOU_THRES   = 0.45
TARGET_FPS  = 15
PROCESS_EVERY = 2

SPANISH = {
    "lemon": "Limon",
    "mandarin": "Mandarina",
    "orange": "Naranja",
    "grapefruit": "Toronja",
}

COUNT_LINE_Y_RATIO = 0.60
MAX_MATCH_DIST     = 60
TRACK_TTL          = 12

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def add_yolov5_to_syspath(repo_dir: str):
    repo_dir = os.path.normpath(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

def es(name: str) -> str:
    return SPANISH.get(name, name)

def fmt_pct(x: float) -> int:
    try:
        return int(round(float(x) * 100))
    except Exception:
        return 0
