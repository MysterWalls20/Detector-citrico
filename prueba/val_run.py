# prueba/val_run.py
import argparse, os, subprocess, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Ruta a best.pt")
    ap.add_argument("--data", required=True, help="Ruta a data.yaml")
    ap.add_argument("--img", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    cmd = [
        sys.executable, "yolov5/val.py",
        "--weights", os.path.normpath(args.weights),
        "--data", os.path.normpath(args.data),
        "--imgsz", str(args.img),          # <- nombres que soporta tu val.py
        "--conf-thres", str(args.conf),    # <-
        "--iou-thres", str(args.iou),      # <-
        "--task", "val",
        "--device", args.device,
        "--workers", str(args.workers),
    ]
    print(">> Ejecutando:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("✅ Resultados en: yolov5/runs/val/exp*/")

if __name__ == "__main__":
    main()

# python ".\prueba\val_run.py" `  --weights "yolov5\runs\train\exp7\weights\best.pt" ` --data "dataset\data.yaml" `  --img 416 `  --device cpu