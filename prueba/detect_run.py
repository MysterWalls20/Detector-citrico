# detect_run.py
import argparse
import os
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Ruta a best.pt")
    ap.add_argument("--source", required=True, help="Imagen | Carpeta | Video | 0 (webcam)")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")  # Usa 'cpu' o '0' si tienes GPU
    args = ap.parse_args()

    # Normaliza rutas con espacios
    weights = os.path.normpath(args.weights)
    source = str(args.source)

    # Usar una raw string para evitar problemas con los backslashes en las rutas
    yolov5_dir = r"C:\Users\OPEN SERVICE EIRL\Documents\UPN\Ciclo 7\Machine learning\Proyecto_Final_Machine Learning\Proyecto Final_Machine learning\yolov5"

    cmd = [
        sys.executable, os.path.join(yolov5_dir, "detect.py"),  # Ruta correcta a detect.py
        "--weights", weights,
        "--img", str(args.img),
        "--conf", str(args.conf),
        "--source", str(source),  # Usa la cámara o imagen/archivo de video
        "--device", args.device
    ]
    print(">> Ejecutando:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print("✅ Salidas en: yolov5/runs/detect/exp*/")

if __name__ == "__main__":
    main()



#cd "C:\Users\OPEN SERVICE EIRL\Documents\UPN\Ciclo 7\Machine learning\Proyecto_Final_Machine Learning\Proyecto Final_Machine learning\prueba"

#python detect_run.py --weights "C:/Users/OPEN SERVICE EIRL/Documents/UPN/Ciclo 7/Machine learning/Proyecto_Final_Machine Learning/Proyecto Final_Machine learning/yolov5/runs/train/exp9/weights/best.pt" --source 0 --img 416 --conf 0.25 --device cpu
