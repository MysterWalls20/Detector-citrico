# fix_weights.py
import torch, os

SRC = r"C:\Users\OPEN SERVICE EIRL\Documents\UPN\Ciclo 7\Machine learning\Proyecto_Final_Machine Learning\Proyecto Final_Machine learning\yolov5\yolov5\runs\train\exp9\weights\best.pt"   # <- ajusta
DST = os.path.join(os.path.dirname(SRC), "best_linux.pt")

ckpt = torch.load(SRC, map_location="cpu", weights_only=False)

# Limpia campos que a veces guardan rutas/objetos de sistema
for k in ["optimizer", "wandb_id", "train_args", "ema", "updates"]:
    ckpt.pop(k, None)

# Asegura que cualquier YAML/attr sea un dict simple (no objetos raros)
m = ckpt.get("model", None)
if m is not None and hasattr(m, "yaml"):
    try:
        m.yaml = dict(m.yaml)
    except Exception:
        pass

torch.save(ckpt, DST)
print("Guardado:", DST)
