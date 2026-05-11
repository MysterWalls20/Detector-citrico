# Detector de Cítricos con YOLOv5

Sistema de detección y clasificación de cítricos (limón, mandarina, naranja, toronja) usando YOLOv5, con interfaz gráfica para PC y Raspberry Pi.

## Estructura del Proyecto

```
├── app/
│   ├── pc/              # Versión para PC (Windows) con PyQt5
│   │   └── Main_App.py
│   └── raspberry/       # Versión para Raspberry Pi con Tkinter + GPIO
│       └── Main_App_Tk.py
├── dataset/             # Dataset de imágenes (estructura vacía en git)
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
├── prueba/              # Scripts de utilidad y análisis
├── yolov5/              # Submódulo de ultralytics/yolov5
├── train_dataset.py     # Script de entrenamiento
├── requirements.txt     # Dependencias
└── instrucciones.txt    # Instrucciones de instalación (Raspberry Pi)
```

## Clases Detectadas

| Clase     | Nombre en español |
|-----------|-------------------|
| lemon     | Limón             |
| mandarin  | Mandarina         |
| orange    | Naranja           |
| grapefruit| Toronja           |

## Instalación

1. Clonar el repositorio (con submódulos):
```bash
git clone --recursive https://github.com/tu-usuario/tu-repo.git
cd tu-repo
```

2. Crear entorno virtual e instalar dependencias:
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

3. (Opcional) Entrenar modelo:
```bash
python train_dataset.py
```

4. Ejecutar la app según tu plataforma:
   - **PC (Windows):** `python app/pc/Main_App.py`
   - **Raspberry Pi:** `python app/raspberry/Main_App_Tk.py`

## Dataset

El dataset no está incluido en el repositorio por su tamaño. Debes agregar las imágenes en:
- `dataset/train/images/`
- `dataset/valid/images/`
- `dataset/test/images/`

## Licencia

Proyecto académico - Universidad Privada del Norte (UPN)
