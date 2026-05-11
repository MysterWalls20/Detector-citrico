#!/usr/bin/env python3
"""
EXPORTAR YOLO ENTRENADO A TFLITE
Convierte tu yolov8n-seg.pt (entrenado) a TFLite correctamente
"""

from ultralytics import YOLO
import sys

print("="*70)
print("EXPORTANDO YOLO CUSTOM A TFLITE")
print("="*70)

# ========================================================================
# 1. VERIFICAR MODELO ORIGINAL
# ========================================================================

MODEL_PATH = "yolov8n-seg.pt"

print(f"\n📦 Cargando modelo entrenado: {MODEL_PATH}")

try:
    model = YOLO(MODEL_PATH)
    print("✅ Modelo cargado")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    sys.exit(1)

# Verificar info del modelo
print(f"\n📊 Información del modelo:")
print(f"   Tipo: {model.task}")
print(f"   Clases: {model.names}")

# ========================================================================
# 2. EXPORTAR A TFLITE
# ========================================================================

print("\n🔄 Exportando a TFLite...")
print("   Opciones:")
print("   - Formato: TFLite")
print("   - Precisión: float16 (optimizado para ARM)")
print("   - Tamaño: 640x640 (estándar YOLO)")

try:
    # Exportar con configuración optimizada para Raspberry Pi
    export_path = model.export(
        format='tflite',           # Formato TensorFlow Lite
        half=True,                 # float16 para reducir tamaño
        int8=False,                # NO cuantizar a int8 (pierde precisión)
        imgsz=640,                 # Tamaño de entrada estándar
        optimize=True,             # Optimizaciones adicionales
        simplify=True,             # Simplificar grafo
    )
    
    print(f"✅ Exportado exitosamente a: {export_path}")
    
except Exception as e:
    print(f"❌ Error en exportación: {e}")
    sys.exit(1)

# ========================================================================
# 3. VERIFICAR EXPORTACIÓN
# ========================================================================

import tensorflow as tf
import os

print("\n🔍 Verificando modelo TFLite exportado...")

tflite_file = str(export_path)

if not os.path.exists(tflite_file):
    print(f"❌ No se encontró: {tflite_file}")
    sys.exit(1)

try:
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("✅ Modelo TFLite válido")
    print("\n📋 Detalles del modelo:")
    print(f"   Input:")
    for detail in input_details:
        print(f"      Shape: {detail['shape']}, Type: {detail['dtype']}")
    
    print(f"   Outputs:")
    for i, detail in enumerate(output_details):
        print(f"      Output {i}: Shape {detail['shape']}, Type: {detail['dtype']}")
    
    # Verificar tamaño del archivo
    size_mb = os.path.getsize(tflite_file) / (1024 * 1024)
    print(f"\n📦 Tamaño del archivo: {size_mb:.2f} MB")
    
except Exception as e:
    print(f"❌ Error verificando modelo: {e}")
    sys.exit(1)

# ========================================================================
# 4. INSTRUCCIONES
# ========================================================================

print("\n" + "="*70)
print("✅ EXPORTACIÓN COMPLETADA")
print("="*70)
print(f"\n📁 Archivo generado: {tflite_file}")
print(f"📦 Tamaño: {size_mb:.2f} MB")
print("\n📝 Próximos pasos:")
print(f"   1. Renombra el archivo si quieres:")
print(f"      mv {tflite_file} yolov8n-seg_custom_float16.tflite")
print(f"   2. Usa este modelo en tu script de detección")
print(f"   3. Este modelo SÍ detectará bananas (está entrenado)")
print("="*70)