# app/main.py
import os, sys, time
import cv2, psutil, torch
import numpy as np

from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal, QThread, QUrl
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QProgressBar, QMessageBox
)

try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    HAS_WEB = True
except Exception:
    HAS_WEB = False

import webbrowser
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from time import perf_counter
from collections import deque


# =========================
# CONFIG:
# =========================


def project_root() -> Path:
    """
    Devuelve la carpeta raíz del proyecto.
    - En desarrollo: .../Proyecto_Final_Machine learning
    - En ejecutable (PyInstaller): carpeta temporal _MEIPASS
    """
    # Si empacas con PyInstaller
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore
    # app/main.py -> subir dos niveles: app/  -->  raiz del repo
    return Path(__file__).resolve().parents[1]

ROOT = project_root()

# YOLOv5 (repo local) y pesos
YOLOV5_DIR = ROOT / "yolov5"

def find_latest_best(weights_root: Path) -> Path:
    """
    Busca el best.pt más reciente dentro de runs/train/**/weights/best.pt
    """
    candidates = list((weights_root / "runs" / "train").glob("**/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError("No se encontró ningún best.pt en yolov5/runs/train/**/weights/")
    return max(candidates, key=lambda p: p.stat().st_mtime)

WEIGHTS_PT = find_latest_best(YOLOV5_DIR)


CAM_INDEX   = 0  # 0 cámara integrada; 1/2 USB externas

# Archivos de mapa/CSV (carpeta 'prueba')
CSV_PATH    = ROOT / "prueba" / "coordenadas.csv"
OUTPUT_HTML = ROOT / "prueba" / "mapa_citricos_csv.html"


CLASS_NAMES = ["lemon", "mandarin", "orange", "grapefruit"]
INFER_SIZE  = 416     # tamaño de entrada a la red
CONF_THRES  = 0.25
IOU_THRES   = 0.45
TARGET_FPS  = 15      # limita FPS reales -> menos CPU
PROCESS_EVERY = 2     # procesa 1 de cada N frames (2 = 30→15 fps de inferencia)


SPANISH = {
    "lemon": "Limon",
    "mandarin": "Mandarina",
    "orange": "Naranja",
    "grapefruit": "Toronja",
}

# ---- Conteo por línea (banda) ----
COUNT_LINE_Y_RATIO = 0.60   # 60% de la altura del frame
MAX_MATCH_DIST     = 60     # px para asociar detección a track previo
TRACK_TTL          = 12     # frames que un track puede estar sin verse antes de morir


# Evita que torch.hub intente comandos git ruidosos
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def add_yolov5_to_syspath(repo_dir: str):
    repo_dir = os.path.normpath(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
        
def es(name: str) -> str:
    return SPANISH.get(name, name)

def fmt_pct(x: float) -> int:
    """Convierte 0.0–1.0 a porcentaje 0–100 redondeado."""
    try:
        return int(round(float(x) * 100))
    except Exception:
        return 0


class VideoThread(QThread):
    frameSignal  = pyqtSignal(np.ndarray)       # frame anotado
    countSignal  = pyqtSignal(dict)             # conteos en pantalla (presentes)
    statsSignal  = pyqtSignal(float, float, float)  # cpu, ram, fps
    passSignal   = pyqtSignal(dict, int)        # NUEVO: totales pasados por la banda, y total general
    metricsSignal = pyqtSignal(dict)  # {'avg_conf': {cls:%, ...}, 'overall_conf':%, 'overall_err':%}


    def __init__(self, cam_index, model, class_names):
        super().__init__()
        self.cam_index = cam_index
        self.model = model
        self.class_names = class_names
        self.running = True
        
        self.conf_sum = {n: 0.0 for n in self.class_names}
        self.conf_cnt = {n: 0   for n in self.class_names}

        # --- tracking simple ---
        self.tracks = {}        # id -> {'cls': int, 'xy': (x,y), 'last_y': float, 'counted': bool, 'ttl': int}
        self.next_id = 1
        self.totals = {n: 0 for n in class_names}
        self.total_overall = 0

        # anti-parpadeo
        self.last_det = np.empty((0, 6))
        self.last_counts = {n: 0 for n in class_names}
        
        self.inf_times = deque(maxlen=60) # últimos tiempos de inferencia
        self.last_avg_infer_ms = 0.0

    def stop(self):
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if not cap.isOpened():
            return

        prev_t, fps = time.time(), 0.0
        counts = {n: 0 for n in self.class_names}
        frame_id = 0

        with torch.no_grad():
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02); continue

                frame_id += 1
                do_process = (frame_id % PROCESS_EVERY == 0)
                annotated = frame.copy()

                h, w = annotated.shape[:2]
                y_line = int(COUNT_LINE_Y_RATIO * h)

                if do_process:
                    
                    t0 = perf_counter()
                    
                    # ---------- INFERENCIA ----------
                    results = self.model(frame, size=INFER_SIZE)
                    infer_ms = (perf_counter() - t0) * 1000.0
                    self.inf_times.append(infer_ms)
                    avg_infer_ms = sum(self.inf_times) / len(self.inf_times)
                    self.last_avg_infer_ms = avg_infer_ms
                    
                    det = results.xyxy[0].cpu().numpy() if hasattr(results, "xyxy") else np.empty((0, 6))

                    # Conteo en pantalla (presentes)
                    counts = {n: 0 for n in self.class_names}
                    det_list = []
                    for *xyxy, conf, cls in det:
                        cls = int(cls)
                        if 0 <= cls < len(self.class_names) and conf >= CONF_THRES:
                            counts[self.class_names[cls]] += 1
                            x1, y1, x2, y2 = map(int, xyxy)
                            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                            det_list.append((cls, conf, (x1, y1, x2, y2), (cx, cy)))
                            
                            # acumular para métricas
                            self.conf_sum[self.class_names[cls]] += float(conf) * 100.0
                            self.conf_cnt[self.class_names[cls]] += 1
                            
                    

                    # ---------- TRACKING ligero por centroides ----------
                    # reduce TTL de tracks existentes
                    for tid in list(self.tracks.keys()):
                        self.tracks[tid]['ttl'] -= 1
                        if self.tracks[tid]['ttl'] <= 0:
                            del self.tracks[tid]

                    # asociar detecciones a tracks por mínima distancia (misma clase)
                    used = set()
                    for tid, tr in list(self.tracks.items()):
                        best_j = -1; best_d = 1e9
                        for j, (cls, conf, box, (cx, cy)) in enumerate(det_list):
                            if j in used or cls != tr['cls']:
                                continue
                            d = np.hypot(cx - tr['xy'][0], cy - tr['xy'][1])
                            if d < best_d:
                                best_d, best_j = d, j
                        # actualizar si hay match
                        if best_j >= 0 and best_d <= MAX_MATCH_DIST:
                            cls, conf, box, (cx, cy) = det_list[best_j]; used.add(best_j)
                            # cruce de línea (arriba -> abajo) y no contado aún
                            if (tr['last_y'] < y_line) and (cy >= y_line) and (not tr['counted']):
                                cname = self.class_names[cls]
                                self.totals[cname] += 1
                                self.total_overall += 1
                                tr['counted'] = True

                            tr['xy'] = (cx, cy)
                            tr['last_y'] = cy
                            tr['ttl'] = TRACK_TTL

                    # crear tracks nuevos para detecciones sin asociar
                    for j, (cls, conf, box, (cx, cy)) in enumerate(det_list):
                        if j in used:
                            continue
                        self.tracks[self.next_id] = dict(
                            cls=cls, xy=(cx, cy), last_y=cy, counted=False, ttl=TRACK_TTL
                        )
                        self.next_id += 1

                    # guardar para anti-parpadeo
                    self.last_det = np.array([[*b[2], b[1], b[0]] for b in det_list])  # x1,y1,x2,y2,conf,cls
                    self.last_counts = counts
                else:
                    # usar última detección
                    avg_infer_ms = self.last_avg_infer_ms
                    det = self.last_det
                    counts = self.last_counts

                # ---------- DIBUJO ----------
                # cajas
                if det.size > 0:
                    for row in det:
                        x1, y1, x2, y2, conf, cls = row
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        cls = int(cls)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 160, 255), 2)
                        conf_pct = fmt_pct(conf); err_pct = 100 - conf_pct
                        name_es = SPANISH.get(self.class_names[cls], self.class_names[cls])
                        label = f"{name_es} {conf_pct}% (error {err_pct}%)"
                        cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,160,255), 2)

                # línea de conteo
                cv2.line(annotated, (0, y_line), (w, y_line), (0, 255, 0), 2)
                cv2.putText(annotated, "Linea de conteo", (10, max(20, y_line-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                # FPS suavizado
                t = time.time(); dt = t - prev_t; prev_t = t
                if dt > 0: fps = 0.9 * fps + 0.1 * (1.0 / dt)

                # Stats
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent

                # --- métricas para gráficos ---
                avg_conf = {}
                tot_sum, tot_cnt = 0.0, 0
                for k in self.class_names:
                    c = self.conf_cnt[k]
                    avg_conf[k] = (self.conf_sum[k] / c) if c else 0.0
                    tot_sum += self.conf_sum[k]; tot_cnt += c
                overall_conf = (tot_sum / tot_cnt) if tot_cnt else 0.0
                overall_err  = max(0.0, 100.0 - overall_conf)

                self.metricsSignal.emit({
                    
                    "avg_conf": {k: (self.conf_sum[k]/self.conf_cnt[k] if self.conf_cnt[k] else 0.0)
                                for k in self.class_names},
                    "overall_conf": (sum(self.conf_sum.values()) / max(1, sum(self.conf_cnt.values()))),
                    "overall_err":  max(0.0, 100.0 - (sum(self.conf_sum.values()) / max(1, sum(self.conf_cnt.values())))),
                    "avg_infer_ms": avg_infer_ms
                })


                # Señales a la UI
                self.frameSignal.emit(annotated)
                self.countSignal.emit(dict(counts))
                self.statsSignal.emit(cpu, ram, fps)
                self.passSignal.emit(dict(self.totals), int(self.total_overall))

        try:
            cap.release()
        except Exception:
            pass



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Detección de Cítricos")
        self.setMinimumSize(QSize(1200, 700))

        # Cargar modelo YOLOv5 local
        try:
            add_yolov5_to_syspath(str(YOLOV5_DIR))
            self.model = torch.hub.load(str(YOLOV5_DIR), 'custom', path=str(WEIGHTS_PT), source='local')
            self.model.eval()
            self.model.conf = CONF_THRES
            self.model.iou  = IOU_THRES
            self.model.max_det = 100
            self.model.to('cpu')
        except Exception as e:
            QMessageBox.critical(self, "Modelo", f"No se pudo cargar YOLOv5.\n\n{e}")
            raise

        # ---- UI ----
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QVBoxLayout(); right = QVBoxLayout()
        root.addLayout(left, 3); root.addLayout(right, 2)

        # Lienzo de vídeo: tamaño fijo para que NO crezca
        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 520)   # <- ajústalo si quieres
        self.video_label.setStyleSheet("background:#111; border:2px solid #444;")
        self.video_label.setAlignment(Qt.AlignCenter)
        left.addWidget(self.video_label, 0, Qt.AlignCenter)

        # Barras CPU/RAM/FPS
        grid = QGridLayout(); left.addLayout(grid)
        f = QFont(); f.setPointSize(10)
        self.cpu_bar = QProgressBar(); self.ram_bar = QProgressBar(); self.fps_bar = QProgressBar()
        for b in (self.cpu_bar, self.ram_bar, self.fps_bar):
            b.setRange(0, 100); b.setValue(0); b.setTextVisible(True)
        grid.addWidget(QLabel("CPU:").setFont(f) or QLabel("CPU:"), 0, 0); grid.addWidget(self.cpu_bar, 0, 1)
        grid.addWidget(QLabel("RAM:").setFont(f) or QLabel("RAM:"), 1, 0); grid.addWidget(self.ram_bar, 1, 1)
        grid.addWidget(QLabel("FPS:"), 2, 0); grid.addWidget(self.fps_bar, 2, 1)
        
        
        # Total pasados        
        self.pasados_box = QLabel("Total de cítricos pasado en la banda transportadora\n\nTotal: 0")
        self.pasados_box.setAlignment(Qt.AlignCenter)
        self.pasados_box.setWordWrap(True)
        self.pasados_box.setStyleSheet("border:2px solid #444; padding:14px;")
        right.addWidget(self.pasados_box)


        # Panel derecho
        hdr = QLabel("Tipo de cítrico (conteo)")
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setStyleSheet("background:#222; color:#fff; padding:8px; border:2px solid #444;")
        right.addWidget(hdr)

        self.count_labels = {}
        for name in CLASS_NAMES:
            lab = QLabel(f"{SPANISH.get(name, name)}: 0")
            lab.setStyleSheet("border:1px solid #444; padding:6px;")
            lab.setFont(f)
            right.addWidget(lab)
            self.count_labels[name] = lab  # clave sigue siendo el nombre en inglés

        """objetivos = [
            "• Clasificar (≥90% por clase).",
            "• Aciertos≥85% / error<15%.",
            "• Reportes y visualizaciones.",
            "• UX ≥70/100 y finalización ≥85%.",
            "• Evaluar preprocesamiento (iluminación).",
            "• Validación en La Hermelinda (tiempo/errores).",
        ]
        box = QLabel("\n".join(objetivos)); box.setWordWrap(True)
        box.setStyleSheet("border:2px solid #444; padding:8px;")
        right.addWidget(box, 1)"""
        
        # --- Gráfico 1: Confianza promedio por clase ---
        self.fig1 = Figure(figsize=(3.6, 1.8), dpi=100)
        self.ax1  = self.fig1.add_subplot(111)
        self.ax1.set_ylim(0, 100)
        self.ax1.axhline(90, linestyle="--", linewidth=1)
        self.bars1 = self.ax1.bar([SPANISH.get(n,n) for n in CLASS_NAMES], [0]*len(CLASS_NAMES))
        self.txt1 = [
            self.ax1.text(b.get_x()+b.get_width()/2.0, 0.5, "0%", ha="center", va="bottom", fontsize=9)
            for b in self.bars1
        ]
        self.canvas1 = FigureCanvas(self.fig1)
        right.addWidget(self.canvas1)

        # --- Gráfico 2: Error global ---
        self.fig2 = Figure(figsize=(3.6, 1.2), dpi=100)
        self.ax2  = self.fig2.add_subplot(111)
        self.ax2.set_ylim(0, 100)
        self.ax2.axhline(15, linestyle="--", linewidth=1)
        self.bar2 = self.ax2.bar(["Error"], [0])
        b = self.bar2[0]
        self.txt_err = self.ax2.text(b.get_x()+b.get_width()/2.0, 0.5, "0%", ha="center", va="bottom", fontsize=9)
        self.canvas2 = FigureCanvas(self.fig2)
        right.addWidget(self.canvas2)
        
        # Panel de métricas globales:
        self.kpi_label = QLabel("Precisión global: -- %   |   Tiempo medio: -- ms")
        self.kpi_label.setStyleSheet("border:1px solid #444; padding:6px;")
        right.addWidget(self.kpi_label)
        
        self._init_charts()
        
        


        # --- Mapa incrustado (abajo a la derecha) ---
        if HAS_WEB and os.path.exists(OUTPUT_HTML):
            self.map_view = QWebEngineView()
            # tamaño similar a tu mockup
            self.map_view.setFixedSize(360, 280)
            self.map_view.setZoomFactor(1.0)
            self.map_view.setStyleSheet("border:2px solid #444;")
            url = QUrl.fromLocalFile(os.path.abspath(OUTPUT_HTML))
            self.map_view.setUrl(url)
            right.addWidget(self.map_view, 0, Qt.AlignBottom)
            
            self._map_mtime = os.path.getmtime(OUTPUT_HTML)
            self._map_timer = QTimer(self)
            self._map_timer.timeout.connect(self._maybe_reload_map)
            self._map_timer.start(5000)  # cada 5 s
            
        else:
            # Fallback si no está PyQtWebEngine o aún no existe el HTML:
            self.map_label = QLabel("Mapa (clic para abrir):\n" + os.path.abspath(OUTPUT_HTML))
            self.map_label.setWordWrap(True)
            self.map_label.setStyleSheet("border:2px solid #444; padding:8px; color:#2a6;")
            self.map_label.setCursor(Qt.PointingHandCursor)
            self.map_label.mousePressEvent = self.open_map
            right.addWidget(self.map_label, 0, Qt.AlignBottom)

        # Hilo de vídeo (comunicación por señales -> seguro para Qt)
        self.thread = VideoThread(CAM_INDEX, self.model, CLASS_NAMES)
        self.thread.frameSignal.connect(self.on_frame)     # llega en hilo principal
        self.thread.countSignal.connect(self.on_counts)
        self.thread.statsSignal.connect(self.on_stats)
        self.thread.passSignal.connect(self.on_pass_update)
        self.thread.metricsSignal.connect(self.on_metrics) # llega en hilo principal
        self.thread.start()

        # Flag de repintado para evitar “recursive repaint”
        self._painting = False
        
    def on_pass_update(self, totals: dict, overall: int):
        lineas = [f"{SPANISH.get(k,k)}: {totals.get(k,0)}" for k in self.count_labels.keys()]
        texto = "Total de cítricos pasado en la banda transportadora\n\n"
        texto += "\n".join(lineas) + f"\n\nTotal: {overall}"
        self.pasados_box.setText(texto)


    # Slots (siempre en hilo principal):
    def on_frame(self, bgr):
        if self._painting:
            return
        self._painting = True
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)
        self._painting = False

    def on_counts(self, counts: dict):
        for name, lab in self.count_labels.items():
            lab.setText(f"{SPANISH.get(name, name)}: {counts.get(name, 0)}")

    def on_stats(self, cpu, ram, fps):
        self.cpu_bar.setValue(int(cpu))
        self.ram_bar.setValue(int(ram))
        self.fps_bar.setValue(min(100, int(fps)))
        self.fps_bar.setFormat(f"{fps:0.1f} fps")

    def open_map(self, _evt):
        if os.path.exists(OUTPUT_HTML):
            webbrowser.open(f"file:///{os.path.abspath(OUTPUT_HTML)}")
        else:
            QMessageBox.information(self, "Mapa",
                "No se encontró el HTML del mapa.\nEjecuta: prueba/generar_mapa_csv.py")
            
    def _maybe_reload_map(self):
        try:
            m = os.path.getmtime(OUTPUT_HTML)
            if m != getattr(self, "_map_mtime", 0):
                self._map_mtime = m
                self.map_view.reload()
        except Exception:
            pass

    def closeEvent(self, e):
        try:
            self.thread.stop()
            self.thread.wait(1000)
        except Exception:
            pass
        super().closeEvent(e)
        
    def add_yolov5_to_syspath(repo_dir: str | Path):
        repo_dir = str(Path(repo_dir).resolve())
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
            
    def _init_charts(self):
        # barras de confianza por clase
        self.ax1.clear()
        self.ax1.set_title("Confianza promedio por clase (%)", fontsize=9)
        self.ax1.set_ylim(0, 100)
        # línea objetivo 90%
        self.ax1.axhline(90, linestyle='--', linewidth=1)
        self.class_order = CLASS_NAMES[:]  # mantener orden
        labels = [SPANISH.get(n, n) for n in self.class_order]
        self.bars1 = self.ax1.bar(labels, [0]*len(labels))
        
        self.txt1 = [
        self.ax1.text(bar.get_x() + bar.get_width()/2.0, 0.5, "0%",
                    ha="center", va="bottom", fontsize=9)
        for bar in self.bars1
]
        
        self.fig1.tight_layout()
        self.canvas1.draw()

        # barra de error global
        self.ax2.clear()
        self.ax2.set_title("Error global (%) – objetivo < 15", fontsize=9)
        self.ax2.set_ylim(0, 100)
        # línea objetivo 15%
        self.ax2.axhline(15, linestyle='--', linewidth=1)
        self.bar2 = self.ax2.bar(["Error"], [0])
        
        b = self.bar2[0]
        self.txt_err = self.ax2.text(b.get_x() + b.get_width()/2.0, 0.5, "0%",
                                    ha="center", va="bottom", fontsize=9)
        
        self.fig2.tight_layout()
        self.canvas2.draw()

    def on_metrics(self, payload: dict):
        # ---- valores recibidos del hilo ----
        avg_conf     = payload.get("avg_conf", {})
        overall_conf = float(payload.get("overall_conf", 0.0))
        overall_err  = float(payload.get("overall_err", 0.0))
        avg_infer_ms = float(payload.get("avg_infer_ms", 0.0))

        # ---- actualizar barras por clase (confianza %) ----
        for i, name in enumerate(CLASS_NAMES):
            val = float(avg_conf.get(name, 0.0))
            b = self.bars1[i]
            b.set_height(val)
            self.txt1[i].set_position((b.get_x() + b.get_width()/2.0, max(1, val) + 1))
            self.txt1[i].set_text(f"{val:.0f}%")
        self.canvas1.draw_idle()

        # ---- actualizar barra de error global ----
        b = self.bar2[0]
        b.set_height(overall_err)
        self.txt_err.set_position((b.get_x() + b.get_width()/2.0, max(1, overall_err) + 1))
        self.txt_err.set_text(f"{overall_err:.0f}%")
        self.canvas2.draw_idle()

        # ---- actualizar KPI (criterio ≥90% y ≤250 ms) ----
        ok_acc  = (overall_conf >= 90.0)
        ok_time = (avg_infer_ms <= 250.0)
        color = "#2ecc71" if (ok_acc and ok_time) else "#e74c3c"
        self.kpi_label.setText(
            f"Precisión global: {overall_conf:0.1f}%   |   Tiempo medio: {avg_infer_ms:0.0f} ms"
        )
        self.kpi_label.setStyleSheet(f"border:1px solid #444; padding:6px; color:{color};")

    

if __name__ == "__main__":
    # Solo CPU (RPi/PC): sin gradientes = menos memoria
    torch.set_grad_enabled(False)
    add_yolov5_to_syspath(YOLOV5_DIR)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
