import os
import sys
import cv2
import torch
import webbrowser

from PyQt5.QtCore import Qt, QTimer, QSize, QUrl
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QProgressBar, QMessageBox
)

try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    HAS_WEB = True
except Exception:
    HAS_WEB = False

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from config import (
    ROOT, YOLOV5_DIR, WEIGHTS_PT, CLASS_NAMES, SPANISH,
    CAM_INDEX, OUTPUT_HTML, CONF_THRES, IOU_THRES, es,
    USING_PRETRAINED, add_yolov5_to_syspath
)
from detector import VideoThread


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Detección de Cítricos")
        self.setMinimumSize(QSize(1200, 700))

        if WEIGHTS_PT is None:
            QMessageBox.critical(self, "Error",
                "No se encontró ningún modelo (best.pt ni yolov5s.pt).\n\n"
                "Descarga yolov5s.pt o entrena con:\n  python train_dataset.py")
            raise FileNotFoundError("No se encontró ningún archivo .pt")

        if USING_PRETRAINED:
            QMessageBox.warning(self, "Aviso",
                "Usando modelo pre-entrenado (yolov5s.pt).\n"
                "Las detecciones NO serán precisas para cítricos.\n\n"
                "Entrena un modelo personalizado con:\n  python train_dataset.py")

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

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QVBoxLayout()
        right = QVBoxLayout()
        root.addLayout(left, 3)
        root.addLayout(right, 2)

        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 520)
        self.video_label.setStyleSheet("background:#111; border:2px solid #444;")
        self.video_label.setAlignment(Qt.AlignCenter)
        left.addWidget(self.video_label, 0, Qt.AlignCenter)

        grid = QGridLayout()
        left.addLayout(grid)
        f = QFont()
        f.setPointSize(10)
        self.cpu_bar = QProgressBar()
        self.ram_bar = QProgressBar()
        self.fps_bar = QProgressBar()
        for b in (self.cpu_bar, self.ram_bar, self.fps_bar):
            b.setRange(0, 100)
            b.setValue(0)
            b.setTextVisible(True)
        grid.addWidget(QLabel("CPU:"), 0, 0)
        grid.addWidget(self.cpu_bar, 0, 1)
        grid.addWidget(QLabel("RAM:"), 1, 0)
        grid.addWidget(self.ram_bar, 1, 1)
        grid.addWidget(QLabel("FPS:"), 2, 0)
        grid.addWidget(self.fps_bar, 2, 1)

        self.pasados_box = QLabel("Total de cítricos pasado en la banda transportadora\n\nTotal: 0")
        self.pasados_box.setAlignment(Qt.AlignCenter)
        self.pasados_box.setWordWrap(True)
        self.pasados_box.setStyleSheet("border:2px solid #444; padding:14px;")
        right.addWidget(self.pasados_box)

        hdr = QLabel("Tipo de cítrico (conteo)")
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setStyleSheet("background:#222; color:#fff; padding:8px; border:2px solid #444;")
        right.addWidget(hdr)

        self.count_labels = {}
        for name in CLASS_NAMES:
            lab = QLabel(f"{es(name)}: 0")
            lab.setStyleSheet("border:1px solid #444; padding:6px;")
            lab.setFont(f)
            right.addWidget(lab)
            self.count_labels[name] = lab

        self.fig1 = Figure(figsize=(3.6, 1.8), dpi=100)
        self.ax1  = self.fig1.add_subplot(111)
        self.ax1.set_ylim(0, 100)
        self.ax1.axhline(90, linestyle="--", linewidth=1)
        self.bars1 = self.ax1.bar([es(n) for n in CLASS_NAMES], [0] * len(CLASS_NAMES))
        self.txt1 = [
            self.ax1.text(b.get_x() + b.get_width() / 2.0, 0.5, "0%", ha="center", va="bottom", fontsize=9)
            for b in self.bars1
        ]
        self.canvas1 = FigureCanvas(self.fig1)
        right.addWidget(self.canvas1)

        self.fig2 = Figure(figsize=(3.6, 1.2), dpi=100)
        self.ax2  = self.fig2.add_subplot(111)
        self.ax2.set_ylim(0, 100)
        self.ax2.axhline(15, linestyle="--", linewidth=1)
        self.bar2 = self.ax2.bar(["Error"], [0])
        b = self.bar2[0]
        self.txt_err = self.ax2.text(b.get_x() + b.get_width() / 2.0, 0.5, "0%", ha="center", va="bottom", fontsize=9)
        self.canvas2 = FigureCanvas(self.fig2)
        right.addWidget(self.canvas2)

        self.kpi_label = QLabel("Precisión global: -- %   |   Tiempo medio: -- ms")
        self.kpi_label.setStyleSheet("border:1px solid #444; padding:6px;")
        right.addWidget(self.kpi_label)

        self._init_charts()

        if HAS_WEB and os.path.exists(OUTPUT_HTML):
            self.map_view = QWebEngineView()
            self.map_view.setFixedSize(360, 280)
            self.map_view.setZoomFactor(1.0)
            self.map_view.setStyleSheet("border:2px solid #444;")
            url = QUrl.fromLocalFile(os.path.abspath(OUTPUT_HTML))
            self.map_view.setUrl(url)
            right.addWidget(self.map_view, 0, Qt.AlignBottom)

            self._map_mtime = os.path.getmtime(OUTPUT_HTML)
            self._map_timer = QTimer(self)
            self._map_timer.timeout.connect(self._maybe_reload_map)
            self._map_timer.start(5000)
        else:
            self.map_label = QLabel("Mapa (clic para abrir):\n" + os.path.abspath(OUTPUT_HTML))
            self.map_label.setWordWrap(True)
            self.map_label.setStyleSheet("border:2px solid #444; padding:8px; color:#2a6;")
            self.map_label.setCursor(Qt.PointingHandCursor)
            self.map_label.mousePressEvent = self.open_map
            right.addWidget(self.map_label, 0, Qt.AlignBottom)

        self.thread = VideoThread(CAM_INDEX, self.model, CLASS_NAMES)
        self.thread.frameSignal.connect(self.on_frame)
        self.thread.countSignal.connect(self.on_counts)
        self.thread.statsSignal.connect(self.on_stats)
        self.thread.passSignal.connect(self.on_pass_update)
        self.thread.metricsSignal.connect(self.on_metrics)
        self.thread.start()

        self._painting = False

    def on_pass_update(self, totals: dict, overall: int):
        lineas = [f"{es(k)}: {totals.get(k, 0)}" for k in self.count_labels.keys()]
        texto = "Total de cítricos pasado en la banda transportadora\n\n"
        texto += "\n".join(lineas) + f"\n\nTotal: {overall}"
        self.pasados_box.setText(texto)

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
            lab.setText(f"{es(name)}: {counts.get(name, 0)}")

    def on_stats(self, cpu, ram, fps):
        self.cpu_bar.setValue(int(cpu))
        self.ram_bar.setValue(int(ram))
        self.fps_bar.setValue(min(100, int(fps)))
        self.fps_bar.setFormat(f"{fps:0.1f} fps")

    def open_map(self, _evt=None):
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

    def _init_charts(self):
        self.ax1.clear()
        self.ax1.set_title("Confianza promedio por clase (%)", fontsize=9)
        self.ax1.set_ylim(0, 100)
        self.ax1.axhline(90, linestyle='--', linewidth=1)
        self.class_order = CLASS_NAMES[:]
        labels = [es(n) for n in self.class_order]
        self.bars1 = self.ax1.bar(labels, [0] * len(labels))

        self.txt1 = [
            self.ax1.text(bar.get_x() + bar.get_width() / 2.0, 0.5, "0%",
                        ha="center", va="bottom", fontsize=9)
            for bar in self.bars1
        ]

        self.fig1.tight_layout()
        self.canvas1.draw()

        self.ax2.clear()
        self.ax2.set_title("Error global (%) – objetivo < 15", fontsize=9)
        self.ax2.set_ylim(0, 100)
        self.ax2.axhline(15, linestyle='--', linewidth=1)
        self.bar2 = self.ax2.bar(["Error"], [0])

        b = self.bar2[0]
        self.txt_err = self.ax2.text(b.get_x() + b.get_width() / 2.0, 0.5, "0%",
                                    ha="center", va="bottom", fontsize=9)

        self.fig2.tight_layout()
        self.canvas2.draw()

    def on_metrics(self, payload: dict):
        avg_conf     = payload.get("avg_conf", {})
        overall_conf = float(payload.get("overall_conf", 0.0))
        overall_err  = float(payload.get("overall_err", 0.0))
        avg_infer_ms = float(payload.get("avg_infer_ms", 0.0))

        for i, name in enumerate(CLASS_NAMES):
            val = float(avg_conf.get(name, 0.0))
            b = self.bars1[i]
            b.set_height(val)
            self.txt1[i].set_position((b.get_x() + b.get_width() / 2.0, max(1, val) + 1))
            self.txt1[i].set_text(f"{val:.0f}%")
        self.canvas1.draw_idle()

        b = self.bar2[0]
        b.set_height(overall_err)
        self.txt_err.set_position((b.get_x() + b.get_width() / 2.0, max(1, overall_err) + 1))
        self.txt_err.set_text(f"{overall_err:.0f}%")
        self.canvas2.draw_idle()

        ok_acc  = (overall_conf >= 90.0)
        ok_time = (avg_infer_ms <= 250.0)
        color = "#2ecc71" if (ok_acc and ok_time) else "#e74c3c"
        self.kpi_label.setText(
            f"Precisión global: {overall_conf:0.1f}%   |   Tiempo medio: {avg_infer_ms:0.0f} ms"
        )
        self.kpi_label.setStyleSheet(f"border:1px solid #444; padding:6px; color:{color};")
