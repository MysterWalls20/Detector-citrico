# app_tk_dashboard.py
import os, sys, time, webbrowser, pathlib, json
from time import perf_counter
from collections import deque
from pathlib import Path

import cv2, torch, psutil, numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


from PIL import Image, ImageTk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvasTk

# ====== CONTROL DE HARDWARE (Motor DC) ======
try:
    import RPi.GPIO as GPIO
    HW_AVAILABLE = True
except ImportError:
    # Permite correr el código en Windows sin explotar
    GPIO = None
    HW_AVAILABLE = False

# Pines ejemplo (BCM). Ajusta según tu cableado real.
MOTOR_IN1 = 23
MOTOR_IN2 = 24
MOTOR_ENA = 18  # PWM

# Pin del servomotor (BCM). AJÚSTALO a tu cableado real.
SERVO_PIN = 25   # por ejemplo
SERVO_DEFAULT_ANGLE = 60  # ángulo predeterminado de la rampa (ajústalo)

class ServoController:
    def __init__(self):
        self.enabled = HW_AVAILABLE
        self.angle = 0  # 0–180
        self.pwm = None

        if not self.enabled:
            print("[ServoController] GPIO no disponible, modo simulación.")
            return

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(SERVO_PIN, GPIO.OUT)

            # 50 Hz típico
            self.pwm = GPIO.PWM(SERVO_PIN, 50)
            self.pwm.start(0)   # sin mover
        except RuntimeError as e:
            print("[ServoController] Error inicializando GPIO:", e)
            print("[ServoController] Pasando a modo simulación.")
            self.enabled = False
            self.pwm = None

    def _angle_to_duty(self, angle: float) -> float:
        angle = max(0, min(180, float(angle)))
        duty = 2.5 + (angle / 180.0) * 10.0
        return duty

    def set_angle(self, angle: float):
        self.angle = max(0, min(180, float(angle)))
        if not self.enabled or self.pwm is None:
            print(f"[ServoController] (sim) ángulo = {self.angle:.1f}°")
            return
        try:
            duty = self._angle_to_duty(self.angle)
            self.pwm.ChangeDutyCycle(duty)
        except Exception as e:
            print("[ServoController] Error en set_angle:", e)

    def stop(self):
        if not self.enabled or self.pwm is None:
            print("[ServoController] (sim) stop")
            return
        try:
            # duty 0 = deja de mandar PWM
            self.pwm.ChangeDutyCycle(0)
        except Exception as e:
            print("[ServoController] Error en stop:", e)

    def cleanup(self):
        if not self.enabled or self.pwm is None:
            return
        try:
            self.stop()      # por si acaso
            self.pwm.stop()
        except Exception as e:
            print("[ServoController] Error en cleanup:", e)
        # OJO: aquí YA NO hacemos GPIO.cleanup()

class MotorController:
    def __init__(self):
        self.enabled = HW_AVAILABLE
        self.speed = 0  # 0–100

        if not self.enabled:
            print("[MotorController] GPIO no disponible, modo simulación.")
            return

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(MOTOR_IN1, GPIO.OUT)
            GPIO.setup(MOTOR_IN2, GPIO.OUT)
            GPIO.setup(MOTOR_ENA, GPIO.OUT)

            self.pwm = GPIO.PWM(MOTOR_ENA, 1000)  # 1 kHz
            self.pwm.start(0)
            self.stop()
        except RuntimeError as e:
            print("[MotorController] Error inicializando GPIO:", e)
            print("[MotorController] Pasando a modo simulación.")
            self.enabled = False
            self.pwm = None

    def set_speed(self, value: int):
        """value: 0–100 (duty cycle)"""
        self.speed = max(0, min(100, int(value)))
        if not self.enabled:
            print(f"[MotorController] (sim) velocidad = {self.speed}%")
            return
        self.pwm.ChangeDutyCycle(self.speed)

    def forward(self):
        if not self.enabled:
            print("[MotorController] (sim) adelante")
            return
        GPIO.output(MOTOR_IN1, GPIO.HIGH)
        GPIO.output(MOTOR_IN2, GPIO.LOW)

    def backward(self):
        if not self.enabled:
            print("[MotorController] (sim) atrás")
            return
        GPIO.output(MOTOR_IN1, GPIO.LOW)
        GPIO.output(MOTOR_IN2, GPIO.HIGH)

    def stop(self):
        if not self.enabled:
            print("[MotorController] (sim) stop")
            return
        try:
            GPIO.output(MOTOR_IN1, GPIO.LOW)
            GPIO.output(MOTOR_IN2, GPIO.LOW)
            if hasattr(self, "pwm") and self.pwm:
                self.pwm.ChangeDutyCycle(0)
            self.speed = 0
        except Exception as e:
            print("[MotorController] Error en stop:", e)

    def cleanup(self):
        if not self.enabled:
            return
        try:
            self.stop()
        except Exception as e:
            print("[MotorController] Error en cleanup:", e)
        # OJO: aquí TAMPOCO hacemos GPIO.cleanup()

# =========================
# CONFIG & RUTAS
# =========================
def project_root() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]  # raíz del proyecto

ROOT = project_root()
YOLOV5_DIR = (ROOT / "yolov5").resolve()
RUNS_DIR   = YOLOV5_DIR / "runs" / "train"

def find_latest_best(runs_dir: Path) -> Path | None:
    cands = list((runs_dir).glob("**/weights/best.pt"))
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None

# Ajusta si tienes una ruta absoluta fija
PESO_ABSOLUTO = ROOT / "yolov5" / "runs" / "train" / "exp9" / "weights" / "best.pt"
WEIGHTS_PT = PESO_ABSOLUTO if PESO_ABSOLUTO.exists() else find_latest_best(RUNS_DIR)
if WEIGHTS_PT is None:
    raise FileNotFoundError("No se encontró best.pt en yolov5/runs/train/**/weights/")

CSV_PATH    = (ROOT / "prueba" / "coordenadas.csv").resolve()
OUTPUT_HTML = (ROOT / "prueba" / "mapa_citricos_csv.html").resolve()

CLASS_NAMES = ["lemon", "mandarin", "orange", "grapefruit"]
SPANISH = {"lemon":"Limom","mandarin":"Mandarina","orange":"Naranja","grapefruit":"Toronja"}

INFER_SIZE   = 416
CONF_THRES   = 0.25
IOU_THRES    = 0.45
TARGET_FPS   = 15
PROCESS_EVERY= 2

# Zona de conteo (en proporción al ancho/alto de la imagen)
ROI_X1_RATIO = 0.05   # 20% desde la izquierda
ROI_X2_RATIO = 0.95   # 80% hacia la derecha
ROI_Y1_RATIO = 0.45   # 45% desde arriba
ROI_Y2_RATIO = 0.75   # 75% desde arriba

MAX_MATCH_DIST     = 60
TRACK_TTL          = 12

def es(name): return SPANISH.get(name, name)
def pct(x):
    try: return int(round(float(x)*100))
    except: return 0

def in_roi(cx, cy, x1, y1, x2, y2):
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)

def add_yolov5_to_syspath(p: Path):
    s = str(p.resolve())
    if s not in sys.path: sys.path.insert(0, s)

# =========================
# DETECTOR (cámara o video)
# =========================
class Detector:
    def __init__(self):
        add_yolov5_to_syspath(YOLOV5_DIR)

        # Parche por si los pesos vinieron de Windows
        _orig_WindowsPath = None
        try:
            _orig_WindowsPath = pathlib.WindowsPath
            pathlib.WindowsPath = pathlib.PosixPath
        except Exception:
            pass

        try:
            self.model = torch.hub.load(
                str(YOLOV5_DIR), 'custom', path=str(WEIGHTS_PT),
                source='local', force_reload=False
            )
        finally:
            if _orig_WindowsPath is not None:
                pathlib.WindowsPath = _orig_WindowsPath

        self.model.eval()
        self.model.conf = CONF_THRES
        self.model.iou  = IOU_THRES
        self.model.max_det = 100
        self.model.to('cpu')
        torch.set_grad_enabled(False)

        # Fuente de video
        self.cap = None
        self.source_type = None  # "cam" | "file"
        self.video_path = None

        # Estado
        self.frame_id = 0
        self.prev_t = time.time()
        self.fps = 0.0

        # Tracking / conteos
        self.tracks = {}
        self.next_id = 1
        self.totals = {n: 0 for n in CLASS_NAMES}
        self.total_overall = 0
        self.last_det = np.empty((0,6))
        self.last_counts = {n:0 for n in CLASS_NAMES}

        # Métricas
        self.conf_sum = {n: 0.0 for n in CLASS_NAMES}
        self.conf_cnt = {n: 0    for n in CLASS_NAMES}
        self.inf_times = deque(maxlen=60)
        self.last_avg_infer_ms = 0.0

    # ----- control de fuente -----
    def set_source_camera(self, cam_index:int=0):
        self.release()
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if os.name=='nt' else cv2.CAP_V4L2)
        if os.name != 'nt':
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.source_type = "cam"
        self.video_path = None
        self._reset_runtime_state()

    def set_source_file(self, path:str):
        self.release()
        self.cap = cv2.VideoCapture(path)
        self.source_type = "file"
        self.video_path = path
        self._reset_runtime_state()

    def _reset_runtime_state(self):
        self.frame_id = 0
        self.last_det = np.empty((0,6))
        self.last_counts = {n:0 for n in CLASS_NAMES}
        self.tracks.clear()
        self.next_id = 1
        # No reseteo totales para mantener acumulado general; si quieres, limpia self.totals

    # ----- ciclo de lectura+anotación -----
    def read_annotated(self):
        if self.cap is None:
            return None, {}, 0.0

        ok, frame = self.cap.read()
        if not ok:
            # si es video y terminó
            if self.source_type == "file":
                # fin de video
                self.release()
            return None, {}, 0.0

        self.frame_id += 1
        h, w = frame.shape[:2]
        
        # Cálculo de la zona de conteo (rectángulo)
        roi_x1 = int(ROI_X1_RATIO * w)
        roi_x2 = int(ROI_X2_RATIO * w)
        roi_y1 = int(ROI_Y1_RATIO * h)
        roi_y2 = int(ROI_Y2_RATIO * h)
        
        annotated = frame.copy()

        do_process = (self.frame_id % PROCESS_EVERY == 0)
        with torch.no_grad():
            if do_process:
                t0 = perf_counter()
                results = self.model(frame, size=INFER_SIZE)
                infer_ms = (perf_counter() - t0) * 1000.0
                self.inf_times.append(infer_ms)
                self.last_avg_infer_ms = sum(self.inf_times)/len(self.inf_times)

                det = results.xyxy[0].cpu().numpy() if hasattr(results,"xyxy") else np.empty((0,6))
                counts = {n:0 for n in CLASS_NAMES}
                det_list=[]
                for *xyxy, conf, cls in det:
                    cls = int(cls)
                    if 0 <= cls < len(CLASS_NAMES) and conf >= CONF_THRES:
                        counts[CLASS_NAMES[cls]] += 1
                        x1,y1,x2,y2 = map(int, xyxy)
                        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
                        det_list.append((cls, conf, (x1,y1,x2,y2), (cx,cy)))

                        # métricas
                        self.conf_sum[CLASS_NAMES[cls]] += float(conf)*100.0
                        self.conf_cnt[CLASS_NAMES[cls]] += 1

                # tracking TTL—
                for tid in list(self.tracks.keys()):
                    self.tracks[tid]['ttl'] -= 1
                    if self.tracks[tid]['ttl'] <= 0:
                        del self.tracks[tid]

                # asociar detecciones a tracks
                # asociar detecciones a tracks
                used = set()
                for tid, tr in list(self.tracks.items()):
                    best_j, best_d = -1, 1e9
                    for j, (cls, conf, box, (cx, cy)) in enumerate(det_list):
                        if j in used or cls != tr['cls']:
                            continue
                        d = np.hypot(cx - tr['xy'][0], cy - tr['xy'][1])
                        if d < best_d:
                            best_d, best_j = d, j

                    if best_j >= 0 and best_d <= MAX_MATCH_DIST:
                        cls, conf, box, (cx, cy) = det_list[best_j]
                        used.add(best_j)

                        # ¿el centro está dentro del rectángulo de conteo?
                        if in_roi(cx, cy, roi_x1, roi_y1, roi_x2, roi_y2) and (not tr['counted']):
                            cname = CLASS_NAMES[cls]
                            self.totals[cname] += 1
                            self.total_overall += 1
                            tr['counted'] = True

                        tr['xy'] = (cx, cy)
                        tr['ttl'] = TRACK_TTL

                #
                for j, (cls, conf, box, (cx, cy)) in enumerate(det_list):
                    if j in used:
                        continue
                    self.tracks[self.next_id] = dict(
                        cls=cls,
                        xy=(cx, cy),
                        counted=False,
                        ttl=TRACK_TTL
                    )
                    self.next_id += 1

                self.last_det = np.array([[*b[2], b[1], b[0]] for b in det_list])
                self.last_counts = counts
            else:
                det    = self.last_det
                counts = self.last_counts

        # dibujar overlay
        if det.size>0:
            for row in det:
                x1,y1,x2,y2,conf,cls = row
                x1,y1,x2,y2 = map(int,(x1,y1,x2,y2)); cls=int(cls)
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,160,255),2)
                confp = pct(conf); errp = 100-confp
                label = f"{es(CLASS_NAMES[cls])} {confp}% (error {errp}%)"
                cv2.putText(annotated, label,(x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,160,255),2)

        # Zona de conteo (rectángulo)
        cv2.rectangle(annotated, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            "Zona de conteo",
            (roi_x1 + 5, max(20, roi_y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        # fps suave
        t=time.time(); dt=t-self.prev_t; self.prev_t=t
        if dt>0: self.fps = 0.9*self.fps + 0.1*(1.0/dt)

        return annotated, counts, self.fps

    def get_metrics(self):
        avg_conf = {}
        tot_sum, tot_cnt = 0.0, 0
        for k in CLASS_NAMES:
            c = self.conf_cnt[k]
            avg_conf[k] = (self.conf_sum[k]/c) if c else 0.0
            tot_sum += self.conf_sum[k]; tot_cnt += c
        overall_conf = (tot_sum/tot_cnt) if tot_cnt else 0.0
        overall_err  = max(0.0, 100.0 - overall_conf)
        return dict(
            avg_conf=avg_conf,
            overall_conf=overall_conf,
            overall_err=overall_err,
            avg_infer_ms=self.last_avg_infer_ms
        )

    def release(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except: pass
        self.cap = None

# =========================
# UI DASHBOARD (Tkinter)
# =========================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Myster Lemon")
        self.geometry("1280x720")
        self.configure(bg="#ffffff")

        self.det = Detector()
        self._imgtk = None
        self.after_id = None
        # Control de hardware (motor DC banda)
        self.motor = MotorController()
        self.current_speed = 0
        # Control de hardware (servomotor rampa)
        self.servo = ServoController()
        self.current_angle = 0
                
        self.servo_ready = False

        # =========================
        # LAYOUT GENERAL
        # =========================
        root = tk.Frame(self, bg="#ffffff")
        root.pack(fill="both", expand=True)

        # --- PANEL IZQUIERDO (VERDE) ---
        sidebar = tk.Frame(root, bg="#2ec50f", width=260, bd=2, relief="solid")
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # Título del panel
        lbl_title = tk.Label(
            sidebar,
            text="Myster Lemon",
            bg="#2ec50f",
            fg="White",
            font=("Arial", 20, "bold"),
            justify="center"
        )
        lbl_title.pack(padx=20, pady=(20, 10), anchor="w")

        # Marco blanco para botones de navegación
        nav_frame = tk.Frame(sidebar, bg="#2ec50f", bd=2, relief="flat")
        nav_frame.pack(fill="x", padx=15, pady=(10, 15))

        def make_nav_button(text, cmd):
            cont = tk.Frame(nav_frame, bg="#2ec50f", pady=5)
            cont.pack(fill="x")
            b = tk.Button(
                cont, text=text, command=cmd,
                bg="white", fg="black",
                bd=2, relief="solid",
                font=("Arial", 10)
            )
            b.pack(fill="x", padx=10)
            return b

        self.btn_nav_monitor = make_nav_button("Botón de monitoreo",
                                               lambda: self.show_page("monitor"))
        self.btn_nav_grafico = make_nav_button("Botón de gráfico",
                                               lambda: self.show_page("graficos"))
        self.btn_nav_mapa    = make_nav_button("Botón de mapa",
                                               lambda: self.show_page("mapa"))
        self.btn_nav_reporte = make_nav_button("Botón de reporte",
                                               lambda: self.show_page("reportes"))
        self.btn_nav_hardware = make_nav_button("Botón Hardware",
                                                lambda: self.show_page("hardware"))

        # KPI pequeño (precisión / tiempo) en el panel naranja
        self.kpi_small = tk.Label(
            sidebar,
            text="Precisión global: 0.0%\nTiempo medio: 0 ms",
            bg="#f2ffe6",
            fg="#2c3e50",
            bd=1,
            relief="solid",
            font=("Arial", 9),
            anchor="w",
            padx=5
        )
        self.kpi_small.pack(fill="x", padx=10, pady=(5, 5))

        # Caja: conteo presente por tipo (primero esto)
        lbl_tipo_hdr = tk.Label(
            sidebar,
            text="Tipo de cítrico (conteo)",
            bg="#333333",
            fg="white",
            bd=1,
            relief="solid",
            font=("Arial", 9, "bold")
        )
        lbl_tipo_hdr.pack(fill="x", padx=10)

        self.box_presentes_frame = tk.Frame(sidebar, bg="#ccff33", bd=0)
        self.box_presentes_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.lbl_presentes = {}
        for name in CLASS_NAMES:
            f = tk.Frame(self.box_presentes_frame, bg="#ccff33")
            f.pack(fill="x", pady=2)
            lab = tk.Label(
                f,
                text=f"{es(name)}: 0",
                bg="white",
                fg="black",
                bd=2,
                relief="solid",
                font=("Arial", 9),
                anchor="w",
                padx=6
            )
            lab.pack(fill="x")
            self.lbl_presentes[name] = lab

        # Caja: total de cítricos (AHORA VA DEBAJO)
        self.box_totales = tk.Label(
            sidebar,
            text="Total de cítricos pasados en la banda transportadora\n\n"
                "Limón: 0\nMandarina: 0\nNaranja: 0\nToronja: 0\n\nTotal: 0",
            bg="#f9f9f9",
            fg="black",
            bd=3,
            relief="solid",
            font=("Arial", 9),
            justify="center"
        )
        self.box_totales.pack(fill="x", padx=10, pady=(0, 5))
        # =========================
        # PANEL DERECHO (CONTENIDO)
        # =========================
        main = tk.Frame(root, bg="#ffffff", bd=2, relief="solid")
        main.pack(side="right", fill="both", expand=True)

        # Título grande
        lbl_main_title = tk.Label(
            main,
            text="DETECTOR DE CÍTRICO",
            bg="#ffffff",
            fg="black",
            font=("Arial", 14, "bold")
        )
        lbl_main_title.pack(pady=(10, 5))

        # Botones de cámara / subir vídeo / detener
        btn_row = tk.Frame(main, bg="#ffffff")
        btn_row.pack(pady=(5, 5))

        self.btn_cam = tk.Button(
            btn_row, text="Cámara", width=12,
            command=self.start_camera
        )
        self.btn_cam.pack(side="left", padx=15)

        self.btn_vid = tk.Button(
            btn_row, text="Subir Video", width=12,
            command=self.open_video
        )
        self.btn_vid.pack(side="left", padx=15)

        self.btn_stop = tk.Button(
            btn_row, text="Detener", width=12,
            command=self.stop_capture
        )
        self.btn_stop.pack(side="left", padx=15)

        # Contenedor donde irán las “páginas” (monitor, gráficos, etc.)
        self.page_container = tk.Frame(main, bg="#ffffff")
        self.page_container.pack(fill="both", expand=True, padx=20, pady=(10, 15))

        # Diccionario de páginas
        self.pages = {}

        # Página de Monitoreo (video)
        self._build_monitor_page()

        # Página de Gráficos
        self._build_graficos_page()

        # Página de Mapa
        self._build_mapa_page()

        # Página de Reportes
        self._build_reportes_page()

        # Página de Hardware (banda + motor)
        self._build_hardware_page()

        # Página inicial
        self.show_page("monitor")

        # Loop de actualización
        self.update_loop()

    # =========================
    # PÁGINAS
    # =========================
    def _build_monitor_page(self):
        page = tk.Frame(self.page_container, bg="#ffffff")
        page.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.video_lbl = tk.Label(page, bd=2, relief="solid", bg="#111111")
        self.video_lbl.pack(fill="both", expand=True)

        self.pages["monitor"] = page

    def _build_graficos_page(self):
        page = tk.Frame(self.page_container, bg="#ffffff")
        page.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Figura 1: confianza promedio por clase
        self.fig1 = Figure(figsize=(5.5, 2.4), dpi=100)
        self.ax1  = self.fig1.add_subplot(111)
        self.ax1.set_title("Confianza promedio por clase (%)", fontsize=10)
        self.ax1.set_ylim(0, 100)
        self.ax1.axhline(90, linestyle="--", linewidth=1)
        labels = [es(n) for n in CLASS_NAMES]
        self.bars1 = self.ax1.bar(labels, [0]*len(labels))
        self.txt1  = [
            self.ax1.text(b.get_x()+b.get_width()/2.0, 0.5, "0%",
                          ha="center", va="bottom", fontsize=9)
            for b in self.bars1
        ]
        self.canvas1 = FigureCanvasTk(self.fig1, master=page)
        self.canvas1.get_tk_widget().pack(fill="x", padx=20, pady=(20, 10))

        # Figura 2: error global
        self.fig2 = Figure(figsize=(5.5, 1.8), dpi=100)
        self.ax2  = self.fig2.add_subplot(111)
        self.ax2.set_title("Error global (%) – objetivo < 15", fontsize=10)
        self.ax2.set_ylim(0, 100)
        self.ax2.axhline(15, linestyle="--", linewidth=1)
        self.bar2 = self.ax2.bar(["Error"], [0])
        b = self.bar2[0]
        self.txt_err = self.ax2.text(b.get_x()+b.get_width()/2.0, 0.5, "0%",
                                     ha="center", va="bottom", fontsize=9)
        self.canvas2 = FigureCanvasTk(self.fig2, master=page)
        self.canvas2.get_tk_widget().pack(fill="x", padx=20, pady=(0, 20))

        self.pages["graficos"] = page

    def _build_mapa_page(self):
        page = tk.Frame(self.page_container, bg="#ffffff")
        page.place(relx=0, rely=0, relwidth=1, relheight=1)

        lbl = tk.Label(
            page,
            text=f"Ruta del mapa:\n{OUTPUT_HTML}",
            bg="#ffffff",
            fg="black",
            justify="left"
        )
        lbl.pack(anchor="w", padx=20, pady=20)

        btn = tk.Button(
            page,
            text="Abrir mapa en navegador",
            command=self.open_map
        )
        btn.pack(anchor="w", padx=20)

        self.pages["mapa"] = page

    def _build_reportes_page(self):
        page = tk.Frame(self.page_container, bg="#ffffff")
        page.place(relx=0, rely=0, relwidth=1, relheight=1)

        lbl = tk.Label(page, text="Generación de reportes (CSV)", bg="#ffffff",
                       fg="black", font=("Arial", 11))
        lbl.pack(anchor="w", padx=20, pady=(20, 10))

        btn = tk.Button(page, text="Exportar reporte (CSV)",
                        command=self.export_report_csv)
        btn.pack(anchor="w", padx=20)

        self.pages["reportes"] = page

    def _build_hardware_page(self):
        page = tk.Frame(self.page_container, bg="#ffffff")
        page.place(relx=0, rely=0, relwidth=1, relheight=1)

        title = tk.Label(
            page,
            text="CONTROL DE BANDA TRANSPORTADORA",
            bg="#ffffff",
            fg="black",
            font=("Arial", 12, "bold")
        )
        title.pack(pady=(20, 10))

        # Info de estado hardware
        self.lbl_hw_status = tk.Label(
            page,
            text="Estado hardware: GPIO habilitado" if HW_AVAILABLE else "Estado hardware: MODO SIMULACIÓN (sin GPIO)",
            bg="#ffffff",
            fg="#2ecc71" if HW_AVAILABLE else "#e74c3c",
            font=("Arial", 10)
        )
        self.lbl_hw_status.pack(pady=(0, 15))

        # Botones de control
        btn_frame = tk.Frame(page, bg="#ffffff")
        btn_frame.pack(pady=(5, 10))

        btn_avanzar = tk.Button(
            btn_frame, text="Avanzar", width=12,
            command=self.motor_forward
        )
        btn_avanzar.pack(side="left", padx=10)
        btn_retroceder = tk.Button(
            btn_frame, text="Retroceder", width=12,
            command=self.motor_backward
        )
        btn_retroceder.pack(side="left", padx=10)

        btn_detener = tk.Button(
            btn_frame, text="Detener", width=12,
            command=self.motor_stop
        )
        btn_detener.pack(side="left", padx=10)

        # Slider de velocidad
        slider_frame = tk.Frame(page, bg="#ffffff")
        slider_frame.pack(pady=(20, 10))

        lbl_slider = tk.Label(
            slider_frame,
            text="Velocidad de la banda (%)",
            bg="#ffffff",
            fg="black",
            font=("Arial", 10)
        )
        lbl_slider.pack()

        self.speed_slider = tk.Scale(
            slider_frame,
            from_=0,
            to=100,
            orient="horizontal",
            length=300,
            showvalue=True,
            command=self.on_speed_change
        )
        self.speed_slider.set(0)
        self.speed_slider.pack(pady=(5, 10))

        # ---- CONTROL DE RAMPA (SERVOMOTOR) ----
        ramp_frame = tk.Frame(page, bg="#ffffff")
        ramp_frame.pack(pady=(10, 10))

        lbl_rampa = tk.Label(
            ramp_frame,
            text="Control de rampa (servomotor) – ángulo (°)",
            bg="#ffffff",
            fg="black",
            font=("Arial", 10)
        )
        lbl_rampa.pack()

        self.servo_slider = tk.Scale(
                    ramp_frame,
                    from_=0,
                    to=180,
                    orient="horizontal",
                    length=300,
                    showvalue=True,
                    command=self.on_servo_change
                )
        self.servo_slider.set(0)
        self.servo_slider.pack(pady=(5, 10))

                # Botón para ir al ángulo predeterminado
        btn_home = tk.Button(
            ramp_frame,
            text="Ir a ángulo predeterminado",
            command=self.servo_go_default
            )
        
        # Botón para detener completamente el servo
        btn_stop_servo = tk.Button(
            ramp_frame,
            text="Detener servo",
            command=self.servo_stop
        )
        btn_stop_servo.pack(pady=(0, 10))

        self.servo_ready = True

        btn_home.pack(pady=(0, 10))

        
        # Ahora ya está creada la UI, a partir de aquí SÍ queremos responder al usuario
        self.servo_ready = True

        self.pages["hardware"] = page

    # Cambiar página visible
    def show_page(self, name: str):
        page = self.pages.get(name)
        if page:
            page.lift()

    # =========================
    # ACCIONES
    # =========================
    def start_camera(self):
        self.det.set_source_camera(0)
        self.show_page("monitor")

    def open_video(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Selecciona un video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")]
        )
        if path:
            self.det.set_source_file(path)
            self.show_page("monitor")

    def stop_capture(self):
        self.det.release()

    def open_map(self):
        if OUTPUT_HTML.exists():
            webbrowser.open(f"file:///{OUTPUT_HTML}")
        else:
            messagebox.showinfo("Mapa", f"No se encontró:\n{OUTPUT_HTML}")

    def export_report_csv(self):
        from tkinter import filedialog
        m = self.det.get_metrics()
        save_path = filedialog.asksaveasfilename(
            title="Guardar reporte CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not save_path:
            return
        try:
            import csv
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Métrica", "Valor"])
                w.writerow(["Precisión global (%)", f"{m['overall_conf']:.2f}"])
                w.writerow(["Error global (%)",     f"{m['overall_err']:.2f}"])
                w.writerow(["Tiempo promedio inferencia (ms)", f"{m['avg_infer_ms']:.2f}"])
                w.writerow([])
                w.writerow(["Totales acumulados"])
                for k in CLASS_NAMES:
                    w.writerow([es(k), self.det.totals.get(k,0)])
                w.writerow(["Total general", self.det.total_overall])
            messagebox.showinfo("Reporte", f"Reporte guardado en:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Reporte", f"No se pudo guardar el CSV:\n{e}")
    # ======== CONTROL MOTOR (HARDWARE) ========
    def motor_forward(self):
        self.motor.forward()
        # Si la velocidad es 0, ponle algo por defecto
        if self.current_speed == 0:
            self.current_speed = 50
            self.speed_slider.set(self.current_speed)
        self.motor.set_speed(self.current_speed)

    def motor_backward(self):
        self.motor.backward()
        if self.current_speed == 0:
            self.current_speed = 50
            self.speed_slider.set(self.current_speed)
        self.motor.set_speed(self.current_speed)

    def motor_stop(self):
        self.motor.stop()
        self.current_speed = 0
        self.speed_slider.set(0)

        # OPCIONAL: devolver la rampa al ángulo predeterminado
        self.current_angle = SERVO_DEFAULT_ANGLE
        self.servo_slider.set(SERVO_DEFAULT_ANGLE)
        self.servo.set_angle(SERVO_DEFAULT_ANGLE)

    def on_speed_change(self, value):
        """Callback del slider de velocidad."""
        try:
            v = int(float(value))
        except ValueError:
            v = 0
        self.current_speed = v
        self.motor.set_speed(v)

    # ======== CONTROL SERVOMOTOR (RAMPA) ========
    def on_servo_change(self, value):
            """Callback del slider de la rampa."""
            # Si la UI está inicializándose, no mover aún el servo
            if not getattr(self, "servo_ready", False):
                return

            try:
                ang = float(value)
            except ValueError:
                ang = 0.0
            self.current_angle = ang
            self.servo.set_angle(ang)
    
    def servo_go_default(self):
        """Lleva la rampa al ángulo predeterminado."""
        ang = SERVO_DEFAULT_ANGLE
        self.current_angle = ang
        # Mover el slider visualmente
        self.servo_slider.set(ang)
        # Y mandar la orden al servo
        self.servo.set_angle(ang)

    def servo_stop(self):
        """Detiene el servo (sin enviar más PWM)."""
        self.servo.stop()
        # Opcional: si NO quieres que el slider se mueva, no toques self.servo_slider
        # Si quieres que visualmente marque 'sin control', podrías dejarlo como está.

    # =========================
    # LOOP DE ACTUALIZACIÓN
    # =========================
    def update_loop(self):
        frame, counts, fps = self.det.read_annotated()
        if frame is not None:
            # ---- video ----
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            import PIL.Image, PIL.ImageTk
            im = PIL.Image.fromarray(rgb)
            im = im.resize((900, 480), PIL.Image.Resampling.BILINEAR)
            self._imgtk = PIL.ImageTk.PhotoImage(image=im)
            self.video_lbl.configure(image=self._imgtk)

            # ---- presentes en panel izquierdo ----
            for name in CLASS_NAMES:
                self.lbl_presentes[name].configure(
                    text=f"{es(name)}: {counts.get(name,0)}"
                )

            # ---- métricas y gráficos ----
            m = self.det.get_metrics()
            avg_conf     = m["avg_conf"]
            overall_conf = float(m["overall_conf"])
            overall_err  = float(m["overall_err"])
            avg_infer_ms = float(m["avg_infer_ms"])

            # KPI pequeño
            ok_acc  = (overall_conf >= 90.0)
            ok_time = (avg_infer_ms <= 250.0)
            color   = "#2ecc71" if (ok_acc and ok_time) else "#e74c3c"
            self.kpi_small.config(
                text=f"Precisión global: {overall_conf:0.1f}%\nTiempo medio: {avg_infer_ms:0.0f} ms",
                fg=color
            )

            # Caja de totales
            lineas = [f"{es(k)}: {self.det.totals.get(k,0)}" for k in CLASS_NAMES]
            texto_tot = "Total de cítricos pasados en la banda transportadora\n\n"
            texto_tot += "\n".join(lineas) + f"\n\nTotal: {self.det.total_overall}"
            self.box_totales.config(text=texto_tot)

            # Gráficos (aunque no estén visibles, se actualizan)
            if hasattr(self, "bars1"):
                for i, name in enumerate(CLASS_NAMES):
                    val = float(avg_conf.get(name,0.0))
                    b = self.bars1[i]
                    b.set_height(val)
                    self.txt1[i].set_position(
                        (b.get_x()+b.get_width()/2.0, max(1,val)+1)
                    )
                    self.txt1[i].set_text(f"{val:.0f}%")
                self.canvas1.draw_idle()

            if hasattr(self, "bar2"):
                b = self.bar2[0]
                b.set_height(overall_err)
                self.txt_err.set_position(
                    (b.get_x()+b.get_width()/2.0, max(1,overall_err)+1)
                )
                self.txt_err.set_text(f"{overall_err:.0f}%")
                self.canvas2.draw_idle()

        # volver a llamar
        self.after_id = self.after(int(1000 / TARGET_FPS), self.update_loop)
   
    def on_close(self):
            if self.after_id:
                self.after_cancel(self.after_id)
            self.det.release()

            # limpiar motor
            if hasattr(self, "motor") and self.motor is not None:
                self.motor.cleanup()

            # limpiar servo
            if hasattr(self, "servo") and self.servo is not None:
                self.servo.cleanup()

            # GPIO.cleanup() UNA sola vez al final, si está disponible
            try:
                if HW_AVAILABLE:
                    GPIO.cleanup()
            except Exception as e:
                print("Error en GPIO.cleanup():", e)

            self.destroy()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
