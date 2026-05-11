import time
import cv2
import numpy as np
import psutil
import torch

from PyQt5.QtCore import pyqtSignal, QThread
from time import perf_counter
from collections import deque

from config import (
    CLASS_NAMES, INFER_SIZE, CONF_THRES, SPANISH,
    COUNT_LINE_Y_RATIO, MAX_MATCH_DIST, TRACK_TTL,
    TARGET_FPS, PROCESS_EVERY, fmt_pct
)


class VideoThread(QThread):
    frameSignal  = pyqtSignal(np.ndarray)
    countSignal  = pyqtSignal(dict)
    statsSignal  = pyqtSignal(float, float, float)
    passSignal   = pyqtSignal(dict, int)
    metricsSignal = pyqtSignal(dict)

    def __init__(self, cam_index, model, class_names):
        super().__init__()
        self.cam_index = cam_index
        self.model = model
        self.class_names = class_names
        self.running = True

        self.conf_sum = {n: 0.0 for n in self.class_names}
        self.conf_cnt = {n: 0   for n in self.class_names}

        self.tracks = {}
        self.next_id = 1
        self.totals = {n: 0 for n in class_names}
        self.total_overall = 0

        self.last_det = np.empty((0, 6))
        self.last_counts = {n: 0 for n in class_names}

        self.inf_times = deque(maxlen=60)
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
                    time.sleep(0.02)
                    continue

                frame_id += 1
                do_process = (frame_id % PROCESS_EVERY == 0)
                annotated = frame.copy()

                h, w = annotated.shape[:2]
                y_line = int(COUNT_LINE_Y_RATIO * h)

                if do_process:
                    t0 = perf_counter()

                    results = self.model(frame, size=INFER_SIZE)
                    infer_ms = (perf_counter() - t0) * 1000.0
                    self.inf_times.append(infer_ms)
                    self.last_avg_infer_ms = sum(self.inf_times) / len(self.inf_times)

                    det = results.xyxy[0].cpu().numpy() if hasattr(results, "xyxy") else np.empty((0, 6))

                    counts = {n: 0 for n in self.class_names}
                    det_list = []
                    for *xyxy, conf, cls in det:
                        cls = int(cls)
                        if 0 <= cls < len(self.class_names) and conf >= CONF_THRES:
                            counts[self.class_names[cls]] += 1
                            x1, y1, x2, y2 = map(int, xyxy)
                            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                            det_list.append((cls, conf, (x1, y1, x2, y2), (cx, cy)))

                            self.conf_sum[self.class_names[cls]] += float(conf) * 100.0
                            self.conf_cnt[self.class_names[cls]] += 1

                    for tid in list(self.tracks.keys()):
                        self.tracks[tid]['ttl'] -= 1
                        if self.tracks[tid]['ttl'] <= 0:
                            del self.tracks[tid]

                    used = set()
                    for tid, tr in list(self.tracks.items()):
                        best_j = -1
                        best_d = 1e9
                        for j, (cls, conf, box, (cx, cy)) in enumerate(det_list):
                            if j in used or cls != tr['cls']:
                                continue
                            d = np.hypot(cx - tr['xy'][0], cy - tr['xy'][1])
                            if d < best_d:
                                best_d, best_j = d, j
                        if best_j >= 0 and best_d <= MAX_MATCH_DIST:
                            cls, conf, box, (cx, cy) = det_list[best_j]
                            used.add(best_j)
                            if (tr['last_y'] < y_line) and (cy >= y_line) and (not tr['counted']):
                                cname = self.class_names[cls]
                                self.totals[cname] += 1
                                self.total_overall += 1
                                tr['counted'] = True

                            tr['xy'] = (cx, cy)
                            tr['last_y'] = cy
                            tr['ttl'] = TRACK_TTL

                    for j, (cls, conf, box, (cx, cy)) in enumerate(det_list):
                        if j in used:
                            continue
                        self.tracks[self.next_id] = dict(
                            cls=cls, xy=(cx, cy), last_y=cy, counted=False, ttl=TRACK_TTL
                        )
                        self.next_id += 1

                    self.last_det = np.array([[*b[2], b[1], b[0]] for b in det_list])
                    self.last_counts = counts
                else:
                    self.last_avg_infer_ms
                    det = self.last_det
                    counts = self.last_counts

                if det.size > 0:
                    for row in det:
                        x1, y1, x2, y2, conf, cls = row
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        cls = int(cls)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 160, 255), 2)
                        conf_pct = fmt_pct(conf)
                        err_pct = 100 - conf_pct
                        name_es = SPANISH.get(self.class_names[cls], self.class_names[cls])
                        label = f"{name_es} {conf_pct}% (error {err_pct}%)"
                        cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 160, 255), 2)

                cv2.line(annotated, (0, y_line), (w, y_line), (0, 255, 0), 2)
                cv2.putText(annotated, "Linea de conteo", (10, max(20, y_line - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                t = time.time()
                dt = t - prev_t
                prev_t = t
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent

                self.metricsSignal.emit({
                    "avg_conf": {k: (self.conf_sum[k] / self.conf_cnt[k] if self.conf_cnt[k] else 0.0)
                                for k in self.class_names},
                    "overall_conf": (sum(self.conf_sum.values()) / max(1, sum(self.conf_cnt.values()))),
                    "overall_err": max(0.0, 100.0 - (sum(self.conf_sum.values()) / max(1, sum(self.conf_cnt.values())))),
                    "avg_infer_ms": self.last_avg_infer_ms
                })

                self.frameSignal.emit(annotated)
                self.countSignal.emit(dict(counts))
                self.statsSignal.emit(cpu, ram, fps)
                self.passSignal.emit(dict(self.totals), int(self.total_overall))

        try:
            cap.release()
        except Exception:
            pass
