import sys
import torch
from PyQt5.QtWidgets import QApplication

from config import YOLOV5_DIR, add_yolov5_to_syspath
from ui import MainWindow


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    add_yolov5_to_syspath(YOLOV5_DIR)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
