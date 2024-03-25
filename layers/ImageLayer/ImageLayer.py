import cv2
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtGui import QImage, QCursor, QPixmap, QIcon

from layers.Layer import Layer
from utils.image import resize2


class ImageLayer(Layer):
    def __init__(self, parent=None, contenxt=None, **kwargs):
        self.image = None
        super(ImageLayer, self).__init__(parent=parent, context=contenxt)
        self.setWindowOpacity(1.0)

    def load_content(self, content, **kwargs):
        if isinstance(content, str):
            raw = cv2.imread(content, cv2.IMREAD_UNCHANGED)
            if raw is None:
                return False, f"Load img:{content} failed"
        else:
            raw = content
        h, w = raw.shape[:2]
        self.raw = raw
        self.states.source = [0, 0, w, h]
        ar = h / w
        self.states.ar = ar
        if raw.ndim == 3:
            if raw.shape[2] == 4:
                img = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA)
            elif raw.shape[2] == 3:
                img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGBA)
            self.image = QImage(img, w, h, QImage.Format_RGBA8888)
        else:
            self.image = QImage(raw, w, h, QImage.Format_Grayscale8)
        self.states.source = [0,0,w,h]
        self.states.raw_sz = [w, h]
        return True, ""

    def get_icon(self):
        return resize2(self.raw, 128)[0]

    def paintEvent(self, ev):
        if self.image is None:
          return

        # view_rect = QtCore.QRect(*self.states.view_rect) if self.states.view_rect is not None else self.rect()

        rect = self.rect()
        w, h = rect.width(), rect.height()
        _w, _h = w, h
        _ar = h / w
        ar = self.states.ar
        if _ar > ar:
            _h = w * ar
        else:
            _w = h / ar
        dx = (w - _w) * 0.5 + self.states.transpose[0]
        dy = (h - _h) * 0.5 + self.states.transpose[0]

        p = QtGui.QPainter(self)
        p.translate(dx, dy)
        p.drawImage(QtCore.QRect(rect.x(), rect.y(), int(_w), int(_h)), self.image, QtCore.QRect(*self.states.source))
        p.end()

    def get_icon(self):
        icon = resize2(self.raw, 128)[0]
        thumb = cv2.cvtColor(icon, cv2.COLOR_BGR2RGB)
        icon = QIcon(QPixmap.fromImage(QImage(thumb,thumb.shape[1], thumb.shape[0], QImage.Format_RGB888)))
        return icon


    # def wheelEvent(self, event):
    #     delta = event.angleDelta().y() / 1200
    #     pos = self.mapFromGlobal(QCursor.pos())
    #     scale = self.opts["scale"]
    #     if scale == "auto":
    #         source = self.opts["source"]
    #         rect = self.opts["show_rect"]
    #         scale = max(source.width(), source.height()) / max(rect.width(), rect.height())
    #     scale = max(1e-6, scale - delta)
    #     self.opts["focus"] = [pos.x(), pos.y()]
    #     self.opts["scale"] = scale
    #     self.update()
    #
    # def mousePressEvent(self, event):
    #     self.latest_pos = event.position()
    #     self.mousePressed = True
    #
    # def mouseReleaseEvent(self, event):
    #     self.mousePressed = False
    #
    # def mouseMoveEvent(self, event):
    #     curr_pos = event.position()
    #     delta = self.latest_pos - curr_pos
    #     print("POS:", curr_pos, self.latest_pos, self.opts["curr_op"])
    #     self.latest_pos = curr_pos
    #     curr_op = self.opts["curr_op"]
    #     op_type = curr_op["type"]
    #     if op_type == "move":
    #         self.opts["translate"] = [delta.x(), delta.y()]
    #     elif op_type == "draw":
    #         pen_cfg = curr_op["pen"]
    #         anx, any = pen_cfg.get("anchor", [0,0])
    #         pattern = pen_cfg["pattern"]
    #         pen = QtGui.QPainter(self.image)
    #         pen.begin()
    #         pen.drawImage(QtCore.QPoint(curr_pos.x()+anx, curr_pos.y+any),pattern)
    #         pen.end()
    #     self.update()
