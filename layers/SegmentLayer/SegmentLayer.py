import cv2
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtGui import QImage, QCursor, QPixmap, QIcon
from PIL import Image
from layers.Layer import Layer
from utils.image import resize2


PALETTE = [[0, 0, 0, 255], [128, 0, 0, 255], [0, 128, 0, 255], [128, 128, 0, 255], [0, 0, 128, 255],
               [128, 0, 128, 255], [0, 128, 128, 255], [128, 128, 128, 255], [64, 0, 0, 255],
               [192, 0, 0, 255], [64, 128, 0, 255], [192, 128, 0, 255], [64, 0, 128, 255],
               [192, 0, 128, 255], [64, 128, 128, 255], [192, 128, 128, 255], [0, 64, 0, 255],
               [128, 64, 0, 255], [0, 192, 0, 255], [128, 192, 0, 255], [0, 64, 128, 255]]

class SegmentLayer(Layer):
    def __init__(self, parent=None, contenxt=None, **kwargs):
        self.image = None
        super(SegmentLayer, self).__init__(parent=parent, context=contenxt)
        self.setWindowOpacity(.5)
        self.cls_names = kwargs.get("classes", [])
        self.cls_palette = kwargs.get("palette", PALETTE)
        self.classes = {}
        self.refresh_img = False

    def gen_rgb_img(self):
        image = np.zeros((*self.raw.shape[:2], 4), dtype=np.uint8)
        image[...,3] = 0
        for cls, info in self.classes.items():
            if info["hide"]:
                continue
            color = self.cls_palette[cls]
            if np.sum(color) == 0:
                continue
            color.extend([255]*(4-len(color)))
            l, t, r, b = info["roi"]
            msk = self.raw[t:b,l:r] == cls
            color = self.cls_palette[cls]
            image[t:b,l:r][msk] = color
        return image

    def load_content(self, content, **kwargs):
        if isinstance(content, str):
            raw = cv2.imread(content, cv2.IMREAD_GRAYSCALE)
            if raw is None:
                return False, f"Load img:{content} failed"
        else:
            raw = content
        self.raw = raw
        ids = np.unique(raw)

        for id in ids:
            curr = raw == id
            ys, xs = np.nonzero(curr)
            self.classes[id] = {
                "roi": [xs.min(), ys.min(),xs.max()+1,ys.max()+1],
                "msk": curr,
                "hide": False
            }

        h, w = raw.shape[:2]
        self.raw = raw
        self.states.source = [0, 0, w, h]
        ar = h / w
        self.states.ar = ar
        self.image = QImage(self.gen_rgb_img(), w, h, QImage.Format_RGBA8888)
        return True, ""

    def get_icon(self):
        return resize2(self.image, 128)[0]

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
        icon = resize2(self.image, 128)[0]
        if not isinstance(icon, QImage):
            thumb = cv2.cvtColor(icon, cv2.COLOR_BGR2RGB)
            icon = QIcon(QPixmap.fromImage(QImage(thumb,thumb.shape[1], thumb.shape[0], QImage.Format_RGB888)))
        else:
            icon = QIcon(QPixmap.fromImage(icon))
        return icon


    def save(self, fpath):
        image = Image.fromarray(self.raw)
        image.save(fpath)