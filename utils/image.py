import cv2
import numpy as np
from PySide6.QtCore import QPoint
from PySide6.QtGui import QImage, QColor, QPainter


def resize2(img, dest_sz, square=False, border=cv2.BORDER_CONSTANT, value=0):
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
    elif isinstance(img, QImage):
        w = img.width()
        h = img.height()
    scale = dest_sz / max(w,h)
    w = int(w * scale + 0.5)
    h = int(h * scale + 0.5)
    if isinstance(img, np.ndarray):
        img = cv2.resize(img, (w, h))
    elif isinstance(img, QImage):
        w = img.width()
        h = img.height()
        img = img.scaled(w, h)
    pt = pb = pl = pr = 0
    if square:
        h, w = img.shape[:2]
        if w > h:
            d = w - h
            pt = d >> 1
            pb = d - pl
        else:
            d = h - w
            pl = d >> 1
            pr = d - pl
        if isinstance(img, np.ndarray):
            img = cv2.copyMakeBorder(img, pt, pb, pl, pr, border, value=value)
        elif isinstance(img, QImage):
            new_img = QImage(w+pl+pr, h+pt+pb, img.format())
            new_img.fill(QColor(0,0,0,0))
            p = QPainter(new_img)
            p.drawImage(QPoint(pl, pt), img)
            img = new_img


    return img, (scale, pt, pb, pl, pr)