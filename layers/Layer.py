from PySide6 import QtCore
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QWidget

from utils.Config import Config


class Layer(QWidget):
    def __init__(self, context, *args, **kwargs):
        super(Layer, self).__init__(*args, **kwargs)
        self.context = context
        self.states = Config({
            "source": None,             # 内容需要显示的区域, list:[left, top, width, height]
            "view_rect": kwargs.get("view_rect", None),          # source在widget上的显示范围, list:[left, top, width, height]
            "scale": kwargs.get("scale", 1),            # 当前的缩放比例, str/float
            "transpose":[0,0],
            "ar": 1.0,                  # content 高/宽, float
            "raw":None,                 # 直接解码的数据内容，any
            "raw_sz":[0,0],              #raw 的尺寸 [w,h]
            "operation":""
        })
        self.__version = "0"
    def version(self):
        return self.__version

    def mouse_to_source(self, pt:QPointF):
        """
        将当前鼠标的位置(Global)映射到内容区域
        :param x:
        :param y:
        :return:
        """
        pt = self.mapFromGlobal(pt)
        source = self.states.source
        rect = self.states.view_rect
        tx, ty = self.states.transpose
        x = pt.x() - tx
        y = pt.y() - ty
        xscale = x / rect.width()
        yscale = y / rect.height()
        if xscale <0 or xscale > 1.0:
            return None, None
        if yscale <0 or yscale > 1.0:
            return None, None

        sx = source.width() * xscale + source[0]
        sy = source.height() * yscale + source[1]
        x = min(self.imw, source[0] + x * sx)
        y = min(self.imh, source[1] + y * sy)
        x = max(0, x)
        y = max(0, y)
        return x, y

    def load_content(self, content, **kwargs):
        return False, "No implement"

    def get_icon(self):
        return None, "No implement"

    def move(self, dx, dy):
        self.scroll(dx, dy)
        # self.states.source[0] += dx
        # self.states.source[1] += dy

    def zoom(self, x, y, dscale):
        rect = self.rect()
        source = self.opts["source"]
        scale = self.opts["scale"]
        focus = self.opts["focus"]
        show_rect = self.opts["show_rect"]
        ar = self.opts["ar"]
        rect_h = rect.height()
        rect_w = rect.width()

        _ar = rect_h / rect_w
        sw, sh = source.width(), source.height()
        if _ar > ar:
            rect_h = rect_w * ar
        else:
            rect_w = rect_h / ar
        if scale != "auto":
            sw = max(10, int(rect_w * scale))
            sw = min(sw, self.imw * 22)
            sh = sw * ar
        x = source.x()
        y = source.y()

        l = t = 0
        if rect_w < rect.width():
            l = (rect.width() - rect_w) * 0.5
        if rect_h < rect.height():
            t = (rect.height() - rect_h) * 0.5
        # t = (rect.height() - rh) * 0.5
        p.translate(l, t)

        wscale = source.width() / rect_w
        hscale = source.height() / rect_h
        if focus is not None and show_rect is not None:
            focus_x, focus_y = self.mapToImage(*focus)
            focus_x = max(0, min(self.imw, focus_x))
            focus_y = max(0, min(self.imh, focus_y))
            dx = (focus_x - source.x()) / source.width() * sw
            dy = (focus_y - source.y()) / source.height() * sh
            x = focus_x - dx
            y = focus_y - dy

            # self.opts["focus"] = None
        translate = self.opts["translate"]
        if translate is not None:
            x += translate[0] * wscale
            y += translate[1] * hscale
            self.opts["translate"] = None
        source = QtCore.QRect(x, y, sw, sh)
        self.opts["source"] = source
        rect = QtCore.QRect(rect.x(), rect.y(), rect_w, rect_h)
        self.opts["show_rect"] = rect

    def save(self, fpath):
        raise NotImplemented()