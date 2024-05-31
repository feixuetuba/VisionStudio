import os
import sys

import cv2
import numpy as np
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import QPoint, QEvent
from PySide6.QtGui import QImage, QCursor, QStandardItem, QPixmap, QIcon, QKeyEvent
from PySide6.QtWidgets import QScrollArea

from layers.ImageLayer import ImageLayer
from utils import load_instance
from utils.Config import Config


class LayerItem(QStandardItem):
    def __init__(self, *args, **kwargs):
        super(LayerItem, self).__init__(*args, **kwargs)
        self.layer = None

    def setLayer(self, layer):
        self.layer = layer


class Multilayer(QtWidgets.QWidget):
    def __init__(self, context, parent=None, max_sz=10000):
        super(Multilayer, self).__init__(parent=parent)
        self.layers = []
        self.context = context
        self.states = Config({
            "transpose": [0, 0],
            "scroll":[0,0],
            "scale": 1,
            "full_sz":None,
            "canvas":{
                "rect":None
            }
        })

        self.max_sz = max_sz
        self.reset()

    def reset(self):
        rect = self.rect()
        sz = min(rect.width(), rect.height())
        self.states = Config({
            "transpose": [0, 0],
            "scroll":[0,0],
            "scale": 1,
            "full_sz":None,
            "canvas":{
                "rect":QtCore.QRect(rect.x(), rect.y(), sz, sz)
            }
        })
        for layer in self.layers:
            layer.setParent(None)
            del layer
        self.layers = []
        self.index = []
        self.mousePressed = False

    def add_layer(self, value, layer_type=None, **kwargs):
        if layer_type is None:
            success = False
            for layer_info in self.context.layer_pkg.values():
                layer = load_instance(layer_info["pkg"], **kwargs, parent=self)
                success, msg = layer.load_content(value)
                if success:
                    break
            if not success:
                return False, msg
        else:
            layer_type = self.context.layer_pkg[layer_type]["pkg"]
            layer = load_instance(layer_type, **kwargs, parent=self)
            success, msg = layer.load_content(value)
        if not success:
            return False, f"Load {value} failed"
        layer.setGeometry(self.states.canvas.rect)
        self.layers.append(layer)
        if self.states.full_sz is None:
            rect = self.states.canvas.rect
            w, h = rect.width(), rect.height()
            self.states.full_sz = layer.states.raw_sz
            self.states.scale = max(w, h) / max(layer.states.raw_sz)

        return True, layer

    def clear_op(self, idx=-1):
        if idx < 0:
            for layer in self.layers:
                layer.states.operation = ""
        else:
            self.layers[idx].states.operation = ""

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if len(self.layers) == 0:
            self.rect()
        else:
            scroll_x, scroll_y = self.cal_canves_rect()
            for layer in self.layers:
                layer.setGeometry(self.states.canvas.rect)
                layer.update()
            self.scroll(scroll_x, scroll_y)

    def cal_canves_rect(self, focus_x=-1, focus_y=-1):
        rect = self.rect()
        crect = self.states.canvas.rect
        scale = self.states.scale
        full_sz = self.states.full_sz
        nw = int(full_sz[0] * scale + 0.5)
        nh = int(full_sz[1] * scale + 0.5)
        if focus_x < 0 or focus_y < 0:
            focus_x = crect.width() // 2
            focus_y = crect.height() // 2
        fx, fy = self.map_to_canvas(focus_x, focus_y)
        fx /= crect.width()
        fy /= crect.height()

        x = focus_x - nw * fx
        y = focus_y - nh * fy

        x -= rect.x()
        y -= rect.y()
        sx = min(0, x)
        sy = min(0, y)
        x = max(0, x)
        y = max(0, y)
        self.states.canvas.rect = QtCore.QRect(x, y, nw, nh)
        return sx, sy

    def map_to_canvas(self, x, y):
        crect = self.states.canvas.rect
        scroll = self.states.scroll
        dx = x - crect.x() - scroll[0]
        dy = y - crect.y() - scroll[1]
        return dx, dy

    def zoom(self, delta):
        full_sz = self.states.full_sz
        if full_sz is None:
            return
        pos = self.mapFromGlobal(QCursor.pos())
        scale = self.states.scale + 0.1 * delta
        max_sz = max(full_sz)
        smax_sz = max(15, max_sz * scale)
        self.states.scale = smax_sz / max_sz
        scroll_x, scroll_y = self.cal_canves_rect(pos.x(), pos.y())
        for layer in self.layers:
            layer.setGeometry(self.states.canvas.rect)
            layer.update()
        self.states.scroll = [scroll_x, scroll_y]
        self.scroll(scroll_x, scroll_y)

    def move(self, dx, dy):
        for layer in self.layers:
            if layer.isVisible():
                layer.move(dx, dy)
                layer.update()
        # x,y = self.states.scroll
        # self.states.scroll = [dx+x, dy+y]
        # self.scroll(dx, dy)

class MultilayerViewer(QtWidgets.QWidget):
    def __init__(self, context: Config, parent=None):
        super(MultilayerViewer, self).__init__(parent=parent)
        self.context = context
        self.setGeometry(0, 0, 640, 480)
        self.curr_mouse_pos = (0, 0)

        self.setContentsMargins(0, 0, 0, 0)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        layout = self.layout()

        spliter = QtWidgets.QSplitter(parent=self)
        spliter.setOrientation(QtCore.Qt.Horizontal)
        spliter.setSizes(QtCore.QIntList(100000, 100000))

        multi_viewer = Multilayer(parent=self, context=self.context)
        self.multi_viewer = multi_viewer
        spliter.addWidget(self.multi_viewer)

        v_spliter = QtWidgets.QSplitter()
        v_spliter.setOrientation(QtCore.Qt.Vertical)
        spliter.addWidget(v_spliter)

        layers_view = QtWidgets.QListView(parent=self)
        layers_view.setWindowTitle("Layers")
        self.layer_items = QtGui.QStandardItemModel()
        layers_view.setModel(self.layer_items)
        # layers_view.model().itemChanged.connect(
        #     lambda x: x.layer.show() if x.checkState() == QtCore.Qt.Checked else x.layer.hide())
        layers_view.model().itemChanged.connect(self.switch_layer_visible)


        self.layers_view = layers_view
        v_spliter.addWidget(layers_view)

        layout.addWidget(spliter)
        self.spliter = spliter

        spliter.setStretchFactor(0, 10)
        spliter.setStretchFactor(1, 2)
        self.mouse_pos = (0, 0)
        self.states = Config({
            "globals":["move", "zoom"],
            "operation":"",
            # "scale": 1.0
            })
        self.sp = spliter

    def switch_layer_visible(self, x):
        if x.checkState() == QtCore.Qt.Checked:
            x.layer.show()
        else:
            x.layer.hide()
    def set_curr_op(self, op, is_global=True):
        if op in self.states.globals and is_global:
            self.multi_viewer.clear_op()
            self.states.operation = op
        else:
            selected = self.multi_viewer.multi_viewer.selectionModel().selection()
            if len(selected) != 0:
                selected.layer.states.operation = op

    def add_layer(self, content, layerName=None, layerType=None, **kwargs):
        idx = self.layer_items.rowCount()
        name = f"Layer{idx}" if layerName is None else layerName
        success, layer = self.multi_viewer.add_layer(content, layerType, **kwargs)
        if not success:
            return success, layer
        item = LayerItem(name)
        item.setEditable(True)
        item.setIcon(layer.get_icon())
        item.setCheckable(True)
        self.layer_items.insertRow(0,item)

        item.setLayer(layer)
        item.setCheckState(QtCore.Qt.Checked)
        self.layers_view.clearSelection()
        self.layers_view.setCurrentIndex(self.layers_view.model().index(idx, 0))
        self.spliter.refresh()
        return True, layer

    def clean(self):
        self.multi_viewer.setGeometry(self.rect())
        self.multi_viewer.reset()
        self.layer_items.clear()
        # self.states.scale = 1.0
    def key_event(self, eve):
        key = QKeyEvent(eve)
        if key.key() == QtCore.Qt.Key_Space:
            count = self.layer_items.rowCount()
            index = self.layers_view.currentIndex()
            next = (index.row() - 1) % count
            for i in range(count):
                if i != next:
                    self.layer_items.item(i, 0).setCheckState(QtCore.Qt.Unchecked)
            self.layer_items.item(next, 0).setCheckState(QtCore.Qt.Checked)
            self.layers_view.setCurrentIndex(self.layers_view.model().index(next, 0))

        return False

    def layer_names(self):
        n = self.layer_items.rowCount()
        names = []
        for i in range(n):
            name = self.layer_items.item(i, 0).text()
            names.append(name)
        return  names

    def get_layer(self, name):
        n = self.layer_items.rowCount()
        for i in range(n):
            curr = self.layer_items.item(i, 0)
            if name == curr.text():
                return curr.layer
        return None

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        self.multi_viewer.zoom(delta)

    def mousePressEvent(self, event):
        self.latest_pos = event.position()
        self.mousePressed = True

    def mouseReleaseEvent(self, event):
        self.mousePressed = False

    def mouseMoveEvent(self, event):
        curr_pos = event.position()
        delta = curr_pos - self.latest_pos
        self.multi_viewer.move(delta.x(), delta.y())
        self.latest_pos = curr_pos

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.multi_viewer.setGeometry(self.rect())





if __name__ == "__main__":
    import json
    layer_pkg = {}
    for dir in os.listdir("../layers"):
        cfg_f = f"../layers/{dir}/layer_desc.json"
        if not os.path.isfile(cfg_f):
            continue
        with open(cfg_f, "r") as fd:
            cfg = json.loads(fd.read())
            layer_pkg[cfg["name"]] = {"pkg":cfg["pkg"], "supported":cfg["supported"]}
    app = QtWidgets.QApplication(sys.argv)
    fpath = "./images"
    if fpath == "":
        exit(-1)
    layers_window = MultilayerViewer(Config({
        "layer_pkg":layer_pkg
    }))
    layers_window.setWindowTitle("Layers")
    layers_window.setGeometry(100, 100, 512, 512)
    layers_window.clean()
    for f in os.listdir(fpath):
        success, msg = layers_window.add_layer(f"{fpath}/{f}")
        if not success:
            print(msg)
        else:
            print(f"Load {fpath}/{f} success")
    layers_window.show()
    app.exec()
