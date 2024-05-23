import logging
import os
import numpy as np
import torch
from PySide6.QtWidgets import QDialogButtonBox
from mmengine.model import revert_sync_batchnorm

from mmseg.apis import init_model, inference_model
from PySide6 import QtWidgets, QtCore
import cv2

class MMSegments:
    def __init__(self, model_dir=None):
        self.models = {}
        if model_dir is None:
            TORCH_HOME = os.environ.get("TORCH_HOME")
            root = f"{TORCH_HOME}/mmsegments"
            if not os.path.isdir(root):
                return
            model_dir = root
        for dir in os.listdir(model_dir):
            sdir = os.path.join(model_dir, dir)
            if not os.path.isdir(sdir):
                continue
            curr = {}
            for f in os.listdir(sdir):
                if f.endswith(".py"):
                    curr["config"] = os.path.join(sdir, f)
                elif f.endswith(".pth"):
                    curr["pth"] = os.path.join(sdir, f)
            if "config" in curr and "pth" in curr:
                self.models[dir] = curr
        self.curr_model = [None, ""]
        self.model_loaded = {}

    def get_methods(self):
        return list(self.models)

    def use(self, method, device="cuda"):
        if method in self.model_loaded:
            if self.model_loaded[method]["device"] != device:
                self.model_loaded[method]["model"].to(torch.device(device))
                self.model_loaded[method]["device"] = device
            self.curr_model = [self.model_loaded[method]["model"], method]
        else:
            if method not in self.models:
                return False
            minfo = self.models[method]
            cfg = minfo["config"]
            pth = minfo["pth"]
            model = init_model(cfg, pth, device=device)
            model.eval()
            if device == 'cpu':
                model = revert_sync_batchnorm(model)
            if self.curr_model[0] is not None:
                self.offload(self.curr_model[1], True)
            self.curr_model = [model, method]
            self.model_loaded[method] = {
                "device": device,
                "model": model
            }
        return True

    def get_classes(self):
        if self.curr_model[0] is None:
            return []
        return self.curr_model[0].dataset_meta["classes"]

    def get_palette(self):
        if self.curr_model[0] is None:
            return []
        return self.curr_model[0].dataset_meta["palette"]

    def offload(self, method, destroy=False):
        if method not in self.model_loaded:
            return  True
        if destroy:
            del self.model_loaded[method]
            if method == self.curr_model[1]:
                self.curr_model = [None, ""]
        else:
            m = self.model_loaded[method]
            if m["device"] == "cpu":
                return True
            model = m["model"]
            model.to(torch.device("cpu"))
            model = revert_sync_batchnorm(model)
            m["model"] = model
            m["device"] = "cpu"
        return True

    def run(self, img, data_only=True):
        cfg

class SegDialog(QtWidgets.QDialog):
    def __init__(self, context, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("MMsegmentations")
        layers = [""]
        layers.extend(context.win.canvas.layer_names())
        layout = QtWidgets.QFormLayout()
        self.layers_comb = QtWidgets.QComboBox(self)
        self.layers_comb.addItems(layers)
        layout.addRow("layers",self.layers_comb)
        if len(layers) >= 2:
            self.layers_comb.setCurrentIndex(1)

        self.methods_comb = QtWidgets.QComboBox(parent=self)
        methods = context.win.mmsegs.get_methods()
        self.methods_comb.addItems(methods)
        layout.addRow("method", self.methods_comb)
        self.setLayout(layout)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        '''创建对话框按钮'''
        self.buttonBox = QDialogButtonBox(QBtn)
        '''对话框确认信号连接到确认槽函数'''
        self.buttonBox.accepted.connect(self.accept)
        '''对话框取消按钮连接到取消槽函数'''
        self.buttonBox.rejected.connect(self.reject)
        self.layout().addWidget(self.buttonBox)

    def add_btn_box(self, layout):
        self.ok_btn = QtWidgets.QPushButton("确定")
        self.cancel_btn = QtWidgets.QPushButton("取消")
        w = QtWidgets.QWidget()
        l = QtWidgets.QHBoxLayout()
        l.addWidget(self.ok_btn)
        l.addWidget(self.cancel_btn)
        w.setLayout(l)
        layout.addWidget(w)
        self.ok_btn.clicked.connect(self.access)
        self.cancel_btn.clicked.connect(self.cancel)



def Init(context, **kwargs):
    win = context.win
    obj = MMSegments(**kwargs)
    logging.info(f"{obj.get_methods()} Load")
    setattr(win, "mmsegs", obj)



def do_seg(context):
    state = context.state
    rect = context.win.rect()
    c = SegDialog(context, parent=context.win)
    cx = rect.x() + rect.width() // 2
    cy = rect.y() + rect.height() // 2
    x = max(cx-150, 0)
    y = max(cy-150, 0)
    c.setGeometry(QtCore.QRect(x,y,300,300))
    # c.setWindowModality(Qt.ApplicationModal)
    c.raise_()
    c.exec()
    method = c.methods_comb.currentText()
    layer_name = c.layers_comb.currentText()
    if layer_name == "":
        return
    if method == "":
        return
    layer = context.win.canvas.get_layer(layer_name)
    if layer is None:
        return
    mmsegs = context.win.mmsegs
    mmsegs.use(method)
    if layer.raw.ndim == 2:
        img = cv2.cvtColor(layer.raw, cv2.COLOR_GRAY2RGB)
    elif layer.raw.shape[2] == 3:
        img = cv2.cvtColor(layer.raw, cv2.COLOR_BGR2RGB)
    elif layer.raw.shape[2] == 4:
        img = cv2.cvtColor(layer.raw, cv2.COLOR_BGRAY2RGB)
    else:
        state.win.log(f"Invalid layer with channgle:{layer.raw.shape[2]}","ERROR")
    result = mmsegs.run(img, data_only=True)
    context.win.canvas.add_layer(result, layerName=f"{layer_name}_seg", layerType="segment",
                                 cls_names=mmsegs.get_classes(),
                                 palette=mmsegs.get_palette()
                                 )




if __name__ == "__main__":
    import cv2
    from mmseg.apis import show_result_pyplot
    mmseg = MMSegments()
    mmseg.use("tattoo")
    img_file = r"4b6bd4e955718.jpg"
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mmseg.run(img, data_only=True)
    # show_result_pyplot(
    #     mmseg.curr_model[0], img_file,result,draw_gt=False, draw_pred=True
    # )
    # print(type(result), result.shape)
    # print(type(mmseg.get_classes()))
    palette = mmseg.get_palette()
    cr = np.zeros_like(img, dtype=np.uint8)
    for i, color in enumerate(palette):
        cr[result == i] = color
    cv2.imwrite("CCC.jpg", cr)
