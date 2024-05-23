import logging
import os
import numpy as np
import torch
import yaml
from PySide6.QtWidgets import QDialogButtonBox
from mmengine.model import revert_sync_batchnorm

from mmseg.apis import init_model, inference_model
from PySide6 import QtWidgets, QtCore
import cv2

from utils.Config import Config


class Diffusion:
    def __init__(self, config_file=None):
        my_dir = os.path.abspath(__file__)
        if config_file is None:
            config_file = f"{my_dir}/diffusion.yaml"
        self.cfg_file = config_file
        with open(config_file, "r") as fd:
            cfg = yaml.safe_load(fd)
            self.cfg = Config(cfg)
        if "base" not in self.cfg:
            self.cfg["base"] = {}
        if "lora" not in self.cfg:
            self.cfg["lora"] = {}
        if "TextureInversion" not in self.cfg:
            self.cfg["TextureInversion"] = {}

    def save(self):
        with open(self.cfg_file, "w") as fd:
            yaml.dump(self.cfg, fd)

    def add_models(self, key, models=[]):
        if isinstance(models, str):
            models = [models]
        save = False
        if key not in self.cfg:
            self.cfg[key] = []
        curr = self.cfg[key]

        for fpath in models:
            if fpath not in curr:
                curr.append(fpath)
                save = True
        if save:
            self.save()

    def generate(self, info):


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
