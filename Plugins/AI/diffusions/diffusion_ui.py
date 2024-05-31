import datetime
import logging
import os
import re
import time

import PySide6
from PIL import Image
import yaml
from PIL.PngImagePlugin import PngInfo
from PySide6.QtWidgets import QDialogButtonBox
from mmengine.model import revert_sync_batchnorm

from Plugins.AI.diffusions.diffusers_util import load_pipeline, run_from_generate_data
from mmseg.apis import init_model, inference_model
from PySide6 import QtWidgets, QtCore, QtGui

from utils.Config import Config
from utils.qtuitools import list_commbo


class Diffusion:
    def __init__(self, config_file=None, **kwargs):
        self.load_config(config_file)
        self.__latest_generate_data = ""
    def load_config(self, config_file):
        my_dir = os.path.dirname(os.path.abspath(__file__))
        if config_file is None:
            config_file = f"{my_dir}/diffusion.yaml"
        self.cfg_file = config_file
        with open(config_file, "r", encoding="utf-8") as fd:
            cfg = yaml.load(fd, Loader=yaml.FullLoader)
            self.cfg = Config(cfg)
        if "base" not in self.cfg:
            self.cfg["base"] = {}
        else:
            for k, v in self.cfg.base.items():
                v["path"] = self.abspath(v["path"])
                if "cfg" in v:
                    v["cfg"] = self.abspath(v["cfg"])

        if "lora" not in self.cfg:
            self.cfg["lora"] = {}
        else:
            for k, v in self.cfg.lora.items():
                v["path"] = self.abspath(v["path"])
        if "TI" not in self.cfg:
            self.cfg["TI"] = {}
        else:
            for k, v in self.cfg.TI.items():
                v["path"] = self.abspath(v["path"])
        if "vae" not in self.cfg:
            self.cfg["vae"] = {}
        else:
            for k, v in self.cfg.vae.items():
                v["path"] = self.abspath(v["path"])
        if "hash" not in self.cfg:
            self.cfg["hash"] = {}
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

    def abspath(self, model_id):
        mdir = os.path.dirname(os.path.abspath(__file__))
        fullp = f"{mdir}/{model_id}"
        if os.path.isfile(fullp) or os.path.isdir(fullp):
            return fullp
        return model_id
    def generate(self, info, n_generate=1):
        minfo = self.cfg.base[info["model"]]
        model_id = minfo['path']
        self.pipeline = load_pipeline(model_id, False, minfo.get("varitent", "fp32"), original_config_file=minfo.get("cfg",None))

        self.__latest_generate_data = info["generate_data"]
        return  run_from_generate_data(self.pipeline, info["generate_data"],
            loras = self.cfg.lora,
            vaes = self.cfg.vae,
            n_generate=n_generate)

    def latest_generate_data(self):
        return self.__latest_generate_data


def update_commbo(commbo:QtWidgets.QComboBox, names):
    curr = commbo.currentText()
    for i in range(commbo.count(), -1, -1):
        commbo.removeItem(i)
    commbo.addItems(names)
    if curr in names:
        idx = names.index(curr)
        commbo.setCurrentIndex(idx)


class SDDialog(QtWidgets.QDialog):
    def __init__(self, context=None, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("StableDiffusion")
        layout = QtWidgets.QFormLayout()
        if context is None:
            self.diffusion = Diffusion()
            self.win = None
        else:
            self.diffusion = context.win.stableDiffusion
            self.win = context.win
        self.setLayout(layout)
        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addWidget(QtWidgets.QLabel("基模"))
        self.base = QtWidgets.QComboBox(parent=self)
        h_layout.addWidget(self.base)
        self.load_btn = QtWidgets.QPushButton("载入数据")
        self.load_btn.clicked.connect(self.open_file)
        h_layout.addWidget(self.load_btn)
        self.refresh_btn = QtWidgets.QPushButton("刷新配置")
        self.refresh_btn.clicked.connect(self.refresh_config)
        h_layout.addWidget(self.refresh_btn)
        layout.addRow(h_layout)

        self.prompt = QtWidgets.QTextEdit(parent=self)
        layout.addRow("prompt", self.prompt)
        self.nprompt = QtWidgets.QTextEdit(parent=self)
        layout.addRow("negative", self.nprompt)

        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addWidget(QtWidgets.QLabel("CFG scale:"))
        self.cfg_scale = QtWidgets.QLineEdit(parent=self)
        h_layout.addWidget(self.cfg_scale)
        h_layout.addWidget(QtWidgets.QLabel("Size:"))
        self.g_width = QtWidgets.QLineEdit(parent=self)
        self.g_width.setValidator(QtGui.QIntValidator())
        self.g_height = QtWidgets.QLineEdit(parent=self)
        self.g_height.setValidator(QtGui.QIntValidator())
        h_layout.addWidget(self.g_width)
        h_layout.addWidget(QtWidgets.QLabel("X"))
        h_layout.addWidget(self.g_height)
        layout.addRow(h_layout)

        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addWidget(QtWidgets.QLabel("Clip skip"))
        self.clip_skip = QtWidgets.QLineEdit(parent=self)
        self.clip_skip.setValidator(QtGui.QIntValidator())
        h_layout.addWidget(self.clip_skip)
        h_layout.addWidget(QtWidgets.QLabel("Seed:"))
        self.seed = QtWidgets.QLineEdit(parent=self)
        self.seed.setValidator(QtGui.QIntValidator())
        h_layout.addWidget(self.seed)
        layout.addRow(h_layout)

        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addWidget(QtWidgets.QLabel("Sampler:"))
        self.sampler = QtWidgets.QComboBox(parent=self)
        h_layout.addWidget(self.sampler)
        h_layout.addWidget(QtWidgets.QLabel("Denoising strength:"))
        self.denoise_strength = QtWidgets.QLineEdit(parent=self)
        self.denoise_strength.setText("0")
        self.denoise_strength.setValidator(QtGui.QDoubleValidator())
        h_layout.addWidget(self.denoise_strength)
        layout.addRow(h_layout)

        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addWidget(QtWidgets.QLabel("VAE"))
        self.vae = QtWidgets.QComboBox(parent=self)

        h_layout.addWidget(self.vae)
        h_layout.addWidget(QtWidgets.QLabel("Texture inversion"))
        self.TI = QtWidgets.QComboBox(parent=self)
        h_layout.addWidget(self.TI)
        layout.addRow(h_layout)

        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addWidget(QtWidgets.QLabel("Steps"))
        self.steps = QtWidgets.QLineEdit(parent=self)
        self.steps.setValidator(QtGui.QIntValidator())
        h_layout.addWidget(self.steps)
        h_layout.addWidget(QtWidgets.QLabel("Num"))
        self.n_generate = QtWidgets.QLineEdit(parent=self)
        self.n_generate.setText("1")
        self.n_generate.setValidator(QtGui.QIntValidator())
        h_layout.addWidget(self.n_generate)
        layout.addRow(h_layout)

        self.lora_widgets = []
        loras = [""]
        loras.extend(list(self.diffusion.cfg.lora.keys()))
        self.max_loras = 5
        for i in range(self.max_loras):
            h_layout = QtWidgets.QHBoxLayout()
            h_layout.addWidget(QtWidgets.QLabel("LORA:"))
            lora = QtWidgets.QComboBox(parent=self)
            lora.addItems(loras)
            h_layout.addWidget(lora)
            h_layout.addWidget(QtWidgets.QLabel("wieght:"))
            weight = QtWidgets.QLineEdit(parent=self)
            weight.setValidator(QtGui.QDoubleValidator())
            h_layout.addWidget(weight)
            self.lora_widgets.append((lora, weight))
            layout.addRow(h_layout)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        '''创建对话框按钮'''
        self.buttonBox = QDialogButtonBox(QBtn)
        '''对话框确认信号连接到确认槽函数'''
        self.buttonBox.accepted.connect(self.accept)
        '''对话框取消按钮连接到取消槽函数'''
        self.buttonBox.rejected.connect(self.reject)
        self.layout().addWidget(self.buttonBox)

        self.generate_data = ""
        self.update_ui()
        gd = self.diffusion.latest_generate_data()
        if len(gd) != 0:
            self.load_generate_data(gd, raw=True)

    def refresh_config(self):
        self.diffusion.load_config(None)
        self.update_ui()

    def update_ui(self):
        # self.sampler.addItems(self.diffusion.cfg.samplers)
        update_commbo(self.sampler, self.diffusion.cfg.samplers)
        vaes = [""]
        vaes.extend(list(self.diffusion.cfg.vae.keys()))
        update_commbo(self.vae, vaes)

        tis = [""]
        tis.extend(list(self.diffusion.cfg.TI.keys()))
        update_commbo(self.TI, tis)

        update_commbo(self.base, list(self.diffusion.cfg.base.keys()))
        loras = [""]
        loras.extend(list(self.diffusion.cfg.lora.keys()))
        for i in range(self.max_loras):
            update_commbo(self.lora_widgets[i][0], loras)


    def accept(self) -> None:
        info = self.build_generate_data()
        ng = int(self.n_generate.text().strip())
        t = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.png")
        for i, image in enumerate(self.diffusion.generate(info, ng)):
            meta = PngInfo()
            meta.add_text(key="generate_data", value=self.generate_data)
            if self.win is None:
                image.save(f"results/{t}", pnginfo=meta)
            else:
                self.win.canvas.add_layer(image, layerName=f"diffusion_seg", layerType="sd",
                                          generate_data=self.generate_data
                                             )
        print("Done")

    def build_generate_data(self):
        model = self.base.currentText().strip()
        info = {"model":model}
        #Hires steps: 16, Hires upscale: 1.5, Hires upscaler: Latent, Denoising strength: 0.5
        params = [f"Steps: {self.steps.text().strip()}"]
        params.append(f"Size: {self.g_width.text().strip()}x{self.g_height.text().strip()}")
        params.append(f"Seed: {self.seed.text().strip()}")
        params.append(f"Model: {model}")
        params.append(f"Sampler: {self.sampler.currentText().strip()}")
        params.append(f"CFG scale: {self.cfg_scale.text().strip()}")
        params.append(f"Clip skip: {self.clip_skip.text().strip()}")
        for k,v in self.diffusion.cfg.hash.items():
            if v == model:
                params.append(f"Model hash: {k}")
                break
        params.append(f"Denoising strength: {self.denoise_strength.text().strip()}")
        params = ", ".join(params)
        loras = []
        for i in range(self.max_loras):
            lora, weight = self.lora_widgets[i]
            w = weight.text()
            if len(w) ==0 or float(w) <= 0:
                continue
            loras.append(f"<lora:{lora.currentText().strip()}:{w}>")
        prompt = self.prompt.toPlainText().strip()
        if len(loras) != 0:
            loras = ", ".join(loras)
            prompt = ", ".join([prompt, loras])
        info["generate_data"] = "\n".join([prompt,
                                           "Negative prompt: "+self.nprompt.toPlainText().strip(),
                                           params])
        self.generate_data = info["generate_data"]
        return info

    def dropEvent(self, event: PySide6.QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            filepath = event.mimeData().urls()[0].toLocalFile()
            self.load_generate_data(filepath)

    def open_file(self):
        file = QtWidgets.QFileDialog.getOpenFileName(parent=self)
        if file is None:
            return
        fpath = file[0]
        self.load_generate_data(fpath)

    def load_generate_data(self, fpath, raw=False):
        """
        加载civitai的generate data
        :return:
        """

        if raw:
            lines = raw.split("\n")
        else:
            if fpath.endswith(".png"):
                image = Image.open(fpath)
                if "generate_data" in image.text:
                    lines = image.text["generate_data"].split("\n")
                else:
                    return False
            else:
                with open(fpath) as fd:
                    lines = fd.readlines()

        pattern = re.compile("<lora:([\w|_]*:\d+\.*\d*)>")
        prompt = lines[0].strip()
        loras = pattern.findall(prompt)
        prompt = pattern.sub("", prompt)
        self.prompt.setText(prompt)
        self.nprompt.setText(lines[1].strip())
        self.params = lines[2].strip().replace("Negative prompt: ", "")

        if len(self.lora_widgets) > 0:
            commbo = self.lora_widgets[0][0]
            c_loras = list_commbo(commbo)
            idx = 0
            for lora in loras:
                name, alpha = lora.split(":")
                if name not in c_loras:
                    logging.error(f"Missing Lora: {name}")
                    continue
                self.lora_widgets[idx][0].setCurrentIndex(c_loras.index(name))
                self.lora_widgets[idx][1].setText(alpha)
                idx += 1
        model_found = (True, "")
        for p in self.params.split(","):
            name, value = p.strip().split(":")
            value = value.strip()
            name = name.lower()
            if name == "cfg scale":
                self.cfg_scale.setText(value)
            elif name == "steps":
                self.steps.setText(value)
            elif name == "size":
                w,h = value.lower().split("x")
                self.g_width.setText(w)
                self.g_height.setText(h)
            elif name == "seed":
                self.seed.setText(value)
            elif name == "model":
                c_base = list_commbo(self.base)
                if value not in c_base:
                    model_found= (False, value)
                else:
                    idx = c_base.index(value)
                    self.base.setCurrentIndex(idx)
            elif name == "sampler":
                c_sampler = list_commbo(self.sampler)
                if value not in c_sampler:
                    logging.error(f"Unknow Sampler model:{value}")
                else:
                    idx = c_sampler.index(value)
                    self.sampler.setCurrentIndex(idx)
            elif name == "clip skip":
                self.clip_skip.setText(value)
            elif name == "denoising strength":
                self.denoise_strength.setText(value)
            elif name == "model hash":
                if value in self.diffusion.cfg.hash:
                    value = self.diffusion.cfg.hash[value]
                    c_base = list_commbo(self.base)
                    idx = c_base.index(value)
                    self.base.setCurrentIndex(idx)
                    model_found = (True, value)
            else:
                logging.warning(f"Unknow params {p.strip()}")
        if not model_found[0]:
            logging.error(f"Base model:", model_found[1], "no found")


def Init(context, **kwargs):
    win = context.win
    obj = Diffusion(**kwargs)
    setattr(win, "stableDiffusion", obj)



def show(context):
    state = context.state
    rect = context.win.rect()
    c = SDDialog(context, parent=context.win)
    cx = rect.x() + rect.width() // 2
    cy = rect.y() + rect.height() // 2
    x = max(cx-150, 0)
    y = max(cy-150, 0)
    c.setGeometry(QtCore.QRect(x,y,300,300))
    # c.setWindowModality(Qt.ApplicationModal)
    c.raise_()
    c.show()





if __name__ == "__main__":
    import cv2
    import sys
    import logging
    logging.basicConfig(level=logging.DEBUG)
    app = QtWidgets.QApplication(sys.argv)
    from mmseg.apis import show_result_pyplot
    sd = SDDialog()
    sd.show()
    app.exec()
    # img_file = r"4b6bd4e955718.jpg"
    # img = cv2.imread(img_file)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # result = mmseg.run(img, data_only=True)
    # show_result_pyplot(
    #     mmseg.curr_model[0], img_file,result,draw_gt=False, draw_pred=True
    # )
    # print(type(result), result.shape)
    # print(type(mmseg.get_classes()))
    # palette = mmseg.get_palette()
    # cr = np.zeros_like(img, dtype=np.uint8)
    # for i, color in enumerate(palette):
    #     cr[result == i] = color
    # cv2.imwrite("CCC.jpg", cr)
