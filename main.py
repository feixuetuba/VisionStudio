import json
import logging
import os.path
from collections import OrderedDict
from functools import partial

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import QEvent
from PySide6.QtGui import QKeySequence, QStandardItem
from PySide6.QtWidgets import QAbstractItemView

from builtin.MultiLayerViewer import MultilayerViewer
from utils import load_attr
from utils.Config import Config

def load_func(str_pkg):
    *pkg, func = str_pkg.split(".")
    pkg = ".".join(pkg)
    func = load_attr(pkg, func)
    return func

class MainWindow(QtWidgets.QMainWindow):
    LOG_VISUAL = 0
    LOG_DEBUG = 1
    LOG_INFO = 2
    LOG_WARN = 3
    LOG_ERROR = 4
    LOG_FATAL = 5

    def __init__(self):
        super(MainWindow, self).__init__()
        self.__temp_project = ".temp.project"
        self.events_listeners = {
            "item_changed":[]
        }
        self.reset_states()
        if os.path.isfile(self.__temp_project):
            self.load_project(self.__temp_project)
        self.setWindowTitle("VisionStudio")
        self.setGeometry(100, 100, 1920, 768)

        self.spliter = QtWidgets.QSplitter(parent=self)

        self.file_items = QtGui.QStandardItemModel()
        self.files_view = QtWidgets.QListView(parent=self)
        self.files_view.setModel(self.file_items)
        self.files_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.files_view.selectionModel().selectionChanged.connect(self.change_curr_item)
        self.spliter.addWidget(self.files_view)

        self.canvas = MultilayerViewer(parent=self, context=self.states)
        self.spliter.addWidget(self.canvas)

        self.spliter.setStretchFactor(0, 2)
        self.spliter.setStretchFactor(1, 8)
        self.setCentralWidget(self.spliter)
        self.curr_dir = None

        self.__LoadConfig()

        self.reload_item_list()

        state = self.states.state
        if state.curr_item is not None:
            idx = state.files.index(state.curr_item)
            state.curr_item = None
            self.files_view.setCurrentIndex(self.file_items.index(idx,0))
        self.log("Ready", self.LOG_INFO)

    def __LoadConfig(self):
        mbar = self.menuBar()
        menu_map = {"ROOT":mbar}
        with open("config.json", encoding="utf-8") as fd:
            meta = json.load(fd)
        win_loc = meta["window location"]
        self.setGeometry(QtCore.QRect(*win_loc))
        self.states.language = meta["language"]
        self.operations = {}
        for plugin in meta["Plugins"]:
            if os.path.isfile(plugin):
                self.__load_plugin_file(plugin, menu_map)
            elif os.path.isdir(plugin):
                for f in os.listdir(plugin):
                    if f.endswith(".plugin.json"):
                        self.__load_plugin_file(f"{plugin}/{f}", self.operations)

    def __load_plugin(self, pcfg, menu_map):
        strloc = pcfg["location"]
        logging.debug(f"Load {strloc}")
        if strloc in menu_map:
            curr = menu_map[strloc]
        else:
            locations = strloc.split("->")
            curr = menu_map["ROOT"]
            n = len(locations)
            for i in range(n):
                _currloc = "->".join(locations[:i+1])
                if _currloc in menu_map:
                    curr = menu_map[_currloc]
                else:
                    _menu = curr.addMenu(locations[i])
                    curr = _menu
                    menu_map[_currloc] = curr

        name = pcfg[self.states.language]
        full_path = f"{strloc}->{name}"
        curr = curr.addAction(name)
        if "icon" in pcfg:
            pass
        # curr.setText(name)
        func = load_func(pcfg["func"])
        # *pkg, func = pcfg["func"].split(".")
        # pkg = ".".join(pkg)
        # func = load_attr(pkg, func)
        curr.triggered.connect(partial(func, context=self.states))
        if "shortcut" in pcfg:
            curr.setShortcut(QKeySequence(pcfg["shortcut"]))
        # print("ADD action", full_path)
        menu_map[full_path] = curr

    def __load_plugin_file(self, fpath, menu_map):
        with open(fpath, "r", encoding="utf-8") as fd:
            meta = json.load(fd)
        for init_func in meta.get("init", []):
            func = load_func(init_func)
            func(self.states)
        for event, listeners in meta.get("listener", {}).items():
            if event not in self.events_listeners:
                self.log(f"No such event:{event}", self.LOG_WARN)
                continue
            for listener in listeners:
                self.events_listeners[event].append(load_func(listener))
        for pcfg in meta["plugins"]:
            self.__load_plugin(pcfg, menu_map)

    def reset_states(self):
        layer_pkg = {}
        for dir in os.listdir("layers"):
            cfg_f = f"layers/{dir}/layer_desc.json"
            if not os.path.isfile(cfg_f):
                continue
            with open(cfg_f, "r", encoding="utf-8") as fd:
                cfg = json.loads(fd.read())
                layer_pkg[cfg["name"]] = {"pkg": cfg["pkg"], "supported": cfg["supported"]}
        self.states = Config({
            "win": self,
            "state": {
                "file_map": OrderedDict(),
                "files":[],
                "curr_item": None,
                "layers":[],
                "classes":[]
            },
            "layer_pkg":layer_pkg,
            "project_file":None
        })

    def reload_item_list(self):
        state = self.states.state
        files = state.files
        self.file_items.clear()
        if len(files) == 0:
            return
        for file in files:
            item = QStandardItem(file)
            item.setEditable(False)
            self.file_items.appendRow(item)
        if state.curr_item is None:
            state.curr_item = files[0]
        self.reload_current()
        self.spliter.refresh()

    def reload_current(self):
        state = self.states.state
        if state.curr_item is None:
            return
        self.canvas.clean()
        curr = state.file_map[state.curr_item]
        for layer, fpath in curr.layers.items():
            self.canvas.add_layer(fpath,layer)

    def log(self, msg, level ,timeout=-1):
        level = ["VISIT","DEBUG","INFO","WAR","ERROR","FATAL"][level]
        self.statusBar().showMessage(f"[{level}]{msg}", timeout)

    def change_curr_item(self, item):
        state = self.states.state
        curr_selected = self.files_view.selectionModel().currentIndex().row()
        fname = state.files[curr_selected]
        if fname == state.curr_item:
            return True
        state.curr_item = fname
        self.reload_current()
        for listener in self.events_listeners["item_changed"]:
            listener(self.states)

    def save_object(self):
        project_file = self.states.project_file
        if project_file is None:
            project_file = self.__temp_project
        with open(project_file, "w") as fd:
            json.dump(self.states.state, fd, indent=4, ensure_ascii=False)

    def load_project(self, fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as fd:
                meta = json.load(fd)
                self.states.state = meta
        except Exception as e :
            self.log("Load cache failed", self.LOG_WARN)
        self.states.project_file = fpath

    def closeEvent(self, event) -> None:
        self.save_object()
        super(MainWindow, self).closeEvent(event)

    def event(self, event):
        super().event(event)
        etype = event.type()
        if etype not in [QEvent.KeyRelease]:
            return False
        self.canvas.key_event(event)
        return True

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()



