import json
import os.path
from collections import OrderedDict

from PySide6 import QtWidgets

from builtin import OpenFileDialog


def new_project(context):
    win = context.win
    info = OpenFileDialog.get_save_file(win,"/","")
    print("===>", info)
    path_list = info["selected"]
    if len(path_list) == 0:
        save_name = info["filename"]
        if len(save_name.strip()) == 0:
            return
        directory = info["directory"]
        save_file = os.path.join(directory, save_name)
    else:
        save_file = path_list[0]
    if os.path.isfile(save_file):
        win.log(f"{save_file} Already exist!", win.LOG_ERROR)
        dlg = QtWidgets.QMessageBox(win)
        dlg.setWindowTitle("覆盖文件")
        dlg.setText('文件已存在，确认覆盖？')
        dlg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        button = dlg.exec()
        dlg.done(0)
        if button != 16384:
            return
    if os.path.isdir(save_file):
        win.log(f"{save_file} is Directory", win.LOG_ERROR)
        return
    if not save_file.endswith(".vsproj"):
        save_file+=".vsproj"
    print("XXX",list(context.state.keys()))
    context.project_file = save_file
    win.save_object()

def open_project(context):
    win = context.win
    info = OpenFileDialog.get_save_file(win,"/","")
    path_list = info["selected"]
    if len(path_list) != "":
        return
    path = path_list[0]
    if os.path.isdir(path):
        win.log("File path is Directory", win.LOG_ERROR)
        return
    win.open_object(path)

def new_layer(context):
    win = context.win
    layers = context.state.layers
    n_layer = len(layers)
    info = OpenFileDialog.get_exist_file(win,"/", f"layer_{n_layer + 1}", multi_ok=False,
                                        layer_types=list(context.layer_pkg.keys()))

    layer_name = info["layer"]
    directory = info["directory"]
    path_list = info["selected"]
    if layer_name in layers:
        win.log(f"{layer_name} already exist!", win.LOG_ERROR)
        return False
    if len(path_list) < 1:
        return False
    if layer_name == "":
        win.log(f"Invalid layer name:{layer_name}")
        return False
    path = path_list[0]
    state = context.state
    file_map = state["file_map"]
    files = state["files"]
    refresh_list = False
    if os.path.isdir(path):
        for f in os.listdir(path):
            fname = os.path.splitext(f)[0]
            if fname not in files:
                refresh_list = True
                file_map[fname] = {"layers": OrderedDict(), "label": "normal"}
            file_map[fname]["layers"][layer_name] = {"path":os.path.join(path, f), "type":info.get("layer_type", None)}
            files.append(fname)

    elif os.path.isfile(path):
        refresh_list = True
        curr = state["curr_item"]
        if curr is None:
            file = os.path.basename(path)
            fname = os.path.splitext(file)[0]
            file_map[fname] = {"layers": OrderedDict(), "label": "normal"}
            curr = fname
            state["curr_item"] = curr
        file_info = file_map[curr]
        file_info.layers[layer_name] = {"path":path, "type":info.get("layer_type", None)}

    layers.append(layer_name)
    if refresh_list:
        win.reload_item_list()
    win.reload_current()
    return True

