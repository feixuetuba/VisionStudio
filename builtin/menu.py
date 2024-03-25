import json
import os.path
from collections import OrderedDict

from builtin import OpenFileDialog


def new_project(context):
    win = context.win
    path_list, layer = OpenFileDialog.get_save_file(win,"/","")
    if len(path_list) != "":
        return
    path = path_list[0]
    if os.path.isdir(path):
        win.log("File path is Directory", win.LOG_ERROR)
        return
    win.open_object(path)


def open_project(context):
    win = context.win
    path_list, layer = OpenFileDialog.get_save_file(win,"/","")
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
    path_list, layer_name = OpenFileDialog.get_exist_file(win,"/", f"layer_{n_layer + 1}", multi_ok=False)
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
            file_map[fname]["layers"][layer_name] = os.path.join(path, f)
            files.append(fname)

    elif os.path.isfile(path):
        curr = state["curr_item"]
        file_info = files[curr]
        file_info.layers[layer_name] = path

    layers.append(layer_name)
    if refresh_list:
        win.reload_item_list()
    win.reload_current()
    return True

