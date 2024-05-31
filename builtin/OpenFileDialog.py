import os

from PySide6 import QtWidgets, QtCore
from PySide6 import QtGui
from PySide6.QtCore import Qt, Signal, QAbstractTableModel, QItemSelectionModel
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QVBoxLayout, QFileIconProvider, QDialogButtonBox, QStyle, QHeaderView, QAbstractItemView


class FileTable(QAbstractTableModel):
    def __init__(self, parent=None, files=[]):
        super(FileTable, self).__init__(parent=parent)
        self.files = []
        self.set_files(files)
        self.headers = ["名称","大小","修改日期"]

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self.headers[section]
        return super().headerData(section, orientation, role)

    @classmethod
    def get_file_info(self, fpath):
        file_info = QtCore.QFileInfo(fpath)
        file_icon = QFileIconProvider()
        icon = QIcon(file_icon.icon(file_info))

        return icon, file_info

    def set_files(self, files):
        self.files = []
        for fpath in files:
                icon, file_info = self.get_file_info(fpath)
                self.files.append([icon, file_info.fileName(), file_info.size(),file_info.metadataChangeTime(), fpath])

    def get_row_info(self, idx):
        return self.files[idx]

    def data(self, index, role):
        if role == Qt.DecorationRole and index.column() == 0:
            return self.files[index.row()][0]
        elif role == Qt.DisplayRole:
            return self.files[index.row()][index.column()+1]

    def rowCount(self, parent = ...):
        return len(self.files)

    def columnCount(self, parent = ...):
        if len(self.files) > 0:
            return len(self.files[0]) - 2
        return 0


class OpenFileDialog(QtWidgets.QDialog):
    def __init__(self, parent=None,
                 entry="/", layer_name="",acc=None, rej=None, accepts=["dir", "file"],
                 enable_multi=True):
        super(OpenFileDialog, self).__init__(parent=parent)
        self.setWindowTitle("路径选择")

        self.accept_list = accepts
        self.acc_func = acc
        self.rej_func = rej
        self.selected_files = []
        layout = QtWidgets.QGridLayout()

        nbtn = QtWidgets.QPushButton()
        nbtn.setMaximumWidth(5)
        nbtn.hide()
        layout.addWidget(nbtn, 0, 0, 1, 1)

        self.pre_btn = QtWidgets.QPushButton(icon=QtWidgets.QApplication.style().standardIcon(QStyle.SP_ArrowBack))
        self.pre_btn.clicked.connect(self.goto_pre)
        layout.addWidget(self.pre_btn, 0,layout.columnCount(),1,1)

        self.upper_btn = QtWidgets.QPushButton(icon=QtWidgets.QApplication.style().standardIcon(QStyle.SP_ArrowUp)) #获取QT内置icon
        self.upper_btn.clicked.connect(self.goto_parent)
        layout.addWidget(self.upper_btn, 0,layout.columnCount(),1,1)
        self.path_edit = QtWidgets.QLineEdit(parent=self)
        self.path_edit.setText(entry)
        layout.addWidget(QtWidgets.QLabel("路径:"),0,layout.columnCount(),1,1)
        layout.addWidget(self.path_edit,0,layout.columnCount(),1,15)
        self.path_edit.textChanged.connect(self.change_path)

        self.layer_label = QtWidgets.QLabel("图层名：",parent=self)
        layout.addWidget(self.layer_label, 0, layout.columnCount(), 1, 1)
        self.layer_edit = QtWidgets.QLineEdit(parent=self)
        self.layer_edit.setText(layer_name)
        layout.addWidget(self.layer_edit, 0, layout.columnCount(), 1, 3)

        self.layer_type_label = QtWidgets.QLabel("图层类型：", parent=self)
        layout.addWidget(self.layer_type_label, 0, layout.columnCount(), 1, 1)
        self.layer_type_combo = QtWidgets.QComboBox(parent=self)
        layout.addWidget(self.layer_type_combo, 0, layout.columnCount(), 1, 3)

        self.table_model = FileTable(parent=self)
        self.table_view = QtWidgets.QTableView()
        self.table_view.setModel(self.table_model)
        self.table_view.doubleClicked.connect(self.open_entry)
        self.table_view.resizeColumnsToContents()
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        if not enable_multi:
            self.table_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table_view,1,0,30,-1)


        #latest line
        self.save_label = QtWidgets.QLabel("名称:")
        self.save_edit = QtWidgets.QLineEdit()
        layout.addWidget(self.save_label, layout.rowCount(), 0,1,1)
        layout.addWidget(self.save_edit, layout.rowCount()-1, 1,1,5)

        self.acc_btn = QtWidgets.QPushButton(parent=self, text="确定")
        self.acc_btn.clicked.connect(self.do_accept)
        self.cancel_btn = QtWidgets.QPushButton(parent=self, text="取消")
        layout.addWidget(self.acc_btn, layout.rowCount()-1, layout.columnCount()-2,1,1)
        layout.addWidget(self.cancel_btn, layout.rowCount()-1, layout.columnCount()-1,1,1)
        self.cancel_btn.clicked.connect(self.do_rej)

        self.setLayout(layout)
        self.show()
        self.curr_path = None
        self.pre_path = None
        if entry != "":
            self.open_path(entry)
        # self.open_path(entry)


    def do_accept(self):
        layer_name = self.layer_edit.text()
        ret = []
        selects = self.table_view.selectionModel().selectedRows()
        for index in selects:
            fpath = self.table_model.get_row_info(index.row())[-1]
            if os.path.isdir(fpath) and "dir" in self.accept_list:
                ret.append(fpath)
            elif os.path.isfile(fpath) and "file" in self.accept_list:
                ret.append(fpath)
        # if len(ret) == 0:
        #     fpath = self.curr_path
        #     if os.path.isdir(fpath) and "dir" in self.accept_list:
        #         ret.append(fpath)
        #     elif os.path.isfile(fpath) and "file" in self.accept_list:
        #         ret.append(fpath)
        self.selected_files = ret
        if self.acc_func is not None:
            self.acc_func(self.curr_path, layer_name)
        self.done(0)
    def do_rej(self):
        if self.rej_func is not None:
            self.rej_func()
        self.done(0)

    def goto_parent(self):
        if self.curr_path is None:
            return
        dirname = os.path.dirname(self.curr_path)
        if os.path.isdir(dirname):
            self.open_path(dirname)

    def goto_pre(self):
        if self.pre_path is None or not os.path.isdir(self.pre_path):
            return
        self.open_path(self.pre_path)

    def open_path(self, epath):
        self.pre_path = self.curr_path
        self.curr_path = epath
        self.path_edit.setText(epath)
        if os.path.isdir(epath):
            files = [os.path.join(epath, f) for f in os.listdir(epath)]
            self.table_model = FileTable(parent=self, files=files)
            self.table_view.setModel(self.table_model)
        else:
            self.do_accept()

    def open_entry(self):
        curr_row = self.table_view.selectionModel().currentIndex()
        fpath = self.table_model.get_row_info(curr_row.row())[-1]
        self.open_path(fpath)

    def change_path(self):
        txt = self.path_edit.text()
        txt = txt.replace('"', '').replace("file:///",'')
        if txt == self.curr_path:
            return
        if os.path.isdir(txt) and "dir" in self.accept_list:
            self.open_path(txt)
        elif os.path.isfile(txt) and "file" in self.accept_list:
            dir_path = os.path.dirname(txt)
            self.open_path(dir_path)
            self.path_edit.textChanged.connect(None)
            self.path_edit.setText(os.path.dirname(txt))
            file = os.path.basename(txt)
            if not self.save_edit.isHidden():
                self.save_edit.setText(file)
            model = self.table_view.selectionModel()
            for i in range(self.table_model.rowCount()):
                if self.table_model.get_row_info(i)[1] == file:
                    # self.table_view.scrollTo(self.table_model.index(i, 0))
                    for c in range(self.table_model.columnCount()):
                        index = self.table_model.index(i, c)
                        model.select(index, QItemSelectionModel.Select)
                    break
            self.path_edit.textChanged.connect(self.change_path)




def get_exist_file(parent=None,default_path="/",
                   default_laye_name="",
                   accepts=["dir", "file"],
                   multi_ok=True,
                   layer_types=[]):
    d = OpenFileDialog(parent=parent,entry=default_path,  layer_name=default_laye_name, accepts=accepts,enable_multi=multi_ok)
    d.save_label.hide()
    d.save_edit.hide()
    if len(layer_types) == 0:
        d.layer_type_combo.hide()
    else:
        d.layer_type_combo.addItems(layer_types)
    d.exec()
    ret = {
        "selected": d.selected_files,
        "directory": d.path_edit.text()
    }
    if not d.layer_edit.hide():
        ret["layer"] = d.layer_edit.text()
    if not d.save_edit.hide():
        ret["filename"] = d.save_edit.text()
    if not d.layer_type_combo.hide():
        ret["layer_type"] = d.layer_type_combo.currentText()
    return ret

def get_save_file(parent=None, default_path="/", default_laye_name="", accepts=["dir", "file"]):
    d = OpenFileDialog(parent=parent,entry=default_path, layer_name=default_laye_name, accepts=accepts, enable_multi=False)
    d.layer_edit.hide()
    d.layer_label.hide()
    d.exec()
    ret = {
        "selected": d.selected_files,
        "directory": d.path_edit.text()
    }
    if not d.layer_edit.hide():
        ret["layer"] = d.layer_edit.text()
    if not d.save_edit.hide():
        ret["filename"] = d.save_edit.text()
    return ret


if __name__ == "__main__":
    import sys


    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()

    app.exec()
