import logging
from functools import partial

from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QKeySequence, QPalette, QColor
from PySide6.QtWidgets import QDialogButtonBox, QStyle

class EditWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setLayout(QtWidgets.QGridLayout())
        self.ids = []

    def append_row(self, name=""):
        layout = self.layout()
        e = QtWidgets.QLineEdit(parent=self)
        e.setText(name)
        b = QtWidgets.QPushButton(icon=QtWidgets.QApplication.style().standardIcon(QStyle.SP_LineEditClearButton))
        if layout is None:
            layout = self.layout()
        n = layout.rowCount()
        layout.addWidget(e,n,0,1,1)
        layout.addWidget(b,n,1,1,1)
        n = len(self.ids)
        if n != 0:
            n = self.ids[-1] + 1
        b.clicked.connect(partial(self.remove_row, item=n))
        self.ids.append(n)

    def remove_row(self, item):
        id = self.ids.index(item)
        idx = id * 2
        layout = self.layout()
        for i in range(2):
            item = layout.itemAt(idx)
            item.widget().hide()
            layout.removeItem(item)
        self.ids.pop(id)

    def get_classes(self):
        classes = []
        layout = self.layout()
        for i in range(len(self.ids)):
            txt = layout.itemAt(i*2).widget().text()
            classes.append(txt)
        return classes


class ClassifyLabelEditDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, classes=[], title="设置类别"):
        super().__init__(parent=parent)
        self.classes = classes
        self.classes_editor = EditWidget(parent=self)
        self.setWindowTitle(title)
        self.refreh_ui(classes)

    def refreh_ui(self, classes):
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(0)
        for c in classes:
            self.classes_editor.append_row(c)
        layout.addWidget(self.classes_editor)
        self.add_btn = QtWidgets.QPushButton("添加")
        layout.addWidget(self.add_btn)
        self.add_btn.clicked.connect(self.new_row)

        self.add_btn_box(layout)
        self.setLayout(layout)

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

    def access(self):
        self.classes = self.classes_editor.get_classes()
        self.done(0)
    def cancel(self):
        self.done(0)

    def new_row(self):
        self.classes_editor.append_row("")


class ClassifyWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, context=None):
        super(ClassifyWidget, self).__init__(parent=parent)
        self.context = context
        self.classes = []
        self.widgets = []
        self.setLayout(QtWidgets.QVBoxLayout())
        self.create_labels()
        self.layout().setSpacing(0)



    def clean_labels(self):
        for w in self.widgets:
            w.hide()
            self.layout().removeWidget(w)
            del w

    def create_labels(self):
        self.clean_labels()
        self.classes = []
        for i, cls in enumerate(self.context.state.classes):
            c = QtWidgets.QCheckBox(self)
            c.setText(cls)
            c.stateChanged.connect(partial(self.change_class, id=i))
            c.setWindowOpacity(0.5)
            self.widgets.append(c)
            self.layout().addWidget(c)
            self.classes.append(cls)
        self.setGeometry(0,0,120,25*len(self.classes))

    def show_labels(self):
        self.raise_()
        context = self.context
        state = context.state
        classes = state.get("classes", [])
        info = state.file_map[state.curr_item]
        labels = info.get("classes", [])
        for i, cls in enumerate(classes):
            if cls in labels:
                self.widgets[i].setChecked(True)
            else:
                self.widgets[i].setChecked(False)

    def change_class(self, curr_state, id):
        state = self.context.state
        cls = self.classes[id]
        info = state.file_map[state.curr_item]
        classes = info.get("classes", [])
        if curr_state == 2:     # checked
            if cls not in classes:
                classes.append(cls)
        elif curr_state == 0:   # unchecked
            if cls in classes:
                i = classes.index(cls)
                classes.pop(i)
        else:
            logging.warning(f"Unknow check state:{curr_state}")
        info["classes"] = classes


def Init(context):
    win = context.win
    widget = ClassifyWidget(parent=win.canvas, context=context)
    setattr(win, "classify_layer", widget)


def set_classes(context):
    state = context.state
    rect = context.win.rect()
    c = ClassifyLabelEditDialog(parent=None, classes=state.get("classes", []))
    cx = rect.x() + rect.width() // 2
    cy = rect.y() + rect.height() // 2
    x = max(cx-150, 0)
    y = max(cy-150, 0)
    c.setGeometry(QtCore.QRect(x,y,300,300))
    # c.setWindowModality(Qt.ApplicationModal)
    c.raise_()
    c.exec()
    state.classes = c.classes
    if not hasattr(context.win, "classify_layer"):
        Init(context)
    else:
        getattr(context.win, "classify_layer").create_labels()


def item_changed(context):
    win = context.win
    if hasattr(context.win, "classify_layer"):
        win.classify_layer.show_labels()




if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    ced = ClassifyLabelEditDialog(classes=["1","2","3","4","5"])
    ced.show()
    app.exec()