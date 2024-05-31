from PySide6 import QtWidgets
def list_commbo(commbo:QtWidgets.QComboBox):
    values = []
    for i in range(commbo.count()):
        values.append(commbo.itemText(i))
    return values