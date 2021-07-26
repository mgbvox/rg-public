from PyQt5.QtWidgets import *


class Alert(QDialog):
    def __init__(self,
                 message='PLACEHOLDER MESSAGE!',
                 title='Alert!',
                 parent=None):
        super().__init__(parent=parent)

        self.setWindowTitle(title)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
        self.buttonBox.accepted.connect(self.close)

        self.layout = QVBoxLayout()
        message = QLabel(message)
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

        self.exec_()
