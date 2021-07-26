from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys

from .alert import Alert

from datetime import datetime


class MetaWindow(QDialog):
    def __init__(self):
        super(MetaWindow, self).__init__()
        # setting window title
        self.setWindowTitle("Scan Meta Info")

        # setting geometry to the window
        self.setGeometry(100, 100, 300, 400)

        # creating a group box
        self.formGroupBox = QGroupBox("Test Metadata")

        # creating combo box to select degree
        self.className = QLineEdit()

        self.teacherFirstName = QLineEdit()
        self.teacherLastName = QLineEdit()

        # creating spin box to select age
        self.testDate = QDateEdit()
        self.testDate.setDate(datetime.now().date())

        # calling the method that create the form
        self.createForm()

        # creating a dialog button for ok and cancel
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        # adding action when form is accepted
        self.buttonBox.accepted.connect(self.validate)

        # creating a vertical layout
        mainLayout = QVBoxLayout()

        # adding form group box to the layout
        mainLayout.addWidget(self.formGroupBox)

        # adding button box to the layout
        mainLayout.addWidget(self.buttonBox)

        # setting lay out
        self.setLayout(mainLayout)

        self.exec_()

    def validate(self):
        data = self.to_json()
        is_valid = False not in [bool(v) for v in data.values()]
        if is_valid:
            self.close()
        else:
            Alert(message='All fields are required!', title='Missing Info')

    def createForm(self):
        # creating a form layout
        layout = QFormLayout()

        layout.addRow(QLabel("Test Date"), self.testDate)

        # setting layout
        self.formGroupBox.setLayout(layout)

    def to_json(self):
        return {
            'testDate': str(self.testDate.date().toPyDate())
        }
