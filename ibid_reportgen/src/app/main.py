import datetime

'''
QT Imports
'''
from PyQt5.QtWidgets import *
from base import appctxt

'''
Data Manipulation Imports
'''
import os.path
import sys
import dominate
from dominate.tags import p, h1, h2

'''
Utils Imports
'''
from data_utils import *
from google_utils import *
from graphics_utils import *

from widgets.meta_window import MetaWindow
from widgets.alert import Alert

from reportgen import Reportgen

'''
Class Definitions
'''

# noinspection PyAttributeOutsideInit
class Window(QWidget):
    '''
    The graphical interface for manipulating the state of reportgen.py
    '''

    def __init__(self):
        QWidget.__init__(self)

        home = os.path.expanduser('~')
        resource_path = os.path.join(home, 'REDACTED')

        # Delegate all non-GUI code to a reportgen object!
        self.rg = Reportgen(
            resource_path=resource_path,
            debug=False,
            log=True,
            answers_only=False,
            answers_and_topics_only=False,
        )

        print(self.rg.resource_path)

        layout = QGridLayout()
        self.setLayout(layout)

        self.default_selector_message = 'Please Select Test Name...'
        self.setStyleSheet("QTextBrowser {font: 12pt Monotype Corsiva}")

        if self.rg.debug:
            Alert(
                message=f'''DEBUG MESSAGE REDACTED

SUBMODE STATUS:
skip_generation = {self.rg.skip_generation}
skip_merge = {self.rg.skip_merge}
delete_non_merged = {self.rg.delete_non_merged}
skip_notifications = {self.rg.skip_notifications}
skip_name_mapping_step = {self.rg.skip_name_mapping_step}
no_upload_aws = {self.rg.no_upload_aws}
no_upload_gcp = {self.rg.no_upload_gcp}
no_upload_db = {self.rg.no_upload_db}''',
                title='DEBUG WARNING')

        self.home()

    def home(self):
        # Buttons
        self.lbl = QLabel(f'REDACTED', self)
        self.layout().addWidget(self.lbl, 0, 0)

        self.log_window = QTextBrowser(self)
        self.log_window.setText(
            'Select your input files below, then hit "Generate Reports."')
        self.layout().addWidget(self.log_window, 1, 0, 1, 2)

        '''
        Get (and save) output path!
        '''
        if not self.rg.mem['io']['outpath']:
            self.rg.OUTPUT_PATH = self.select_dir()
            self.rg.mem['io']['outpath'] = self.rg.OUTPUT_PATH
            self.rg.save_mem()
        else:
            self.rg.OUTPUT_PATH = self.rg.mem['io']['outpath']
        print(self.rg.OUTPUT_PATH)
        self.stats_window = QTextBrowser(self)
        self.stats_window.setText(self.rg.OUTPUT_PATH)
        self.layout().addWidget(self.stats_window, 1, 2, 1, 2)

        self.STATS_MEM = {}
        self.update_stats({'output_path': self.rg.OUTPUT_PATH,
                           'test_name': self.rg.TEST_NAME})

        self.input_file_selector = QPushButton(self)
        self.input_file_selector.setText('Select Input Files')
        self.input_file_selector.clicked.connect(self.select_input_files)
        self.layout().addWidget(self.input_file_selector, 2, 0, 1, 2)

        self.test_name_selector = QComboBox()

        # Include the rest of the button options as per usual:
        self.test_name_selector.addItem(self.default_selector_message)
        self.test_name_selector.addItems(self.rg.test_name_options)
        self.test_name_selector.activated.connect(self.set_test_name_and_meta_info)
        self.layout().addWidget(self.test_name_selector, 3, 0, 1, 4)

        self.input_3 = QPushButton(self)
        self.input_3.setText('Generate Reports')
        self.input_3.clicked.connect(self.do_generate)
        self.layout().addWidget(self.input_3, 4, 0, 1, 2)

        self.do_more = QPushButton(self)
        self.do_more.setText('Do More')
        self.do_more.clicked.connect(self.reset_to_do_more)
        self.layout().addWidget(self.do_more, 5, 0, 1, 4)

        self.answers_only_toggle = QComboBox()
        self.answers_only_toggle.addItem('Answers Only: NO')
        self.answers_only_toggle.addItem('Answers Only: YES')
        self.answers_only_toggle.addItem('Answers Only: YES, But Plot Topics')
        self.answers_only_toggle.activated.connect(self.handle_answers_only_toggle)
        self.layout().addWidget(self.answers_only_toggle, 6, 2, 1, 2)

        self.change_output = QPushButton(self)
        self.change_output.setText("Select Output Folder")
        self.change_output.clicked.connect(self.change_output_folder)
        self.layout().addWidget(self.change_output, 2, 2, 1, 2)

        self.notify_students_checkbox = QCheckBox("Notify Students")
        self.notify_students_checkbox.setChecked(not self.rg.skip_notifications)
        self.notify_students_checkbox.stateChanged.connect(self.toggle_notifications)
        self.layout().addWidget(self.notify_students_checkbox, 6, 1, 1, 2)

        self.refresh_breakdowns = QPushButton(self)
        self.refresh_breakdowns.setText("Refresh Breakdowns")
        self.refresh_breakdowns.clicked.connect(self.rg.db_update)
        self.layout().addWidget(self.refresh_breakdowns, 4, 2, 1, 2)

    def toggle_notifications(self):
        self.rg.skip_notifications = not self.rg.skip_notifications
        print(f'Skip Notifications: {self.rg.skip_notifications}')

    def set_test_name_and_meta_info(self):
        # meta = MetaWindow()
        # For now, set test date as reportgen execution date
        now = datetime.datetime.now().date()
        meta = {
            'test_date': f'{now.month}-{now.day}-{now.year}'
        }
        self.rg.test_meta_info = meta

        selected_value = self.test_name_selector.currentText()
        if selected_value != self.default_selector_message:
            self.rg.TEST_NAME = selected_value
        self.update_stats({'test_name': self.rg.TEST_NAME,
                           'report_date': self.rg.test_meta_info['test_date']})

    def handle_answers_only_toggle(self):
        selected_value = self.answers_only_toggle.currentText()
        self.rg.answers_only = (selected_value == 'Answers Only: YES')
        self.rg.answers_and_topics_only = (selected_value == 'Answers Only: YES, But Plot Topics')
        print(f'Chosen Mode: {selected_value}')

    def update_stats(self, params):
        self.STATS_MEM.update(params)
        doc = dominate.document()
        with doc.body:
            for k, v in self.STATS_MEM.items():
                title = k.replace('_', ' ').title()
                h2(title + ':')
                p(v if v else ' ')
        self.stats_window.setText(str(doc))

    def select_files(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        default_dir = self.rg.mem.get('input_select_default_dir', os.path.expanduser('~'))
        files, _ = QFileDialog.getOpenFileNames(self,
                                                "QFileDialog.getOpenFileNames()",
                                                default_dir,
                                                "All Files (*);;Python Files (*.py)",
                                                options=options)
        if files:
            return files
        else:
            return []

    def select_input_files(self):
        files = self.select_files()
        if files:
            self.rg.save_selected_input_files_to_memory(files)
        chosen_files_text = f"Chosen Files:{self.rg.nl}*{f'{self.rg.nl}*'.join(self.rg.chosen_files)}"
        self.log_window.setText(chosen_files_text)

    def select_dir(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        default_dir = self.rg.mem.get('output_select_default_dir', os.path.expanduser('~'))
        return QFileDialog.getExistingDirectory(self, "Select new Output Folder.", directory=default_dir,
                                                options=options)

    def change_output_folder(self):
        new_outpath = self.select_dir()
        if new_outpath:
            self.rg.change_output_folder(new_outpath)
        QMessageBox.about(self, 'Output Folder Changed', f'Output folder changed to {self.rg.OUTPUT_PATH}')
        self.update_stats({'output_path': self.rg.OUTPUT_PATH})

    def reset_to_do_more(self):
        self.rg.chosen_files = []
        self.rg.TEST_NAME = None
        self.home()

    def do_generate(self):
        if self.rg.TEST_NAME:
            self.rg.gen_report_data(return_data=False, reset_after=True)
        else:
            QMessageBox.about(self, 'Error', f'Please select a valid test name from the dropdown menu.')


if __name__ == '__main__':
    GUI = Window()
    GUI.show()
    exit_code = appctxt.app.exec_()  # 2. Invoke appctxt.app.exec_()
    sys.exit(exit_code)
