'''
This is an import-only object version of main.py.
For manipulating the app via python.
'''

'''
QT Imports
'''
from PyQt5.QtWidgets import *

'''
Data Manipulation Imports
'''
from data_utils import retry
from typing import Callable
import numpy as np
import json
import os.path
import shutil as sh
import sys
import os
import pandas as pd
from datetime import (datetime, timedelta)
from tqdm import tqdm
from bs4 import BeautifulSoup
from glob import glob
import time
from pathlib import Path

'''
Utils Imports
'''
from aws_utils import *
from data_utils import *
from google_utils import *
from graphics_utils import *
from upload_utils import *
import score_converters

from widgets.meta_window import MetaWindow
from widgets.alert import Alert

'''
HTML Tools
'''
import dominate
from dominate.tags import *
from dominate.util import raw


class Reportgen():
    def __init__(self,
                 resource_path: str,
                 debug: bool = True,
                 log: bool = True,
                 answers_only: bool = False,
                 answers_and_topics_only: bool = False,
                 ):
        self.log = log

        '''
        Debug toggle and debug bypass steps
        '''
        self.debug = debug  # Toggles all below to False when on.
        self.skip_generation = self.debug and True
        self.skip_merge = self.debug and (self.skip_generation or True)
        self.skip_notifications = self.debug and True
        self.skip_name_mapping_step = self.debug and True
        self.no_upload_aws = self.debug and True
        self.no_upload_gcp = self.debug and True
        self.no_upload_db = self.debug and True
        self.check_for_reruns = self.debug and False

        self.delete_non_merged = True

        '''
        Default User-Toggleable States
        '''
        self.answers_only = answers_only
        self.answers_and_topics_only = answers_and_topics_only

        self.nl = '\n'

        '''
        Key Init Steps
        '''
        home_path = os.environ['HOME']
        # Path to core data - local breakdown copy, etc...
        self.data_path = os.path.join(home_path, 'REDACTED')
        if not os.path.exists(self.data_path):
            print('This appears to be a fresh install - initializing cache!')
            os.mkdir(self.data_path)
        self.mem_path = os.path.join(self.data_path, 'memory.json')
        self.resource_path = resource_path  # 'src/main/resources/'
        self.install_resource_path = os.path.join(self.resource_path, 'install')

        '''
        Reinit Memory if Not Found
        '''
        if not os.path.exists(self.mem_path):
            print('Memfile not found; reinitializing from backup.')
            mem_install_path = os.path.join(self.install_resource_path, 'memory_init.json')
            sh.copy(mem_install_path, self.mem_path)

        '''
        Load Memory, and reinit memory files if not found.
        Then save.
        '''
        self.mem = load_json(self.mem_path)
        self.db_path = os.path.join(self.data_path, 'db')
        if not os.path.exists(self.db_path):
            os.mkdir(self.db_path)

        if not self.mem['paths_set']:
            print('Memory Key Paths not set; recreating them from defaults now.')
            for k in self.mem['db_info'].keys():
                mkdir_force(os.path.join(self.db_path, k))

            self.mem['db_info']['bkdn']['path'] = os.path.join(self.db_path, 'bkdn')
            self.mem['db_info']['protocols']['path'] = os.path.join(self.db_path, 'protocols')
            self.mem['paths_set'] = True

            self.save_mem()

        '''
        I/O path selection
        '''
        self.BREAKDOWN_PATH = self.mem['db_info']['bkdn']['path']
        self.BKDN_SHEET_ID = self.mem['db_info']['bkdn']['sheet_id']
        self.TEST_PROTOCOLS_PATH = self.mem['db_info']['protocols']['path']
        self.TEST_PROTOCOLS_SHEET_ID = self.mem['db_info']['protocols']['sheet_id']

        self.OUTPUT_PATH = None  # Will be set by widget during init.

        self.LAST_UPDATE = parse_datetime(self.mem['db_info']['last_updated'])
        if (not self.LAST_UPDATE) or (self.LAST_UPDATE < (datetime.now() - timedelta(days=1))):
            print('UPDATING TEST DATABASE')
            self.db_update(as_init=True)
            self.mem['db_info']['last_updated'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            self.save_mem()

        self.bkdn_dfs = pickle_load(self.BREAKDOWN_PATH)
        self.test_name_options = [k for k in self.bkdn_dfs.keys() if k[0] != '_']

        self.TEST_NAME = None
        self.NAME_ID_MAP = dict()
        self.SECTIONS = None
        self.sections_present = []

        self.bkdn = None
        self.test_order = None

        self.chosen_files = []

        self.raw_reports = dict()
        self.student = None

        self.test_meta_info = None

    def home(self):
        '''
        Include the below two lines to refresh options after db update:
        '''
        self.bkdn_dfs = pickle_load(self.BREAKDOWN_PATH)
        self.test_name_options = [k for k in self.bkdn_dfs.keys() if k[0] != '_']

        print_conditional(f'Output path: {self.OUTPUT_PATH}', self.log)

    def set_params(self, test_name: str,
                   test_meta_info: dict,
                   chosen_files: list):
        self.TEST_NAME = test_name
        assert test_meta_info.get('test_date') != None
        self.test_meta_info = test_meta_info
        print_conditional(self.test_meta_info, self.log)
        self.chosen_files = chosen_files

        assert self.params_set()

    def db_update(self, as_init=False):
        persisted_creds_path = os.path.join(self.resource_path, 'REDACTED')
        CREDS_PATH = persisted_creds_path
        TOKEN_PATH = os.path.join(self.resource_path, 'REDACTED')
        service = sheets_auth(TOKEN_PATH, CREDS_PATH)
        for key, value in self.mem['db_info'].items():
            if key != 'last_updated':
                print_conditional(f'Updating: {key.upper()}', self.log)
                self.sheet_update(service, value)
        if not as_init:
            # re_init home
            self.home()

    def sheet_update(self, service, sheet_info):
        meta = service.spreadsheets().get(spreadsheetId=sheet_info['sheet_id']).execute()
        sheets = [s['properties']['title'] for s in meta['sheets']]
        params = {'spreadsheetId': sheet_info['sheet_id'], 'ranges': sheets, 'majorDimension': 'ROWS'}
        result = service.spreadsheets().values().batchGet(**params).execute()
        all_sheets = result['valueRanges']
        sheet_dfs = {name: self.sheet_to_df(sheet) for name, sheet in tqdm(zip(sheets, all_sheets))}
        pickle_save(sheet_info['path'], sheet_dfs)

    def sheet_to_df(self, sheet):
        # Call the Sheets API
        values = sheet.get('values', [])
        df = pd.DataFrame.from_records(values).fillna('N/A')
        h = df.iloc[0]
        df = df.iloc[1:]
        df.columns = h
        return df

    def save_mem(self):
        # Save and reload.
        write_json(self.mem, self.mem_path)
        self.mem = load_json(self.mem_path)
        print_conditional(self.mem['db_info']['last_updated'], self.log)

    def save_selected_input_files_to_memory(self, files):
        # Save selected files location so future selection opens the same dir
        default_dir = str(Path(files[0]).parent)
        self.mem['input_select_default_dir'] = default_dir
        self.save_mem()
        self.chosen_files = files

    def change_output_folder(self, new_outpath):
        self.mem['output_select_default_dir'] = str(Path(new_outpath).parent)
        self.mem['io']['outpath'] = new_outpath
        self.save_mem()
        self.OUTPUT_PATH = new_outpath

    def reset_to_do_more(self):
        self.chosen_files = []
        self.TEST_NAME = None

    def params_set(self):
        must_be_non_null = [
            self.TEST_NAME,
            self.test_meta_info

        ]
        must_be_true = [
            len(self.chosen_files) > 0,
        ]

        print_conditional([must_be_non_null, must_be_true], self.log)
        return (None not in must_be_non_null) and (False not in must_be_true)

    def prepare_for_parse(self):
        self.bkdn = self.bkdn_dfs[self.TEST_NAME]
        # Remove extraneous whitespace from data.
        for c in self.bkdn.columns:
            self.bkdn[c] = self.bkdn[c].apply(lambda x: str(x).strip())
        # Remove all bash-reserved chars from bkdn:
        self.bkdn = self.bkdn.applymap(remove_problematic_chars)

        # Alter column names for ease of indexing:
        self.bkdn.columns = [strp_nonanum(c).lower() for c in self.bkdn.columns]
        self.bkdn = self.bkdn.replace('', np.nan)

        '''
        Compress Subtopics into One Column:
        '''
        subtopic_cols = self.bkdn[[col for col in self.bkdn.columns if 'subtopic' in col.lower()]]
        if self.debug:
            pickle_save(ensure_path(os.path.join(self.data_path, f'debug/{self.TEST_NAME}/subtopic_cols.pickle')),
                        subtopic_cols)

        self.bkdn = self.bkdn.drop(subtopic_cols, axis=1)

        if self.debug:
            pickle_save(
                ensure_path(os.path.join(self.data_path, f'debug/{self.TEST_NAME}/debug_bkdn_prejoin.pickle')),
                self.bkdn)

        self.test_order = self.bkdn.section.unique()

        # Contatenate subtopics with +, while dropping NANs.
        self.bkdn['subtopic'] = subtopic_cols.apply(lambda x: ' + '.join([str(v) for v in x.dropna().values]),
                                                    axis=1)
        self.bkdn.index = pd.Index([str(i) for i in self.bkdn.index])

        # Update Sections, Subsections
        self.SECTIONS = self.bkdn.section.unique()

    def parse_reports(self):
        '''
        :return: a DataFrame containing the score data from a given test.
        '''
        assert self.params_set(), 'You are attempting to parse reports out of sequence. Please run self.set_params.'

        self.chosen_files = sorted(self.chosen_files)
        report_paths = sorted([os.path.relpath(p) for p in self.chosen_files]).copy()
        debug_report_paths = sorted([os.path.abspath(p) for p in self.chosen_files]).copy()

        # This function fills missing student data with NANs between reports.
        # Returns any files that are completely empty.
        empty_files = fix_missing_students(report_paths.copy())
        empty_fnames = [Path(p).name for p in empty_files]
        print_conditional(['Ignoring: ', empty_fnames], self.log)

        report = pd.DataFrame()

        if self.debug:
            pickle_save(os.path.join(self.data_path, f'debug/{self.TEST_NAME}/bkdn_parse_step.pickle'), self.bkdn)
            pickle_save(os.path.join(self.data_path, f'debug/{self.TEST_NAME}/report_paths_parse_step.pickle'),
                        debug_report_paths)

        # Iterate through reports (separated by section)

        files_to_upload = []
        for path in report_paths:
            if Path(path).name not in empty_fnames:
                files_to_upload.append(os.path.abspath(path))

                print_conditional(self.SECTIONS, self.log)
                # try:
                print_conditional(self.bkdn.columns, self.log)
                path_fname = path.split('/')[-1].split('.')[0]
                print_conditional(path_fname, self.log)
                filtered = self.bkdn[self.bkdn.correspondingfilename.str.upper() == path_fname.upper()]

                try:
                    section = filtered.section.unique()[0]
                except:
                    Alert(
                        f"Couldn't find Corresponding_File_Names {self.bkdn.correspondingfilename.unique()} for test {self.TEST_NAME} in input files; are you using the right reports?",
                        title='Wrong Test Name (Likely)')
                    assert False
                subsection = filtered.subsection
                a_df = pd.read_csv(path)
                # Reformat columns for easy indexing:
                a_df.columns = ['_'.join(c.split()) for c in a_df.columns]

                self.raw_reports[section] = a_df

                # Copy df for safety.
                a_new = a_df.copy()

                key = a_new[a_new.First_Name == 'Answer Key'].iloc[:, 9:].T
                key.columns = ['key']

                students_data = a_new[a_new.First_Name != 'Answer Key']
                names = students_data['First_Name'] + ' ' + students_data['Last_Name']

                # Map each student to their student ID (gradecam)
                section_name_id_map = pd.concat([students_data.Student_ID, names], axis=1)
                section_name_id_map.columns = ['id', 'name']
                section_name_id_map = section_name_id_map.set_index('name').to_dict()['id']
                self.NAME_ID_MAP.update(section_name_id_map)

                responses = students_data.iloc[:, 9:]
                # Grab and format student names
                students = responses.set_index(names).T

                if self.debug:
                    pickle_save(
                        os.path.join(self.data_path, f'debug/{self.TEST_NAME}/students_isolated_from_report.pickle'),
                        students)

                # Fillna with "N/A" so that empty answers are not dropped from the df.
                data = pd.concat([key, students], axis=1).fillna('N/A')

                data['section'] = section
                data['subsection'] = subsection
                self.sections_present.append(section)

                # Remove duplicated student name columns - students with duplicate entries.
                data = (data.T.loc[~data.T.index.duplicated(keep='first')]).T

                report = pd.concat([report, data], axis=0, sort=True)

            else:
                print_conditional(f'Skipping: {path}', self.log)

        if self.debug:
            pickle_save(os.path.join(self.data_path, f'debug/{self.TEST_NAME}/parsed_report.pickle'), report)
            pickle_save(os.path.join(self.data_path, f'debug/{self.TEST_NAME}/name_id_map.pickle'), self.NAME_ID_MAP)

        return report

    def generate(self, data):
        student_in_section = (data.groupby('section').count() > 0)
        print_conditional('Students in each section:', self.log)
        print_conditional(student_in_section, self.log)

        # Further parse breakdown now that parse_reports has detected which subsections are present in the data.
        self.bkdn = self.bkdn[self.bkdn.section.apply(lambda x: x in self.sections_present)]

        # Reindex to allow for subsection alignment:
        self.bkdn.index = self.bkdn.section + '-' + self.bkdn.ques
        data.index = data.section + '-' + data.index

        students = [c for c in data.columns.dropna() if ('key' != c.lower()) \
                    and ('section' != c.lower()) \
                    and ('subsection' != c.lower())]

        if self.debug:
            pickle_save(ensure_path(os.path.join(self.data_path, f'debug/{self.TEST_NAME}/debug_bkdn.pickle')),
                        self.bkdn)
            pickle_save(
                ensure_path(os.path.join(self.data_path, f'debug/{self.TEST_NAME}/{self.TEST_NAME}_data.pickle')),
                data)
            pickle_save(ensure_path(os.path.join(self.data_path, f'debug/{self.TEST_NAME}/students_debug.pickle')),
                        students)

        test_output_path = os.path.join(self.OUTPUT_PATH,
                                        f'{self.TEST_NAME} - {self.test_meta_info["test_date"]}')

        if not os.path.exists(test_output_path):
            os.mkdir(test_output_path)

        all_report_contexts = []
        for student in tqdm(students):
            student_id = self.NAME_ID_MAP[student]

            already_run = False
            already_run_message = ''
            if self.check_for_reruns:
                already_run_message, already_run = is_already_run(student_id, self.TEST_NAME)
            if not already_run:
                student_output_path = os.path.join(test_output_path, student)
                if os.path.exists(student_output_path):
                    sh.rmtree(student_output_path)
                os.mkdir(student_output_path)

                report_context = {'sections': [],
                                  'outdir': student_output_path,
                                  'test_outdir': test_output_path,
                                  'student_outdir': student_output_path,
                                  'student': student,
                                  'test_name': self.TEST_NAME,
                                  'bkdn_dfs': self.bkdn_dfs}

                if student != 'key':

                    student_answers = data[['key', student]].copy()
                    student_answers = check_answers(student_answers, student)

                    report_context['answers'] = student_answers
                    report_context['bkdn'] = self.bkdn

                    student_ans_w_bkdn = pd.concat([student_answers, self.bkdn], axis=1, sort=False)  # .dropna()
                    if self.debug:
                        pickle_save(ensure_path(
                            os.path.join(self.data_path, f'debug/{self.TEST_NAME}/{student}_ansWbkdn.pickle')),
                            student_ans_w_bkdn)

                    '''
                    HERE: Iterate through sections!
    
                    data filestructure:
                    data-|
                         |-TEST_NAME -|-student_1 -|-section1-(imgs)
                                      |            |-section2
                                      |            |-output_html
                                      |            |-output_pdf
                                      |-student_2...
    
                    '''
                    for section in self.test_order:
                        # Only process the data if the section is present, and the student is in the section:
                        if (section in self.sections_present) and (student_in_section.get(student, {}).get(section)):
                            report_context['sections'].append(section)
                            report_context[section] = dict()

                            section_imgs_path = os.path.join(student_output_path, section)
                            if not os.path.exists(section_imgs_path):
                                os.mkdir(section_imgs_path)

                            section_df = student_ans_w_bkdn[student_ans_w_bkdn.section == section].copy()
                            # fill in any nans:
                            null_values = [np.nan, '']
                            section_df = section_df.applymap(lambda x: 'NA' if x in null_values else x).fillna('NA')

                            if self.debug:
                                pickle_save(ensure_path(os.path.join(self.data_path,
                                                                     f'debug/{self.TEST_NAME}/{student}_{section}_df.pickle')),
                                            section_df)

                            report_context[section]['df'] = section_df

                            '''
                            Calc Section Data for Cover Sheet:
                            '''
                            raw_data = self.raw_reports[section]
                            print_conditional(section, self.log)
                            # If there are any NA values in the 'correct' column, cast back to boolean FALSE.
                            report_context[section]['n_correct'] = section_df.correct.apply(
                                lambda x: False if x == 'NA' else x).sum()
                            report_context[section]['n_total'] = section_df.shape[0]
                            report_context['teacher_name'] = \
                                (raw_data.Teacher_First_Name + ' ' + raw_data.Teacher_Last_Name).dropna().iloc[0]

                            paths_dict = dict()
                            if (not self.answers_only) and (not self.skip_generation):
                                # try
                                topics_plot_path = section_topics_barplot_gen(section_df, student, section,
                                                                              section_imgs_path)

                                subtopic_imgs_path = os.path.join(section_imgs_path, 'subsections')
                                if not os.path.exists(subtopic_imgs_path):
                                    os.mkdir(subtopic_imgs_path)

                                subtopics_plot_paths = section_subtopics_barplot_gen(section_df, student, section,
                                                                                     subtopic_imgs_path)

                                paths_dict = {'topic_fig': topics_plot_path,
                                              'subtopic_figs': subtopics_plot_paths}

                            if not self.skip_generation:
                                self.write_report(student, self.TEST_NAME, section, \
                                                  section_df, paths_dict, test_output_path)

                    # Outside of sections loop - only do once per student!
                    if self.debug:
                        pickle_save(
                            ensure_path(f"{self.data_path}/debug/{self.TEST_NAME}/{student}_coversheet_data.pickle"),
                            report_context)

                    report_context['scores'] = score_converters.convert(report_context)

                    if not self.skip_generation:
                        self.write_coversheet(report_context)

                    all_report_contexts.append(report_context)
            else:
                print(already_run_message)
        return test_output_path, all_report_contexts

    def gen_report_data(self, return_data=False, reset_after=False):
        '''
        Ensure no None values where necessary.
        '''

        assert self.params_set(), 'You are attempting to generate reports out of sequence. Please run self.set_params.'

        self.prepare_for_parse()

        data = self.parse_reports()

        test_output_path, all_report_contexts = self.generate(data)
        if len(all_report_contexts) > 0:
            self.merge_and_upload_reports(test_output_path, all_report_contexts)
        else:
            print('No reports to merge! Stopping run.')

        if reset_after:
            self.reset_to_do_more()

        if return_data:
            return {
                'output_dest': test_output_path,
                'contexts': all_report_contexts
            }

    def write_report(self, student, TEST_NAME, section, section_df, img_paths_dict, test_output_path):
        fname = f'{student}_{TEST_NAME}_{section}_report'.upper() + '.html'

        report_path = os.path.join(test_output_path, student)

        '''
        COPY resource_dir to outpath dir.
        '''

        path_to_styles = os.path.join(self.resource_path, 'base/styles')
        style_dest = os.path.join(report_path, 'styles')

        '''
        Move images, CSS and JS styles into student output dir if it doesn't already exist.
        '''
        if not os.path.exists(style_dest):
            print_conditional('Creating styles dir in:', self.log)
            print_conditional(style_dest, self.log)
            sh.copytree(path_to_styles, os.path.join(report_path, 'styles'))
            sh.copytree(os.path.join(self.resource_path, 'base/imgs'), os.path.join(report_path, 'imgs'))

        outpath = os.path.join(report_path, fname)
        doc = dominate.document(title=fname)

        '''
        Table Report Row Gen
        '''
        v = section_df.iloc[:, 1:4]
        v.columns = ['A', 'correct', 'Q']

        table_styles, table_html = make_report_table(section_df)

        '''
        Make the Doc
        '''
        with doc.head:
            link(rel='stylesheet', href='styles/css/bootstrap.css')
            link(rel='stylesheet', href="styles/css/report_styles.css")
            raw(table_styles)

        with doc.body:
            '''
            REPORT PAGE (Answers, comments)
            '''
            div(h1(f'{student.title()} - Section: {section.upper()}'), _class='jumbotron text-center')
            p(f'{student}{section}{TEST_NAME}', _class='d-none', id='Info')
            with div(_class='container'):
                '''
                Row 1: Answers Table row:
                '''
                with div(_class='row'):
                    '''
                    raw(t.to_html(index=False, \
                                  classes=["table-bordered", "table-striped", "table-hover"], \
                                  justify='center'))
                    '''
                    raw(table_html)
                with div(_class='row'):
                    # u(h3('Evaluation', id='eval'))
                    pass

            if not self.answers_only:
                '''
                Charts Page
                Plot Main Topic Fig:
                '''
                # Break page
                raw('<p style="page-break-before: always"></p>')

                with div(_class='container'):
                    '''
                    Row 2: Img reports
                    '''
                    with div(_class='row'):
                        topic_img = img_paths_dict['topic_fig']
                        div(img(src=topic_img, _class='img-fluid'), _class='col-12 text-center')

                '''
                Plot Subtopic Figs
                '''
                if not self.answers_and_topics_only:
                    subtopic_imgs = img_paths_dict['subtopic_figs']
                    n_per_page = 4
                    n_per_row = 2
                    subtopic_img_lists = chunk_list(subtopic_imgs, n_per_page)
                    for list_idx, img_list in enumerate(subtopic_img_lists):
                        with div(_class='container'):
                            img_grid = div(_class='container')
                            curr_row = div(_class='row')
                            for idx, ipath in enumerate(img_list):
                                if (idx % n_per_row == 0) and (idx > 1):
                                    # Append finished row:
                                    img_grid.add(curr_row)
                                    # New row:
                                    curr_row = div(_class='row')
                                img_div = div(_class='col-6')
                                img_div.add(img(src=ipath, _class='img-fluid'))
                                curr_row.add(img_div)

                        if list_idx < len(subtopic_img_lists) - 1:
                            # Break page after each grid of images.
                            raw('<p style="page-break-before: always"></p>')

            '''
            Post-Construction JS
            '''
            # For Table:
            script(src='styles/js/table_formatter.js')

        write_html(doc, outpath)

    def write_coversheet(self, report_context):

        student = report_context['student']
        if self.debug:
            pickle_save(f"{self.data_path}/debug/{self.TEST_NAME}/{student}_context.pickle", report_context)

        test_name = report_context['test_name']
        fname = f'{student}_{test_name}_COVERSHEET'.upper() + '.html'

        report_path = os.path.join(report_context['test_outdir'], student)

        outpath = os.path.join(report_path, fname)

        doc = dominate.document(title=fname)

        with doc.head:
            link(rel='stylesheet', href='styles/css/bootstrap.css')
            link(rel='stylesheet', href="styles/css/report_styles.css")
            raw('''
            <style>
            .container {
                height: 100%;
            }
            </style>
            ''')

        with doc.body:
            with div(_class='container'):
                div(_class='row', style='height: 2em')
                # Header Image:
                with div(_class='row', style='height: 15em'):
                    with div(_class='col-12 text-center'):
                        img(src='REDACTED',
                            _class='img-fluid', style='height: 15em')
                with div(_class='row', style="text-align: center"):
                    div('REDACTED', _class='col',
                        style="display: inline-block; width: 100%")

                with div(_class='row'):
                    with div(_class='col align-self-center'):
                        h3(f'{self.TEST_NAME} Score Sheet')
                        p(strong('NAME:'), f' {student}', style='font-size: 2em')
                        # p(f'Test Edition: {}')
                        p(strong('CLASS/TUTOR:'),
                          f' {report_context.get("teacher_name")}',
                          style='font-size: 2em')

                        with table(_class='table'):
                            thead(tr([th('')] + [th(strong(section), scope='col') for section in
                                                 report_context['sections']]))
                            with tr(scope='row', style={'text-align': 'center'}):
                                td(strong('Correct:'))
                                for section in report_context['sections']:
                                    section_data = report_context[section]
                                    td(f'{section_data["n_correct"]}/{section_data["n_total"]}')

                with div(_class='row jumbotron'):
                    scores = report_context['scores']
                    if self.debug:
                        pickle_save(
                            os.path.join(self.data_path, f'debug/{self.TEST_NAME}/{student}_score_conversions.pickle'),
                            scores)
                    with table(_class='table'):
                        thead(tr([th('')] + [strong(th(section.upper(), scope='col')) for section, _ in scores]))
                        with tr(scope='row', style={'text-align': 'center'}):
                            td(strong('Scaled:'))
                            for _, score in scores:
                                td(f'{score}')

        write_html(doc, outpath)

    def merge_and_upload_reports(self, test_output_path, all_report_contexts):
        '''
        Make a post to the db app!        
        '''
        if not self.no_upload_db:
            print_conditional('Uploading to DB app:', self.log)
            for report_context in all_report_contexts:
                upload_to_db(report_context, self.NAME_ID_MAP, self.test_meta_info)
        else:
            print_conditional('Skipping REDACTED Mongo upload (DEBUG)', self.log)

        test_dir = test_output_path

        student_names = [n for n in os.listdir(test_dir) if not n.startswith('.')]

        if not self.skip_merge:
            for student in student_names:
                print_conditional(f'Converting: {student.upper()}', self.log)
                test_name = self.TEST_NAME
                merged_pdf_fname = f'{student}_{test_name}_MERGED'.upper() + '.pdf'
                outdir = os.path.join(test_output_path, student)

                '''
                Get list of all files to merge:
                '''
                html_files = []
                for path, dirs, files in os.walk(outdir):
                    for f in files:
                        if '.html' in f:
                            fpath = os.path.join(path, f)
                            if 'COVERSHEET' in f:
                                html_files.insert(0, fpath)
                            else:
                                html_files.append(os.path.join(path, f))

                print_conditional('Converting Reports to PDF:', self.log)
                watermark_path = os.path.join(self.resource_path, 'base/imgs/watermark.pdf')
                pdf_paths = [html_to_pdf(f, watermark_path) for f in tqdm(html_files)]

                '''
                Sort PDF paths by breakdown section order (self.test_order):
                '''
                # get coversheet
                cover = [pdf_paths[0]]
                # get non-coversheet
                non_cover = pdf_paths[1:]
                # sort (in place) by matching sections within pdf filenames
                non_cover.sort(
                    key=lambda x:
                    [idx for idx, section in enumerate(self.test_order) if f'_{section.upper()}_REPORT' in x][0])
                # list concatenate, and overwrite the old pdf_paths order:
                pdf_paths = cover + non_cover

                print_conditional('Merging:', self.log)
                merged_pdf_dest = os.path.join(outdir, merged_pdf_fname)
                merge_pdfs(pdf_paths, out_fname=merged_pdf_dest)

                '''
                Upload the merged pdf to AWS bucket:
                '''
                student_id = None
                if not self.no_upload_aws:
                    student_id = self.NAME_ID_MAP[student]
                    s3_dest = f'REDACTED'
                    retry(upload_to_s3, 5, merged_pdf_dest, s3_dest)
                if not self.no_upload_gcp:
                    gcp_dest = f'REDACTED'
                    retry(upload_blob, 5, self.resource_path,
                          'REDACTED', merged_pdf_dest, gcp_dest)

                '''
                NOTIFY the student:
                '''
                # Will notify if not told to skip.
                # Will also NOT notify a student if their student_id can't be located.
                if (not self.skip_notifications):
                    if not student_id:
                        Alert(
                            f"Can't find student id {student_id} - either you're in debug mode and using a fake student, or there's something wrong with our student id database.",
                            title='Student ID Missing')
                    else:
                        base_url = 'REDACTED'
                        api = 'REDACTED'
                        url = os.path.join(base_url, api)
                        request_body = {'gradecam_id': student_id,
                                        'report_type': self.TEST_NAME}
                        print_conditional('Notifying:', self.log)
                        print_conditional(request_body, self.log)
                        resp = retry(requests.post, 5, url, data=request_body)
                        if resp.status_code != 200:
                            print_conditional(f'Error notifying {student_id} at {url}', self.log)
                else:
                    print_conditional('Skipping Notification step!', self.log)

                '''
                Delete all but the merged output
                '''
                if self.delete_non_merged:
                    out_contents = os.listdir(outdir)
                    for f in out_contents:
                        path = os.path.join(outdir, f)
                        if os.path.isfile(path):
                            if '_MERGED.' not in path:
                                os.remove(path)

                print_conditional(f'Report published to {merged_pdf_dest}!', self.log)
        else:
            print_conditional('Skipping report merge process (DEBUG).', self.log)

        '''
        Update S3 name-id map with local mappings.
        '''
        if not self.skip_name_mapping_step:
            print_conditional('Updating name-id records on S3...', self.log)
            retry(download_from_s3, 5, 'REDACTED', 'REDACTED')
            with open('REDACTED', 'r') as jf:
                remote_name_id_map = json.loads(jf.read())

            # Remove old version
            os.remove('REDACTED')
            # Update in memory
            remote_name_id_map.update(self.NAME_ID_MAP)

            # Write out as new (updated) version
            with open('REDACTED', 'w') as jf_out:
                jf_out.write(json.dumps(remote_name_id_map))

            # Finally, upload updated version
            # TO S3
            retry(upload_to_s3, 5, 'REDACTED', 'REDACTED')
            # TO GCP
            if not self.no_upload_gcp:
                retry(upload_blob, 5, self.resource_path,
                      'REDACTED', 'REDACTED', 'REDACTED')
            os.remove('REDACTED')
        else:
            print_conditional('Skipping name map step (DEBUG).', self.log)

        '''
        When all is finished:
        '''
        if not self.mem.get('aws_configured'):
            print_conditional('AWS not configured on this computer', self.log)
            if 'mgb' in os.environ['HOME']:
                print_conditional('Configuring as REDACTED', self.log)
                retry(aws_login, 5, os.path.abspath(self.resource_path), as_root=True)
            else:
                print_conditional('Configuring as REDACTED', self.log)
                retry(aws_login, 5, os.path.abspath(self.resource_path))
            self.mem['aws_configured'] = True
            self.save_mem()
        if not self.no_upload_aws:
            print_conditional('Uploading Files to AWS - Do Not Close Any Windows.', self.log)
            retry(update_test_archives_to_aws, 5, self.chosen_files, self.TEST_NAME)
        else:
            print_conditional('Skipping AWS archive upload (DEBUG).', self.log)
