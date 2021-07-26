from bs4 import BeautifulSoup
from dateparser import parse
from difflib import SequenceMatcher
import json
import numpy as np
import os
import pandas as pd
import PyPDF2
from PyPDF2 import PdfFileMerger
import re
import requests
import shutil as sh
import shutil

def print_conditional(out, condition):
    if condition:
        print(out)

def remove_problematic_chars(x, bad = r'\/'):
    return re.sub(bad,'_',x)

def strp_nonanum(s):
    return re.sub(r'[^A-Za-z0-9]', '', s)

def write_json(dict_, outfile='json_out.json'):
    out_data = json.dumps(dict_)
    with open(outfile, 'w+') as f:
        f.write(out_data)
    return dict_

def load_json(path):
    with open(path) as json_file:
        jdata = json.load(json_file)
    return jdata

def rm_hidden(root_path):
    # Grab hidden files:
    hidden_files = [os.path.join(root_path, f) for f in os.listdir(root_path) if f.startswith('.')]

    # Remove them:
    for f in hidden_files:
        try:
            sh.rmtree(f)
        except:
            try:
                os.remove(f)
            except:
                print('error removing {}'.format(f))

    # Grab again:
    hidden_files = [os.path.join(root_path, f) for f in os.listdir(root_path) if f.startswith('.')]

    if len(hidden_files) > 0:
        error_string = '''
        Operation did not remove all hidden files as expected.
        Remaining files (you may need to remove manually):
        {}
        '''.format(hidden_files)
        raise AssertionError(error_string)

def longestSubstring(str1, str2):
    # initialize SequenceMatcher object with
    # input string
    seqMatch = SequenceMatcher(None, str1, str2)

    # find match of longest sub-string
    # output will be like Match(a=0, b=0, size=5)
    match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))

    # return longest substring
    return str1[match.a: match.a + match.size]

def get_shared_string(a, b):
    '''
    Returns the bkdn sheet title that most closely matches the input filenames.
    '''
    matches = []
    for i in a:
        for j in b:
            match = longestSubstring(i.lower(), j.lower())
            if match:
                matches.append((len(match), match, i))
    return sorted(matches)[-1][-1].strip()

def concat_w_str(x, s):

    null_values = [str(s).lower() for s in ['N/A', np.nan, 'None', 'NaT', 'NAN']]

    to_concat = [str(v) for v in x if str(v).lower() not in null_values]

    res = f'{s}'.join(to_concat)

    return res

def mkdir_force(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def parse_datetime(s):
    if s:
        return parse(s)
    else:
        return False

def df_styler(val):
    """
    applies css styles to each val in a df.
    """
    text_dec = ''
    font_weight = ''
    if '*' in val:
        color = 'red'
        text_dec = 'underline overline'
        font_weight = 'bold'
    else:
        color = 'black'

    return f'color: {color}; text-decoration: {text_dec}; font-weight: {font_weight}'

def markup_answers(r):

    r.A = str(r.A)

    if not r.correct:
        r.A = str(r.A) + '*'
    return r[['Q', 'A']]

def make_report_table(df, n_cols=3, fillna=True, fillval=''):
    '''
    Styles an input df, and returns the style tag and raw html
    :param df: student section_df
    :param n_cols: number of cols for a table - default three
    :param fillna: whether to fill NANs or not
    :param fillval: if fillna, what to fill with.
    :return: raw html style tag and table tag
    '''

    v = df.iloc[:, 1:4]
    v.columns = ['A', 'correct', 'Q']
    v = v.apply(markup_answers, axis=1)

    n_vals = len(v)
    col_height = n_vals // n_cols
    if n_vals % n_cols != 0:
        col_height += 1
    subcols = []
    for i in range(n_cols):
        subcol = v.iloc[(i * col_height):(i * col_height) + col_height].reset_index(drop=True)
        subcols.append(subcol)
    reshaped = pd.concat(subcols, axis=1)
    reshaped.columns = np.array([[f'Q_{i}', f'A_{i}'] for i in range(int(reshaped.shape[1] / 2))]).ravel()

    if fillna:
        reshaped = reshaped.fillna(fillval)

    styled = reshaped.style.applymap(df_styler).hide_index()

    html_unique_cols = styled.render()

    soup = BeautifulSoup(html_unique_cols, features='lxml')

    for th in soup.find_all('th'):
        if '_' in th.text:
            th.string = th.text[0]
        else:
            th.string = ''

    table_styles = soup.find('style')
    table_html = list(soup.find('body').children)[0]
    table_classes = ["table-bordered", "table-striped", "table-hover"]
    table_html['class'] = table_html.get('class', '') + ' '.join(table_classes)

    return str(table_styles), str(table_html)

def write_html(html, fname='out.html'):
    with open(fname, 'w+') as f:
        f.write(str(html))

'''
Free conversion using WKTOHTML
'''
import subprocess
def html_to_pdf(path, watermark_path, out_fname=None, convert_fname=False):
    out_dir = '/'.join(path.split('/')[:-1])
    in_fname = path.split('/')[-1]
    if (not out_fname) or (convert_fname):
        split = path.split('.')
        pdf_outpath = '.'.join(split[:-1]) + '.pdf'
        out_fname = pdf_outpath

    subprocess.run(["cd", out_dir])
    subprocess.run(["wkhtmltopdf", '--enable-local-file-access','-B', '10mm', path, pdf_outpath])

    print('Watermarking PDF!')
    watermark_fname = ('WM_'+in_fname).replace('.html', '.pdf')
    watermarked_outpath = os.path.join(out_dir,watermark_fname)
    watermark(pdf_outpath, watermark_path, watermarked_outpath)

    return watermarked_outpath

def watermark(input_file, watermark_file, output_file):
    # create a pdf writer object for the output file
    pdf_writer = PyPDF2.PdfFileWriter()

    with open(input_file, "rb") as filehandle_input:
        # read content of the original file
        pdf = PyPDF2.PdfFileReader(filehandle_input)

        with open(watermark_file, "rb") as filehandle_watermark:
            # read content of the watermark
            watermark = PyPDF2.PdfFileReader(filehandle_watermark)

            for page in pdf.pages:
                # merge the two pages
                page.mergePage(watermark.getPage(0))

                # add page
                pdf_writer.addPage(page)

            with open(output_file, "wb") as filehandle_output:
                # write the watermarked file to the new file
                pdf_writer.write(filehandle_output)

def merge_pdfs(pdfs, out_fname):
    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(pdf, import_bookmarks=False)

    merger.write(out_fname)
    merger.close()

def ensure_path(path):
    filtered_path = path
    final = path.split('/')[-1]
    file_in_path = ('.' in final)
    if file_in_path:
        filtered_path = '/'.join(path.split('/')[:-1])
    if not os.path.exists(filtered_path):
        os.makedirs(filtered_path)
    return path

def chunk_list(l,n):
    return [l[i:i+n] for i in range(0,len(l),n)]

def gen_row(shape, idx):
    raw = pd.Series([np.nan for _ in range(shape)])
    raw.index = idx
    return raw

def fix_missing_students(paths):
    dfs = [pd.read_csv(path) for path in paths]

    #Log filenames and list indices for empty student files:
    empty_files = [paths[idx] for idx, df in enumerate(dfs) if len(df) <= 1]
    empty_files_indices = [idx for idx, df in enumerate(dfs) if len(df) <= 1]
    print(empty_files)
    print(empty_files_indices)

    # Remove empty files from consideration
    paths = [path for idx, path in enumerate(paths) if idx not in empty_files_indices]
    dfs = [df for idx, df in enumerate(dfs) if idx not in empty_files_indices]

    ids = [set(df['Student ID']) for df in dfs]

    missing = dict()  # {i:dict() for i in range(len(dfs))}
    for i, id_set in enumerate(ids):
        for j, jd_set in enumerate(ids):
            if i != j:
                diff = id_set - jd_set
                for id_ in diff:
                    # Read: j is missing id_, and they can be found in i.
                    if not j in missing:
                        missing[j] = {id_: i}
                    else:
                        if not id_ in missing[j]:
                            missing[j][id_] = i

    for missing_idx, d in missing.items():
        for student_id, found_idx in d.items():
            # get id info from df that contains the missing data:
            info = dfs[found_idx][dfs[found_idx]['Student ID'] == student_id].loc[:, :'Version']

            # create an empty row in the shape of the df (cols axis) which is MISSING the data:
            empty_row = gen_row(dfs[missing_idx].shape[1], dfs[missing_idx].columns)

            # Fill in the identifying info in empty_row
            # NOTE: info.squeeze() turns into into a series.
            print(info.squeeze())
            empty_row.loc[:'Version'] = info.squeeze()
            empty_row.Score = 0

            # Overwrite the old df (which had missing data) with its new version,
            # which now contains the missing student id and dummy NAN answers.
            dfs[missing_idx] = pd.concat([dfs[missing_idx], pd.DataFrame(empty_row).T])

    for df, path in zip(dfs, paths):
        df.sort_values('First Name').to_csv(path, index=False)

    return empty_files

def check_answers(df, student):
    def process_key(x):
        s = str(x)
        if 'or' in s and len(s) > 2:
            return [v.strip() for v in s.split('or')]
        else:
            return [s]
    old_key = df.key.copy()
    df.key = df.key.apply(process_key)
    df['correct'] = df.apply(lambda r: str(r[student]) in r.key, axis=1)
    df.key = old_key
    return df

def retry(f, n, *fargs, **fkwargs):
    for t in range(n):
        try:
            res = f(*fargs, **fkwargs)
            return res
        except ValueError:
            print(f'Failed {t} time(s); trying again {n-t} more time(s).')
            continue