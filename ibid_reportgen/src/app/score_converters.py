from google_utils import pickle_save, pickle_load
import dominate
from dominate.tags import *
from dominate.util import raw
import pandas as pd
import numpy as np
import itertools as it
import re
from copy import deepcopy

# Names of tests which require no scaler.
NO_SCALER_NEEDED = [
    'Hunter - 01 - DIAG'
]


def get_scaler(context):
    bkdn_dfs = context['bkdn_dfs']
    test_name = context['test_name']
    if test_name in NO_SCALER_NEEDED:
        return 'NO_SCALER_NEEDED'
    elif 'SHSAT' in test_name:
        return bkdn_dfs['_SCALE - SHSAT']
    else:
        try:
            print(f'Searching for "_SCALE - {test_name}"...')
            scaler = bkdn_dfs[f'_SCALE - {test_name}']
            print('Found!')
            return scaler
        except:
            return None


def scaler_exists(scaler):
    return type(scaler) != type(None)


def convert(context):
    '''
    Should ALWAYS return a list of tuples, e.g.:
    [(section, converted_score)...]
    '''

    # Cast section names to all caps
    for k in context['sections']:
        context[k.upper()] = context[k]
    context['sections'] = [s.upper() for s in context['sections']]

    test_name = context['test_name']

    print(f'Test Name is : {test_name}')

    # In the event that a valid scaler has not been provided:
    scaler = get_scaler(context)
    if scaler_exists(scaler):
        # Match SHSAT tests:
        if 'SHSAT' in test_name:
            print('Using SHSAT converter')
            scaled = shsat_converter(context, scaler)

        # Match SAT tests:
        elif re.match(r'SAT - \d\d.*', test_name):
            print('Using SAT Diagnostic Converter')
            scaled = sat_diag_converter(context, scaler)

        # Match ACT tests:
        elif test_name == 'ACT - N - DIAG':
            print('Using ACT N DIAG Converter')
            scaled = act_n_converter(context, scaler)

        elif test_name == 'Hunter - 01 - DIAG':
            print('Using Hunter - 01 - DIAG Converter')
            scaled = hunter_scaler(context)
        else:
            scaled = [('', 'UNAVAILABLE')]
        print(f'{context["student"]} Scaled Score:')
        print(scaled)
        return scaled
    else:
        print(f'No available conversions for {test_name}')
        return [('', 'UNAVAILABLE')]


def hunter_scaler(context):
    # simple percent correct
    converted_scores = []
    total_correct = 0
    total_questions = 0

    for section in context['sections']:
        try:
            n_correct = int(context[section]['n_correct'])
            total_correct += n_correct
            n_total = int(context[section]['n_total'])
            total_questions += n_total
            converted = int((n_correct / n_total) * 100)
        except:
            print(f'ERROR on {section}')
            converted = 0
        converted_scores.append([section, converted])
    converted_scores.append(['TOTAL', int((total_correct / total_questions) * 100)])
    return converted_scores


def shsat_converter(context, scaler):
    converted_scores = []
    for section in context['sections']:
        try:
            n_correct = context[section]['n_correct']
            converted = scaler.T[n_correct]['New']
        except:
            converted = 0
        converted_scores.append((section, converted))

    total = str(int(sum([float(j) for i, j in converted_scores])))
    converted_scores.append(['Total', total])

    return converted_scores


def handle_sat_extraction(context, key):
    section = context.get(key)
    n_correct = 0
    if section:
        n_correct = section['n_correct']
    return n_correct


def sat_diag_converter(context, scaler):
    scaler.columns = [re.sub(r'[^A-z]+', ' ', c).strip().replace(' ', '_').lower() for c in scaler.columns]
    print(scaler.columns)
    math_raw = handle_sat_extraction(context, 'MATH NO CALC') + \
               handle_sat_extraction(context, 'MATH W CALC') + 1

    math_score = int(scaler.math_section_score.get(math_raw, 0))

    wl_raw = handle_sat_extraction(context, 'WRITLANG') + 1
    wl_score = int(scaler.writing_and_language_test_score.get(wl_raw, 0))

    reading_raw = handle_sat_extraction(context, 'READING') + 1
    reading_score = int(scaler.reading_test_score.get(reading_raw, 0))

    # modification per stuart request (9-16-2020)
    reading_and_writing_score = reading_score + wl_score

    total_score = math_score + wl_score + reading_score

    return [['Reading & Writing', reading_and_writing_score], ['Math', math_score], ['TOTAL', total_score]]


def proc_act_values(x):
    try:
        split = [int(i) for i in str(x).split(',')]
        range_ = list(range(split[0], split[-1] + 1))
        return range_
    except:
        return [np.nan]


def act_n_converter(context, scaler):
    scaler.columns = ['SCALED', 'ENGLISH', 'MATH', 'READING', 'SCIENCE']
    scaler = scaler.applymap(proc_act_values)
    section_scores = []
    for section in scaler.columns[1:]:
        score_aligned = np.concatenate(
            scaler.apply(lambda row: list(it.product(row[section], row.SCALED)), axis=1).values.ravel())
        score_map = {k: v for k, v in score_aligned}
        n_correct = context[section]['n_correct']
        section_scores.append([section, int(score_map[n_correct])])

    total = int(np.array(section_scores)[:, 1].astype(int).mean())
    section_scores.append(['TOTAL', total])
    return section_scores
