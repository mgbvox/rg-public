from datetime import (datetime, timedelta)
import typing
from typing import Union, Tuple
from pymongo import MongoClient
import urllib
from data_utils import retry
from dateparser import parse
import re

user = urllib.parse.quote_plus('REDACTED')
password = urllib.parse.quote_plus('REDACTED')
db = urllib.parse.quote_plus('REDACTED')

MONGO_URL = f'REDACTED'

MONGO_CLIENT = MongoClient(MONGO_URL)

DB = MONGO_CLIENT.REDACTED
COLLECTION = DB.report
print('Collection: ', COLLECTION)

'''
context keys:
REDACTED
'''


def upload_to_db(report_context, NAME_ID_MAP, test_meta_info):
    print('DB Upload status:')

    jdata = {
        'student_id': NAME_ID_MAP[report_context['student']],
        'student_name': report_context['student'].upper(),
        'test_name': report_context['test_name'].upper(),
        'scores': report_context['scores'],
        'test_date': test_meta_info['test_date']
    }

    exists = bool(retry(COLLECTION.find_one, 5, jdata))
    if exists:
        print(f'This record already exists in the database - skipping:')
        print(jdata)
    else:
        status = retry(COLLECTION.insert_one, 5, jdata)
        if status.acknowledged:
            print(f'Uploaded data for {report_context["student"]}:')
            print(jdata)
        else:
            print(f'Upload failed for {report_context["student"]}!')


def is_already_run(student_id: str, test_name: str) -> Tuple[str, bool]:
    results = list(COLLECTION.find({'student_id': re.compile(student_id, re.IGNORECASE),
                                    'test_name': re.compile(test_name, re.IGNORECASE)}))
    if results:
        date_tuples = [(r['test_date'], r) for r in results]
        latest = sorted(date_tuples, key=lambda x: x[0])[-1][-1]
        try:
            date = latest['test_date']

            if isinstance(date, str):
                date = parse(date)

            date_line = f'\n{latest["test_name"]} ran on {date.month}/{date.day}/{date.year}'
        except:
            date_line = ''

        message = f'''Student {latest["student_name"]} (ID: {latest["student_id"]}) already run.'''
        message += date_line

        return (message, True)
    else:
        return ('Good to go!', False)
