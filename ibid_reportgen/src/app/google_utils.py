'''
Google API
'''
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from google.oauth2 import service_account
from google.cloud import storage

from pathlib import Path
import pickle
import os
import pandas as pd
from tqdm import tqdm

'''
For Google Sheets API:
'''


def pickle_save(path, obj):
    if '.pickle' not in path:
        path += '.pickle'
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(path):
    if '.pickle' not in path:
        path += '.pickle'
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
        return b


# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SHEET_ID = 'REDACTED'


def sheets_auth(token_path, creds_path):
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)
    return service


'''
For GCP Bucket IO
'''


def _get_gs_client(resource_path: str) -> storage.Client:
    credspath = os.path.join(resource_path, 'REDACTED')
    credentials = service_account.Credentials.from_service_account_file(credspath)
    client = storage.Client(project='REDACTED', credentials=credentials)
    return client


def list_blobs_with_prefix(bucket_name, prefix, client: storage.Client, delimiter=None) -> list:
    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = client.list_blobs(
        bucket_name, prefix=prefix, delimiter=delimiter
    )
    return list(blobs)


def download_blob(blob, dst, client: storage.Client) -> None:
    with open(dst, 'wb') as file_obj:
        client.download_blob_to_file(blob, file_obj)


def download_gs_folder(bucket, gs_path, dst_path, client: storage.Client) -> None:
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    blobs = list_blobs_with_prefix(bucket, gs_path, client=client)
    blobs = list(blobs)
    print('Download starting - will print "DONE" when finished!')
    for blob in tqdm(blobs):
        fname = Path(blob.name).name
        dst = os.path.join(dst_path, fname)
        print(f'Downloading: {fname}')
        print(f'To: {dst}')
        download_blob(blob, dst, client=client)
    print("DONE")


def upload_blob(resource_path, bucket_name, source_file_name, destination_blob_name):
    client = _get_gs_client(resource_path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {os.path.join(bucket_name, destination_blob_name)}."
    )
