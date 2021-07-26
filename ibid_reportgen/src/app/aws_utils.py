import os
import pandas as pd
from pathlib import Path
import shutil
import subprocess
import boto3
from datetime import datetime


def get_s3():
    try:
        s3 = boto3.client('s3')
        assert 'REDACTED' in [b['Name'] for b in s3.list_buckets()['Buckets']], \
            'AWS is not configured on this device. Attempting autoconfig.'
        print('S3 configured through Boto3.')
        return s3
    except:
        aws_login(script_dir='REDACTED')
        return get_s3()


S3 = get_s3()


def upload_to_s3(file, dest, bucket_name='REDACTED', s3=S3):
    '''
    Uploads a file object as a blob.
    '''
    # Remove s3:// from bucket name if erroneously included.
    bucket_name = bucket_name.replace('s3://', '')
    # Log upload
    print(f'Uploading {file} to s3://{os.path.join(bucket_name, dest)}')
    # Run upload!
    try:
        with open(file, "rb") as f:
            s3.upload_fileobj(f, bucket_name, dest)
    except:
        print(f'Unable to upload {file} to s3://{os.path.join(bucket_name, dest)}.\n'
              f'This is likely an internet connectivity error.')


def download_from_s3(source, dest, bucket_name='REDACTED', s3=S3):
    '''
    Downloads a file object from AWS.
    '''
    try:
        s3.download_file(bucket_name, source, dest)
    except:
        print('Unable to perform download - please check your internet connection.')


from awscli.clidriver import create_clidriver


def aws_cli(*cmd):
    '''
    To run:
    aws_cli(['s3', 'sync', './src', 's3://REDACTED', '--delete'])
    '''

    old_env = dict(os.environ)
    try:

        # Environment
        env = os.environ.copy()
        env['LC_CTYPE'] = u'en_US.UTF'
        os.environ.update(env)

        # Run awscli in the same process
        exit_code = create_clidriver().main(*cmd)

        # Deal with problems
        if exit_code > 0:
            raise RuntimeError('AWS CLI exited with code {}'.format(exit_code))
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def aws_login(script_dir, as_root=False):
    print('Script Dir is:')
    print(script_dir)
    # make a call to login bash script
    if as_root:
        cmd_path = os.path.join(script_dir, "REDACTED")
        subprocess.call(['bash', f'{cmd_path}'])
    else:
        cmd_path = os.path.join(script_dir, "REDACTED")
        subprocess.call(['bash', f'{cmd_path}'])


def list_files(path, match='.csv'):
    return [Path(os.path.join(path, f)) for f in os.listdir(path) if match in f]


def update_test_archives_to_aws(files, test_name):
    now = datetime.now().date().isoformat()
    archive_location = f'test_archives/{test_name}/{now}'
    print('Archiving!')
    for f in files:
        upload_to_s3(f, os.path.join(archive_location, Path(f).name))
