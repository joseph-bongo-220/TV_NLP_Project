import boto3
from botocore.exceptions import ClientError
from app_config import get_config
import os
import re
import pickle
import numpy as np
from decimal import Decimal
import pandas as pd
import json
from Scraper import Genius_TV_Scraper

config = get_config()

def handle_numpy(data):
    for key, value in data.items():
        if type(value)==type(numpy.ndarray(list())):
            data[key] = [Decimal(str(i)) for i in value]
        elif type(value)==type(dict()) and type(value[list(value.keys())[0]])==type(numpy.ndarray(list())):
            for key2, val2 in value.items():
                value[key2] = [Decimal(str(i)) for i in val2]
    return data

def bulk_push_to_s3():
    s3 = boto3.client("s3")
    s3_data = config["aws"]["files_to_s3"]
    bucket_name = config["aws"]["s3_bucket_name"]

    for local_file in s3_data:
        #keeping names the same here
        s3_file = local_file
        try:
            s3.upload_file(local_file, bucket_name, s3_file)
            print("Uploaded "+ local_file + " to " + s3_file + " in S3 succefully!")
        except FileNotFoundError:
            print("The file " + local_file + " was not found.")

def bulk_push_to_dynamo():
    s3_data = config["aws"]["files_to_s3"]
    pickle_files = [f for f in os.listdir('.') if os.path.isfile(f) and re.search(".pkl",f) is not None and f not in s3_data]

    dynamo = boto3.resource('dynamodb')
    dynamo_table = dynamo.Table(config["aws"]["Dynamo_Table"])

    for _file in pickle_files:
        print(_file)
        with open(_file, 'rb') as f:
            data = pickle.load(f)

        data = handle_numpy(data)

        data.update({"-name-":_file[:-4]})

        dynamo_table.put_item(Item=data)

def get_s3_script_data(show, client, path, bucket_name = config["aws"]["s3_bucket_name"], s3_path=None, seasons = None):
    if s3_path is None:
        s3_path=path

    try:
        obj = client.get_object(Bucket=bucket_name, Key=s3_path)
        data = pickle.loads(obj['Body'].read())

    except ClientError as ex:
        if ex.response["Error"]["Code"] == 'NoSuchKey':
            print("Gathering Data from Genius.com")
            Scraper = Genius_TV_Scraper(show=show, seasons=seasons)
            data = Scraper.get_scripts()
            data.to_pickle(path)
            s3.upload_file(path, bucket_name, s3_path)
            print("Uploaded "+ path + " to " + s3_path + " in s3 succefully!")
        else:
            raise ex
    return data

def get_dynamo_data(item, table, resource, partition_key=config["aws"]["partition_key"], embedding=False):
    dynamo_table = resource.Table(table)
    resp = dynamo_table.get_item(Key={partition_key:item})
    
    try:
        data = resp["Item"]
        del data[partition_key]
        if embedding:
            for key, value in data.items():
                data[key]=np.array([float(i) for i in value])
    except KeyError as ex:
        data = "No Response"

    return data    

if __name__ == '__main__':
    bulk_push_to_s3()
    bulk_push_to_dynamo()
