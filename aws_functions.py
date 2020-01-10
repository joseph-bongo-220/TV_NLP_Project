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
import psycopg2
import NLP

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

def get_script_data(show, connection, cols = ["*"], bucket_name = config["aws"]["s3_bucket_name"], s3_path=None, seasons = None):
    if seasons is None:
        query = "SELECT {cols} FROM {show}".format(show = re.sub(" ", "_", show).lower(), cols = ", ".join(cols))   
    else:
        seasons = ["'" + str(i) + "'" for i in seasons]
        query = """SELECT {cols} FROM {show}
            WHERE season in {season}""".format(show = re.sub(" ", "_", show).lower(), season = "(" + ", ".join(seasons) + ")", cols = ", ".join(cols))

    try:
        print(query)
        data = pd.read_sql(query, connection)
        print("Data Gathered from PostgreSQL Database")

    except pd.io.sql.DatabaseError:
        path = config[show]["pickle_path"]

        if s3_path is None:
            s3_path=path    
        
        client = boto3.client("s3")

        try:
            obj = client.get_object(Bucket=bucket_name, Key=s3_path)
            data = pickle.loads(obj['Body'].read())
            put_show_rds(data, show, connection = connection)

        except ClientError as ex:
            if ex.response["Error"]["Code"] == 'NoSuchKey':
                print("Gathering Data from Genius.com")
                Scraper = Genius_TV_Scraper(show=show, seasons=seasons)
                data = Scraper.get_scripts()
                put_show_rds(data, show, connection = connection)
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

def pkls_to_rds(shows = config["app"]["shows"]):
    """dumps the show pkl files from S3 into Postgres"""
    # connect to postgres db
    connection = connect_to_rds(username = os.environ["RDS_USERNAME"], password = os.environ["RDS_PASSWORD"])

    # iterate over each show
    for show in shows:
        # get dataframe from S3
        data = NLP.get_pickle_data_frames(show)
        put_show_rds(data, show, connection)

def put_show_rds(data, show, connection = None, new = False):
    if connection is None:
        connection = connect_to_rds(username = os.environ["RDS_USERNAME"], password = os.environ["RDS_PASSWORD"])

    cursor = connection.cursor()

    # remove non ASCII characters
    data["narration"] = [re.sub("’", "'", x) for x in data["narration"]]
    data["narration"] = [re.sub("\t", " ", x) for x in data["narration"]]
    
    # create csv path write csv file to local machine
    csv_path = re.sub(" ", "_", show).lower()+".csv"
    data.to_csv(csv_path, sep = '\t', index = False)
    
    if new:
        # SQL to create a table for the given show
        create_sql = """CREATE TABLE {show} (
            character_name text,
            line text,
            narration text,
            show text,
            episode text,
            season integer,
            url text)""".format(show = re.sub(" ", "_", show).lower())

        # create table
        cursor.execute(create_sql)
        connection.commit() 

    # populate table with csv
    with open(csv_path, 'r', encoding='utf-8') as row:
        # skip column names
        next(row)
        cursor.copy_from(row, re.sub(" ", "_", show).lower(), sep='\t')
    connection.commit()
    print(show + " table created. Data populated from " + csv_path + ".")

    # remove csv path from local machine
    os.remove(csv_path)  
    print("Removing " + csv_path)

def connect_to_rds(username, password, host = os.environ["RDS_ENDPOINT"], port = config["aws"]["postgres"]["port"], db_name = config["aws"]["postgres"]["name"]):
    try:
        connection = psycopg2.connect(
            host = host,
            port = port,
            user = username,
            password = password,
            database = db_name)
        print("Connected to AWS RDS")
        return connection
    except:
        print("""Something has gone wrong. Please verify that your username
         and password are correct and/or that your account has the proper permissions""")

def delete_table(table_name):
    connection = connect_to_rds(username = os.environ["RDS_USERNAME"], password = os.environ["RDS_PASSWORD"])
    cursor = connection.cursor()
    delete_sql = "DROP TABLE {table} CASCADE;".format(table = table_name)
    cursor.execute(delete_sql)
    connection.commit()
    print(table_name + " Deleted")

if __name__ == '__main__':
    # delete_table("game_of_thrones")
    # pkls_to_rds()
    connection = connect_to_rds(username = os.environ["RDS_USERNAME"], password = os.environ["RDS_PASSWORD"])
    try:
        office = pd.read_sql("SELECT * FROM the_office", connection)
    except pd.io.sql.DatabaseError:
        print("memes")
    print(list(office.columns))
