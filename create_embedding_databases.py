
import os
import textwrap
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
import pickle


def create_textual_rep_table(schema_name, table_name):
    return textwrap.dedent(f"""
        Name of the schema: {schema_name}
        Name of the table: {table_name}
    """)

def create_textual_rep_table_column(schema_name, table_name, all_column_names):
    return textwrap.dedent(f"""
        Name of the schema: {schema_name}
        Name of the table: {table_name}
        Name of the columns: {str(all_column_names)}
    """)

def create_textual_rep_table_data(dataset_dir, file_name, schema_name, table_name, sample_count):
    df = pd.read_excel(f"{dataset_dir}/{file_name}")

    sample_count = min(sample_count, df.shape[0])
    random_sample = df.sample(n=sample_count, random_state=42)

    column_names = df.columns.to_list()

    sample_data = {}
    for column in column_names:
        sample_data[column] = random_sample[column].to_list()
    
    return textwrap.dedent(f"""
        Name of the schema: {schema_name}
        Name of the table: {table_name}
        Sample data from the table: {sample_data}
    """)

def create_textual_rep_table_stat(dataset_dir, file_name, schema_name, table_name):
    df = pd.read_excel(f"{dataset_dir}/{file_name}")
    column_names = df.columns.to_list()
    table_stat = {}
    for column in column_names:
        # print(column)
        column_dtype = df[column].dtype
        if pd.api.types.is_string_dtype(column_dtype):
            # max_len = df[column].apply(len).max()
            # min_len = df[column].apply(len).min()
            max_len = df[column].apply(lambda x: len(str(x)) if isinstance(x, str) else 0).max()
            min_len = df[column].apply(lambda x: len(str(x)) if isinstance(x, str) else 0).min()
            table_stat[column] = {
                'type': 'string',
                'max_length': max_len.item(),
                'min_length': min_len.item()
            }
        elif pd.api.types.is_numeric_dtype(column_dtype):
            max_val = df[column].max()
            min_val = df[column].min()
            table_stat[column] = {
                'type': 'numeric',
                'max': max_val.item(),
                'min': min_val.item()
            }
        elif pd.api.types.is_bool_dtype(column_dtype):
            table_stat[column] = {
                'type': 'boolean',
                'unique_values': df[column].unique()
            }
        elif pd.api.types.is_datetime64_dtype(column_dtype):
            max_date = df[column].max()
            min_date = df[column].min()
            table_stat[column] = {
                'type': 'datetime',
                'max_date': max_date,
                'min_date': min_date
            }
        else:
            table_stat[column] = {'type': str(column_dtype)}
    
    return textwrap.dedent(f"""
        Name of the schema: {schema_name}
        Name of the table: {table_name}
        Row count: {df.shape[0]}
        Column count: {df.shape[1]}
        Column statistics of the table: {table_stat}
    """)


def create_textual_rep_column(schema_name, table_name, column_name):
    return textwrap.dedent(f"""
        Name of the schema: {schema_name}
        Name of the table: {table_name}
        Name of the column: {column_name}
    """)

def create_textual_rep_column_data(dataset_dir, file_name, schema_name, table_name, column_name, sample_count):
    df = pd.read_excel(f"{dataset_dir}/{file_name}")

    sample_count = min(sample_count, df.shape[0])
    random_sample = df.sample(n=sample_count, random_state=42)
    sample_data = random_sample[column_name].to_list()
    
    return textwrap.dedent(f"""
        Name of the schema: {schema_name}
        Name of the table: {table_name}
        Name of the column: {column_name}
        Sample data from the column: {sample_data}
    """)

def create_textual_rep_column_stat(dataset_dir, file_name, schema_name, table_name, column_name):
    df = pd.read_excel(f"{dataset_dir}/{file_name}")
    column_stat = {}

    column_dtype = df[column_name].dtype

    if pd.api.types.is_string_dtype(column_dtype):
        max_len = df[column_name].apply(lambda x: len(str(x)) if isinstance(x, str) else 0).max()
        min_len = df[column_name].apply(lambda x: len(str(x)) if isinstance(x, str) else 0).min()

        column_stat['type'] = 'string'
        column_stat['max_length'] = max_len.item()
        column_stat['min_length'] = min_len.item()

    elif pd.api.types.is_numeric_dtype(column_dtype):
        max_val = df[column_name].max()
        min_val = df[column_name].min()

        column_stat['type'] = 'numeric'
        column_stat['max_length'] = max_val.item()
        column_stat['min_length'] = min_val.item()

    elif pd.api.types.is_bool_dtype(column_dtype):
        column_stat['type'] = 'boolean'
        column_stat['unique_values'] = df[column_name].unique()
 
    elif pd.api.types.is_datetime64_dtype(column_dtype):
        max_date = df[column_name].max()
        min_date = df[column_name].min()

        column_stat['type'] = 'datetime'
        column_stat['max_date'] = max_date
        column_stat['min_date'] = min_date

    else:
        column_stat['type'] = str(column_dtype)
    
    return textwrap.dedent(f"""
        Name of the schema: {schema_name}
        Name of the table: {table_name}
        Name of the column: {column_name}
        Column statistics: {column_stat}
    """)



def create_embedding_vector(model, tokenizer, txt):
    inputs = tokenizer(txt, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()



def get_model_tokenizer(model_name):
    if model_name == 'bert-base-uncased':
        model = BertModel.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)

    return model, tokenizer



# def main():

dataset_dir = 'Datasets/AdventureWorks2022_Modified'
file_names = os.listdir(dataset_dir)
data_file_count = 0


sample_count = 5


table_embeddings_db = {}
column_embeddings_db = {}

model_name = 'bert-base-uncased'
model, tokenizer = get_model_tokenizer(model_name)


for file_name in file_names:

    print(f"Processing {file_name}")

    schema_name, table_name = file_name.split('.')[0].split('_')

    if schema_name != '000000':
        data_file_count += 1

        ####################################
        #
        # CREATE EMBEDDINGS FOR TABLE
        #
        ####################################

        # get embedding text for schema name and table_name
        embedding_type = '01'
        txt = create_textual_rep_table(schema_name, table_name)
        # print(txt)
        embedding = create_embedding_vector(model, tokenizer, txt)
        table_embeddings_db[f"{schema_name}_{table_name}_{embedding_type}"] = embedding


        # get the name of the columns of the table 
        df = pd.read_excel(f"{dataset_dir}/{file_name}", nrows=0)
        column_names = df.columns.to_list()

        # get embedding text for schema name, table_name and all column names
        embedding_type = '02'
        txt = create_textual_rep_table_column(schema_name, table_name, column_names)
        # print(txt)
        embedding = create_embedding_vector(model, tokenizer, txt)
        table_embeddings_db[f"{schema_name}_{table_name}_{embedding_type}"] = embedding


        # get embedding text for schema name, table_name and sample data
        embedding_type = '03'
        txt = create_textual_rep_table_data(dataset_dir, file_name, schema_name, table_name, sample_count)
        # print(txt)
        embedding = create_embedding_vector(model, tokenizer, txt)
        table_embeddings_db[f"{schema_name}_{table_name}_{embedding_type}"] = embedding
        

        # get embedding text for schema name, table_name and column stats
        embedding_type = '04'
        txt = create_textual_rep_table_stat(dataset_dir, file_name, schema_name, table_name)
        # print(txt)
        embedding = create_embedding_vector(model, tokenizer, txt)
        table_embeddings_db[f"{schema_name}_{table_name}_{embedding_type}"] = embedding


        ####################################
        #
        # CREATE EMBEDDINGS FOR COLUMNS
        #
        ####################################

        for column_name in column_names:

            # get embedding text for schema name, table_name and column_name
            embedding_type = '01'
            txt = create_textual_rep_column(schema_name, table_name, column_name)
            # print(txt)
            embedding = create_embedding_vector(model, tokenizer, txt)
            column_embeddings_db[f"{schema_name}_{table_name}_{column_name}_{embedding_type}"] = embedding

            # get embedding text for schema name, table_name, column_name, sample data for column
            embedding_type = '02'
            txt = create_textual_rep_column_data(dataset_dir, file_name, schema_name, table_name, column_name, sample_count)
            # print(txt)
            embedding = create_embedding_vector(model, tokenizer, txt)
            column_embeddings_db[f"{schema_name}_{table_name}_{column_name}_{embedding_type}"] = embedding     

            # get embedding text for schema name, table_name, column_name, sample data for column
            embedding_type = '03'
            txt = create_textual_rep_column_stat(dataset_dir, file_name, schema_name, table_name, column_name)
            # print(txt)
            embedding = create_embedding_vector(model, tokenizer, txt)
            column_embeddings_db[f"{schema_name}_{table_name}_{column_name}_{embedding_type}"] = embedding        

    # if data_file_count == 1:
    #     break

with open('table_embeddings_db.pkl', 'wb') as table_f:
    pickle.dump(table_embeddings_db, table_f)

with open('column_embeddings_db.pkl', 'wb') as column_f:
    pickle.dump(column_embeddings_db, column_f)