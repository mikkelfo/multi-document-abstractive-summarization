import gzip
import json
import pandas as pd
import os
import string
import json
import random
import math

''' 
    *****     DaNewsroom     *****
'''
random.seed(10)

def split_dataframe(name, df, ratios=[0.9, 0.05, 0.05]):
    assert sum(ratios) == 1
    N = len(df)

    train = math.ceil(N*ratios[0])
    val = math.ceil(N*ratios[1])
    test = math.ceil(N*ratios[2])

    df.sample(frac=1, random_state=10)

    val_data = df[:val]
    test_data = df[val:val+test]
    train_data = df[val+test:]

    for split, data in zip(['train', 'test', 'validation'], [train_data, test_data, val_data]):
        dir = f'data/processed/danewsroom/{name}'
        if not os.path.isdir(dir):
            os.mkdir(dir)
        data.to_csv(f'{dir}/{split}.csv')
        

def segment_newsroom_by_type(file='data/raw/danewsroom.csv'):
    df = pd.read_csv(file)

    # Remove empty summaries
    df = df[df.summary.str.translate(str.maketrans('', '', string.punctuation)) != '']
    # Split into the 3 subsets (key, dataframe)
    for name, dataframe in df.groupby('density_bin'):
        dataframe.to_csv(f'data/raw/danewsroom_{name}.csv')
        # Only keep relevant information
        cleaned_df = dataframe[['text', 'summary']]
        split_dataframe(name, cleaned_df)


# Convert jsonl.gz file to .csv
def gz_to_csv(file="danewsroom.jsonl.gz"):
    data = [json.loads(line) for line in gzip.open(f'data/raw/{file}', 'rb')]
    df = pd.DataFrame(data)
    new_name = file.split(".")[0]    # Take first part of filename (excluding .type)
    df.to_csv(f'data/raw/{new_name}.csv')


if __name__ == '__main__':
    segment_newsroom_by_type()