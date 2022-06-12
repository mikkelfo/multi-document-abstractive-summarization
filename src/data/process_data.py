import gzip
import json
import pandas as pd
import os
import string
from pathlib import Path
import json
import random
import math
from utils import read_jsonl_gz

random.seed(10)

''' 
    *****     CNN/Dailymail     ***** 
'''
def process_cnndm():
    data = []
    for media in ['cnn', 'dailymail']:
        dir = 'data/raw/' + media + '/stories'
        files = list(Path(dir).glob('*.story'))

        for file in files:
            text = file.read_text(encoding="utf-8")
            # Splits story and highlights
            story, *highlights = text.split("@highlight")
            story = clean_lines(story.split('\n\n'), media)
            highlight = '. '.join([h.replace("\n", "") for h in highlights])

            # Skip empty articles with empty story/highlight
            if story == '' or highlights == []:
                continue

            data.append((story, highlight))
    with open('data/processed/cnn-dm.json', 'w') as file:
        json.dump(data, file, indent=4)


def split_cnndm(ratios=[0.9, 0.05, 0.05]):
    assert sum(ratios) == 1
    with open('data/processed/cnn-dm.json', 'r') as file:
        data = json.load(file)
    N = len(data)

    train = math.ceil(N*ratios[0])
    val = math.ceil(N*ratios[1])
    test = math.ceil(N*ratios[2])

    random.shuffle(data)

    val_data = data[:val]
    test_data = data[val:val+test]
    train_data = data[val+test:]

    for name, data in zip(['train', 'test', 'validation'], [train_data, test_data, val_data]):
        with open(f'data/processed/cnn-dm/{name}.json', 'w') as file:
            json.dump(data, file, indent=4)


def clean_lines(lines, media):
    '''
        Removes source, author and time stamps
    '''
    cleaned = []
    for line in lines:
        if media == 'cnn':
            index = line.find('(CNN) -- ')
            if index > -1:
                line = line[index + len('(CNN) -- '):]                  # Remove source
        elif media == 'dailymail':
            if line[:2] == 'By' or 'PUBLISHED:' == line or 'UPDATED:' == line:  # Remove author / time stamps
                continue
            if line[6:9] == 'EST':
                continue
            if line == '\n| ':
                continue

        cleaned.append(line)                      # Store as string
    # Remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0]
    return ' '.join(cleaned)


''' 
    *****     DaNewsroom     *****
'''
def segment_newsroom_by_type(file='data/raw/danewsroom.csv'):
    random.seed(10)
    df = pd.read_csv(file)

    # Remove empty summaries
    df = df[df.summary.str.translate(str.maketrans('', '', string.punctuation)) != '']
    # Split into the 3 subsets (key, dataframe)
    for name, dataframe in df.groupby('density_bin'):
        dataframe.to_csv(f'data/raw/danewsroom_{name}.csv')
        # Only keep relevant information
        cleaned_df = dataframe[['text', 'summary']]
        split_dataframe(name, cleaned_df)


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


# Convert jsonl.gz file to .csv
def gz_to_csv(file="danewsroom.jsonl.gz"):
    data = [json.loads(line) for line in gzip.open(f'data/raw/{file}', 'rb')]
    df = pd.DataFrame(data)
    new_name = file.split(".")[0]    # Take first part of filename (excluding .type)
    df.to_csv(f'data/raw/{new_name}.csv')


'''
    *****       WCEP        *****
'''
def process_wcep(dir, cluster_size):
    for name in ['train', 'test', 'validation']:
        data_generator = read_jsonl_gz(f'data/raw/{dir}/{name}.jsonl.gz')
        data = []
        for cluster in data_generator:
            articles = cluster['articles']
            text = [x['text'] for x in articles[:cluster_size]]
            data.append((cluster['summary'], text))
        with open(f'data/processed/{dir}/{name}{cluster_size}.json', 'w') as file:
            json.dump(data, file, indent=4)


'''
    *****       DUC 2004    *****
'''
def process_duc():
    docs, references = [], []
    doc_path = 'data/raw/duc/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs'
    summ_path = 'data/raw/duc/reference'
    dirs = [int(dir) for dir in os.listdir(doc_path)]

    for dir in dirs:
        dir_summ = [open(f'{summ_path}/Task{dir}_reference{i}.txt', 'r').read() for i in range(1, 5)]
        dir_docs = [open(f'{doc_path}/{dir}/D{i}.txt', 'r').read().replace("\n", "").replace("``", '"').replace("''", '"') for i in range(1, 11)]
        references.append(dir_summ)
        docs.append(dir_docs)

    with open('data/processed/duc/reference.json', 'w') as f:
        json.dump(references, f, indent=4)
    with open('data/processed/duc/docs.json', 'w') as f:
        json.dump(docs, f, indent=4)

        

if __name__ == '__main__':
    # process_cnndm()
    # split_cnndm()
    # process_wcep('wcep', 8)
    # process_wcep('wcep', 100)
    process_duc()