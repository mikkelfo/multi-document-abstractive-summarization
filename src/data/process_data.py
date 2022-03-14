import gzip
import json
import pandas as pd
import os
import string
from pathlib import Path
import pickle
import json
import random
import math

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
            highlight = clean_lines(highlights, media)

            # Skip empty articles with empty story/highlight
            if story == '' or highlights == []:
                continue

            data.append((story, highlight))
    file = open('data/processed/cnn-dm.json', 'w')
    json.dump(data, file, indent=4)


def split_cnndm(ratios=[0.9, 0.05, 0.05]):
    assert sum(ratios) == 1
    file = open('data/processed/cnn-dm.json', 'r')
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
        with open(f'data/processed/cnn-dm_{name}.json', 'w') as file:
            json.dump(data, file, indent=4)


# CNN cleaning taken from: https://machinelearningmastery.com/prepare-news-articles-text-summarization/
# Dailymail cleaning created from investigation
def clean_lines(lines, media):
    '''
        Removes source, author and time stamps then
        lowercases, removes punctuation and removes non-alpha characters
    '''
    cleaned = []
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        if media == 'cnn':
            index = line.find('(CNN) -- ')
            if index > -1:
                line = line[index + len('(CNN) -- '):]                  # Remove source
        elif media == 'dailymail':
            if line[:2] == 'By' or 'PUBLISHED:' == line or 'UPDATED:' == line:  # Remove author / time stamps
                continue
        line = line.split()                                 # Split in whitespace
        line = [word.lower() for word in line]              # Converts to lowercase
        line = [word.translate(table) for word in line]     # Remove punctuation 
        line = [word for word in line if word.isalpha()]    # Remove non-alpha characters (digits)
        cleaned.append(' '.join(line))                      # Store as string
    # Remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0]
    return ' '.join(cleaned)


''' 
    *****     DaNewsroom     *****
'''
# Convert jsonl.gz file to .csv
def gz_to_csv(file="danewsroom.jsonl.gz"):
    data = [json.loads(line) for line in gzip.open(f'data/raw/{file}', 'rb')]
    df = pd.DataFrame(data)
    new_name = file.split(".")[0]    # Take first part of filename (excluding .type)
    df.to_csv(f'data/raw/{new_name}.csv')


if __name__ == '__main__':
    # process_cnndm()
    split_cnndm()