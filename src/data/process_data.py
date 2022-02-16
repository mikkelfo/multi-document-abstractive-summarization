import gzip
import json
import pandas as pd
import os
import string
from rouge import Rouge
from pathlib import Path
import pickle

''' 
    *****     CNN/Dailymail     ***** 
'''
def chunk_files(chunk_dir, chunk_size=1000):
    '''
        Chunks the raw files
            The files are processed by process_stories
            and cleaned by clean_lines

        Note: Each chunk is not exactly 1000 stories, as empty stories (n=114) are discarded
    '''
    media = chunk_dir.split("/")[0]
    dir_path = 'data/raw/chunks/' + media
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    num_chunk = 0
    files = list(Path('data/raw/' + chunk_dir).glob('*.story'))
    for i in range(0, len(files), chunk_size):
        chunk = files[i:(i+chunk_size)]
        processed_chunk = process_stories(chunk, media)    # Note: Chunk_size is not exactly 1000 due to the discard behaviour in process_stories
        with open(f'{dir_path}/chunk_{num_chunk}.pickle', 'wb') as handle:
            pickle.dump(processed_chunk, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Chunk", num_chunk)
        num_chunk += 1


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


def process_stories(chunk, media):
    '''
        
        Note: Discards empty stories
    '''
    processed_chunk = {
        'story': [],
        'highlight': []
    }
    for file in chunk:
        text = file.read_text(encoding="utf-8")
        # Splits story and highlights
        story, *highlights = text.split("@highlight")
        story = clean_lines(story.split('\n\n'), media)
        highlight = clean_lines(highlights, media)

        if story == '' or highlights == []:
            continue
        
        processed_chunk['story'].append(story)
        processed_chunk['highlight'].append(highlight)

    return processed_chunk


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
    chunk_files('cnn/stories')
    chunk_files('dailymail/stories')