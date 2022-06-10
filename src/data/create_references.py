import json
from nltk.tokenize import sent_tokenize
import os

if not os.path.isdir('data/processed/references'):
    os.mkdir('data/processed/references')

reference, text = zip(*json.load(open('data/processed/wcep/test.json', 'r')))
reference = list(reference)
reference = ['\n'.join(sent_tokenize(ref)) for ref in reference]
with open('data/processed/references/wcep.json', 'w') as f:
    json.dump(reference, f, indent=4)

text, references = zip(*json.load(open('data/processed/cnn-dm/test.json', 'r')))
reference = list(reference)
reference = ['\n'.join(sent_tokenize(ref)) for ref in reference]
with open('data/processed/references/cnn-dm.json', 'w') as f:
    json.dump(reference, f, indent=4)

import pandas as pd
df = pd.read_csv('data/processed/danewsroom/abstractive/test.csv')
reference = df['summary'].to_list()
reference = ['\n'.join(sent_tokenize(ref)) for ref in reference]
with open('data/processed/references/danewsroom.json', 'w') as f:
    json.dump(reference, f, indent=4)

