import json
import spacy
import pandas as pd

def lowercase(array):
    return [el.text.lower() for el in array]

def get_summaries(*summaries):
    extracted = []
    for summ in summaries:
        with open(f'summaries/best/{summ}.json', 'r') as f:
            data = json.load(f)
        if 'wcep' in summ:
            data = sum(sum(data, []), [])
        else:
            data = sum(data, [])
        extracted.append(data)
    return extracted

def get_original(file):
    if 'danewsroom' in file:
        data = pd.read_csv(f'data/processed/{file}/test.csv')
        text, summary = data['text'].to_list(), data['summary'].to_list()
    else:
        with open(f'data/processed/{file}/test.json', 'r') as f:
            data = json.load(f)
        if file == 'wcep':
            summary, text = zip(*data)
        elif file == 'cnn-dm':
            text, summary = zip(*data)
    return text, summary

def NER_overlap(pipeline, text, reference, summaries):
    nlp = spacy.load(pipeline, exclude=["tagger", "parser", "attribute_ruler", "lemmatizer"])

    summaries = [reference] + summaries

    overlap = [0] * len(summaries)
    ent_count = [0] * len(summaries)
    for i, t in enumerate(text):
        if i in [len(summ) for summ in summaries]:
            print("Stopped early")
            print([len(summ) for summ in summaries])
            break
        if type(t) == str:
            text_ents = lowercase(nlp(t).ents)
        elif type(t) == list:
            text_ents = lowercase(sum([list(nlp(x).ents) for x in t], []))

        for j, summ in enumerate(summaries):
            summ_ents = lowercase(nlp(summ[i]).ents)
            count = [1 if ent in text_ents else 0 for ent in summ_ents]
            if len(count) > 0:
                overlap[j] += sum(count) / len(count)
                ent_count[j] += len(count)

    for i in range(len(summaries)):
        print(overlap[i] / len(text), ent_count[i])


if __name__ == '__main__':
    hypotheses = get_summaries('cnn-dm', 'cnn-dm-xlm')
    text, reference = get_original('cnn-dm')
    NER_overlap('en_core_web_trf', text, reference, hypotheses)
    print()

    hypotheses = get_summaries('wcep-mean', 'wcep-serial')
    text, reference = get_original('wcep')
    NER_overlap('en_core_web_trf', text, reference, hypotheses)

    hypotheses = get_summaries('da-xlm', 'da-xlm-512')
    text, reference = get_original('danewsroom/abstractive')
    NER_overlap('da_core_news_trf', text, reference, hypotheses)
