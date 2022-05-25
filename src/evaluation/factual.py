import json
import spacy

def lowercase(array):
    return [el.text.lower() for el in array]

def NER_overlap_cnn(pipeline):
    with open('summaries/best/cnn-dm.json', 'r') as f:
        hypothesis = json.load(f)
    hypothesis = sum(hypothesis, [])
    with open('summaries/best/cnn-dm-xlm.json', 'r') as f:
        hypothesis_xlm = json.load(f)
    hypothesis_xlm = sum(hypothesis_xlm, [])
    with open('data/processed/cnn-dm/test.json', 'r') as f:
        data = json.load(f)
    text, reference = zip(*data)

    nlp = spacy.load(pipeline, exclude=["tagger", "parser", "attribute_ruler", "lemmatizer"])

    correct_hyp, correct_xlm, correct_ref  = 0, 0, 0
    for i in range(len(hypothesis)):
        doc_hyp = lowercase(nlp(hypothesis[i]).ents)
        doc_xlm = lowercase(nlp(hypothesis_xlm[i]).ents)
        doc_ref = lowercase(nlp(reference[i]).ents)
        doc_text = lowercase((nlp(text[i]).ents))

        count_hyp = [1 if ent in doc_text else 0 for ent in doc_hyp]
        count_xlm = [1 if ent in doc_text else 0 for ent in doc_xlm]
        count_ref = [1 if ent in doc_text else 0 for ent in doc_ref]

        correct_hyp += sum(count_hyp) / len(count_hyp) if len(count_hyp) > 0 else 0
        correct_xlm += sum(count_xlm) / len(count_xlm) if len(count_xlm) > 0 else 0
        correct_ref += sum(count_ref) / len(count_ref) if len(count_ref) > 0 else 0

    print(correct_hyp / len(hypothesis))
    print(correct_xlm / len(hypothesis))
    print(correct_ref / len(hypothesis))

def NER_overlap_wcep(pipeline):
    with open('summaries/best/wcep-mean.json', 'r') as f:
        hypothesis = json.load(f)
    hypothesis = sum(sum(hypothesis, []), [])
    with open('summaries/best/wcep-serial.json', 'r') as f:
        hypothesis_serial = json.load(f)
    hypothesis_serial = sum(sum(hypothesis_serial, []), [])
    with open('data/processed/wcep/test.json', 'r') as f:
        data = json.load(f)
    reference, text = zip(*data)


    nlp = spacy.load(pipeline, exclude=["tagger", "parser", "attribute_ruler", "lemmatizer"])

    correct_hyp, correct_ref, correct_serial = 0, 0, 0
    for i in range(len(hypothesis)):
        doc_hyp = lowercase(nlp(hypothesis[i]).ents)
        doc_serial = lowercase(nlp(hypothesis_serial[i]).ents)
        doc_ref = lowercase(nlp(reference[i]).ents)
        doc_text = lowercase(sum([list(nlp(t).ents) for t in text[i]], []))

        count_hyp = [1 if ent in doc_text else 0 for ent in doc_hyp]
        count_ref = [1 if ent in doc_text else 0 for ent in doc_ref]
        count_serial = [1 if ent in doc_text else 0 for ent in doc_serial]

        correct_hyp += sum(count_hyp) / len(count_hyp) if len(count_hyp) > 0 else 0
        correct_serial += sum(count_serial) / len(count_serial) if len(count_serial) > 0 else 0
        correct_ref += sum(count_ref) / len(count_ref) if len(count_ref) > 0 else 0

    print(correct_hyp / len(hypothesis))
    print(correct_serial / len(hypothesis))
    print(correct_ref / len(hypothesis))


if __name__ == '__main__':
    NER_overlap_cnn('en_core_web_sm')
    print()
    NER_overlap_cnn('en_core_web_trf')