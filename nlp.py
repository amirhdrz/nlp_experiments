"""
IMPORTANT NOTES:
 - This file contains private API keys, don't share publicy.
"""
from google.cloud import language

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn

from syntax import *

# Google Knowledge Graph API key
api_key = "AIzaSyDPZceqCgLVGytRa14EOvYfcYarjfqMLm0"


def read_data():
    df = pd.read_csv('sample_questions.csv')
    return df[pd.notnull(df['CATEGORY'])]


def build_train_dataset(dataframe):
    tokens = []
    categories = []
    for i, row in dataframe.iterrows():
        text = row['TEXT']
        category = row['CATEGORY']

        simplified_tokens = simplify(text)
        if simplified_tokens:
            tokens.append(simplified_tokens)
            categories.append(category)

    return tokens, categories


def build_dictionary(tokens):
    vocab = set()
    for toks in tokens:
        for tk in toks:
            lemma = tk.lemma
            vocab.add(lemma)
            synsets = wn.synsets(lemma)
            if synsets:
                for s in synsets:
                    lemma_names = s.lemma_names()
                    lemma_names = [l for l in lemma_names if l.find('_') == -1]
                    [vocab.add(n) for n in lemma_names]

    return vocab


def vectorize(tokens, categories, vocab):
    count_vect_x = CountVectorizer(vocabulary=vocab)

    simplified_sents = []
    for toks in tokens:
        simplified_sents.append(' '.join([tk.lemma.lower() for tk in toks]))

    x = count_vect_x.transform(simplified_sents)

    return x, count_vect_x


def train_model(text_matrix, categories):
    bdt = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators=500,
        algorithm="SAMME")

    bdt.fit(text_matrix, categories)

    return bdt


def predict(model, count_vect_x, text):

    simplified_tokens = simplify(text)
    if not simplified_tokens:
        raise Exception('Could not simplify sentence.')

    vec_text = count_vect_x.transform([' '.join([tk.lemma.lower() for tk in simplified_tokens])])

    prediction = model.predict(vec_text)
    print(prediction)
    return prediction


def train_pipeline():
    df = read_data()
    tokens, cats = build_train_dataset(df)
    vocab = build_dictionary(tokens)

    text_matrix, count_vect_x = vectorize(tokens, cats, vocab)
    bdt = train_model(text_matrix, cats)
    return bdt, count_vect_x

def syntax_text(text):
    """Detects syntax in the text."""
    language_client = language.Client()

    # Instantiates a plain text document.
    document = language_client.document_from_text(text)

    # Detects syntax in the document. You can also analyze HTML with:
    #   document.doc_type == language.Document.HTML
    tokens = document.analyze_syntax()

    # for token in tokens:
    #     print('{}: {}'.format(token.part_of_speech, token.text_content))

    return tokens

def simplify(text):
    google_tokens = syntax_text(text)

    root = Token.build_tree_from_google_tokens(google_tokens)

    simple_dependents = []

    if root.lemma == 'be':
        if 'ATTR' in root: simple_dependents.append(root['ATTR'])
        if 'ACOMP' in root: simple_dependents.append(root['ACOMP'])
        if 'ADVMOD' in root: simple_dependents.append(root['ADVMOD'])

        if 'NSUBJ' in root:
            nsubj = root['NSUBJ']
            if 'NN' in nsubj: simple_dependents.extend(nsubj['NN'])
            if 'AMOD' in nsubj: simple_dependents.append(nsubj['AMOD'])
            simple_dependents.append(nsubj)
    elif root.edge_index == 0 and root.part_of_speech == 'VERB':
        # This is a naive test for an imperative clause.
        if 'DOBJ' in root:
            dobj = root['DOBJ']
            if 'NN' in dobj: simple_dependents.extend(dobj['NN'])
            simple_dependents.append(dobj)

    elif root.part_of_speech == 'NOUN':
        # If the root is Noun, keep all the noun compound modifiers 'NN'
        if 'NN' in root: simple_dependents.extend(root['NN'])
        simple_dependents.append(root)

    print('simplified2', [tk.lemma for tk in simple_dependents])
    return simple_dependents

