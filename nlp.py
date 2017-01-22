"""
bdt,vcx,predictions,cats,sents,text_matrix,misindex,misdf=train_pipeline(verbose_output=True)

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
from utils import *

# Google Knowledge Graph API key
api_key = "AIzaSyDPZceqCgLVGytRa14EOvYfcYarjfqMLm0"

_DEBUG = True

if _DEBUG: pd.set_option('display.width', 140)

def read_data(data_file):
    """
    Reads csv data_file and removes all rows that don't have their CATEGORY set.
    :param data_file: filename
    :return: Pandas dataframe
    """
    df = pd.read_csv(data_file + '.csv')
    return df[pd.notnull(df['CATEGORY'])]


def simplify_dataframe(dataframe):
    tokens = []
    categories = []
    num_rows = len(dataframe)
    for i, row in dataframe.iterrows():

        print('Progress: {}/{}'.format(i+1, num_rows), end="\r")

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


def build_count_vectorizer(tokens, categories, vocab):
    count_vect_x = CountVectorizer(vocabulary=vocab)

    simplified_sents = []
    for toks in tokens:
        simplified_sents.append(token_to_sent(toks))

    return count_vect_x


def token_to_sent(tokens):
    return ' '.join([tk.lemma.lower() for tk in tokens])


def document_tokens_to_sents(tokens):
    sents = []
    for toks in tokens:
        sents.append(token_to_sent(toks))
    return sents


def train_model(text_matrix, categories):
    bdt = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=3),
        n_estimators=500,
        algorithm="SAMME")

    bdt.fit(text_matrix, categories)

    return bdt


def predict(model, count_vectorizer, text):

    simplified_tokens = simplify(text)
    if not simplified_tokens:
        raise Exception('Could not simplify sentence.')

    vec_text = [token_to_sent(simplified_tokens)]

    prediction = model.predict(vec_text)
    print(prediction)
    return prediction


def train_pipeline(train_file='sample_questions', verbose_output=False):
    # Reads train_file csv
    df = read_data(train_file)

    # Uses Google dependency parser to simplify the sentences
    print('Simplifying text ...')
    document_tokens, cats = simplify_dataframe(df)
    cats = np.array(cats)

    # Builds vocabulary from the training data
    vocab = build_dictionary(document_tokens)

    # Builds a vectorizer to convert sentences to BOW model
    # and transforms all the training data into a BOW matrix
    vectorizer = build_count_vectorizer(document_tokens, cats, vocab)
    sents = document_tokens_to_sents(document_tokens)
    text_matrix = vectorizer.transform(sents)

    # Trains a classifier
    print('Training model ...')
    bdt = train_model(text_matrix, cats)

    # Tests the classifier on the training set
    print('Testing model ...')
    predictions = bdt.predict(text_matrix)
    true_pos = predictions == cats
    print('Accuracy', round(np.sum(true_pos) / len(predictions), 2))

    if verbose_output:
        misclass_index = np.nonzero(true_pos == False)[0]
        misclass_df = df.iloc[misclass_index]['TEXT'].copy().to_frame('Text')
        misclass_df['Prediction'] = pd.Series(predictions[misclass_index], index=misclass_df.index)
        misclass_df['Ground Truth'] = pd.Series(cats[misclass_index], index=misclass_df.index)


        return bdt, vectorizer, predictions, cats, sents, text_matrix, misclass_index, misclass_df
    else:
        return bdt, vectorizer


def training_set_test(model, count_vectorizer, test_file):
    df = read_data(test_file)
    document_tokens, cats = simplify_dataframe(df)
    test_sents = document_tokens_to_sents(document_tokens)
    text_matrix = count_vectorizer.transform(test_sents)

    predictions = model.predict(text_matrix)
    return predictions


def syntax_text(text):
    """
    Parses text using Google dependency parser and returns
    Google-typed tokens
    :param text: string
    :return: google.cloud.language.syntax.Token
    """
    language_client = language.Client()
    document = language_client.document_from_text(text)
    tokens = document.analyze_syntax()
    return tokens


def simplify(text, verbose=False):
    """
    Simplifies string text to few descriptive tokens that try to preserve the intent.
    :param text: string
    :return: syntax.Token
    """
    google_tokens = syntax_text(text)

    root = Token.build_tree_from_google_tokens(google_tokens)

    simple_dependents = []

    if root.lemma == 'be':

        if verbose: print('root (be) at index {}'.format(root.edge_index))

        if 'ATTR' in root:
            if verbose: print('attr ({}) attached to root'.format(root['ATTR'].lemma))
            simple_dependents.append(root['ATTR'])

        if 'ACOMP' in root:
            if verbose: print('acomp ({}) attached to root'.format(root['ACOMP'].lemma))
            simple_dependents.append(root['ACOMP'])

        if 'ADVMOD' in root:
            if verbose: print('advmod ({}) attached to root'.format(root['ADVMOD'].lemma))
            simple_dependents.append(root['ADVMOD'])

        if 'NSUBJ' in root:
            nsubj = root['NSUBJ']

            if verbose: print('nsubj ({}) attached to root'.format(nsubj.lemma))

            if 'NN' in nsubj:
                if verbose: print('nn(s) ({}) attached to nsubj'.format(token_to_sent(nsubj['NN'])))
                simple_dependents.extend(nsubj['NN'])

            if 'AMOD' in nsubj:
                if verbose: print('amod attached to nsubj')
                simple_dependents.append(nsubj['AMOD'])

            simple_dependents.append(nsubj)


    elif root.edge_index == 0 and root.part_of_speech == 'VERB':
        # This is a naive test for an imperative clause.

        if verbose: print('text begins with verb ({})'.format(root.lemma))

        if 'DOBJ' in root:
            dobj = root['DOBJ']
            if verbose: print('dobj ({}) attached to root'.format(dobj.lemma))

            if 'NN' in dobj:
                if verbose: print('dobj ({}) has NN(s) attached to it'.format(dobj.lemma))
                simple_dependents.extend(dobj['NN'])

            simple_dependents.append(dobj)

            if verbose: print('dobj ({}) attached to root'.format(dobj.lemma))

    elif root.part_of_speech == 'VERB':
        # TODO: this is very experimental

        if verbose: print('root ({}) is verb'.format(root.lemma))

        if 'DOBJ' in root:
            dobj = root['DOBJ']
            if verbose: print('dobj ({}) attached to root'.format(dobj.lemma))

            if 'NN' in dobj:
                if verbose: print('dobj ({}) has NN(s) attached to it'.format(dobj.lemma))
                simple_dependents.extend(dobj['NN'])

            simple_dependents.append(dobj)

        simple_dependents.append(root)


    elif root.part_of_speech == 'NOUN':
        # If the root is Noun, keep all the noun compound modifiers 'NN'

        if verbose: print('root ({}) is noun'.format(root.lemma))

        if 'NN' in root:
            if verbose: print('nn(s) ({}) attached to root'.format([nn.lemma for nn in root['NN']]))
            simple_dependents.extend(root['NN'])

        simple_dependents.append(root)


    # print('simplified2', [tk.lemma for tk in simple_dependents])
    return simple_dependents

def debug_simplify(text):
    tokens = simplify(text, verbose=True)
    print()
    print_y('     input: ' + text)
    print('simplified:', token_to_sent(tokens))
    print()
    return tokens


def debug_simplify_file(data_file='sample_questions', model=None, cvx=None):
    df = read_data(data_file)

    for i, row in df.iterrows():
        cat = row['CATEGORY']
        text = row['TEXT']

        tokens = simplify(text)
        simplified_sent = token_to_sent(tokens)

        print_y('Row ' + str(i + 2))
        print(text, cat)

        if not (model or cvx):
            prediction = predict(model, cvx, text)
            print_g(simplified_sent, prediction)
        else:
            print_g(simplified_sent)
        print()
