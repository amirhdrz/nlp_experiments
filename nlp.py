"""
bdt,cvx,predictions,cats,sents,text_matrix,misindex,misdf=train_pipeline(verbose_output=True)

IMPORTANT NOTES:
 - This file contains private API keys, don't share publicly.
"""

import time
import concurrent.futures
from typing import List

from google.cloud import language

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn

from syntax import *
from utils import *
from entity import Entity

# Google API key
api_key = "AIzaSyDPZceqCgLVGytRa14EOvYfcYarjfqMLm0"

_DEBUG = True

if _DEBUG:
    pd.set_option('display.width', 140)


class AnnotatedText(object):

    def __init__(self, text : str):
        if not text:
            raise ValueError("text should be set.")
        self._text = input
        self._tokens = None
        self._simplified_tokens = None
        self._entities = None

    @property
    def text(self):
        return self._text

    @property
    def tokens(self) -> List[Token]:
        return self._tokens

    @property
    def simplified_tokens(self) -> List[Token]:
        return self._simplified_tokens

    @property
    def entities(self) -> List[Entity]:
        return self._entities


# TODO: the categories in the CSV file should be checked against the Intent enum
def read_data(data_file):
    """
    Reads csv data_file and removes all rows that don't have their CATEGORY set.
    :param data_file: filename
    :return: Pandas dataframe
    """
    df = pd.read_csv(data_file + '.csv')
    return df[pd.notnull(df['CATEGORY'])]


def simplify_parallel(dataframe):
    tokens = []
    categories = []
    num_rows = len(dataframe)
    progress_count = 1

    future_to_ind = {}

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:

        # Scheduling the function call
        for i, row in dataframe.iterrows():
            fs = executor.submit(syntax_text, row['TEXT'])
            future_to_ind[fs] = i

        for future in concurrent.futures.as_completed(future_to_ind):

            print('Progress: {}/{}'.format(progress_count, num_rows), end="\r")
            progress_count += 1

            index = future_to_ind[future]
            try:
                # After getting the google tokens, it runs them through
                # the simplify function
                google_tokens = future.result()
                root_token, _ = Token.build_tree_from_google_tokens(google_tokens)
                simplified_tokens = simplify(root_token)

                tokens.append(simplified_tokens)
                categories.append(dataframe.ix[index]['CATEGORY'])
            except Exception as exc:
                print('exception at %d' % index, exc)


    print()
    print("%.2f seconds" % (time.time() - start_time))

    return tokens, categories


def simplify_dataframe(dataframe):
    tokens = []
    categories = []
    num_rows = len(dataframe)
    i = 1
    for line_num, row in dataframe.iterrows():

        print('Progress: {}/{}'.format(i, num_rows), end="\r")
        i += 1

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
    # bdt = AdaBoostClassifier(
    #     DecisionTreeClassifier(max_depth=3),
    #     n_estimators=500,
    #     algorithm="SAMME")

    bdt = RandomForestClassifier(n_estimators=100, max_depth=8)

    bdt.fit(text_matrix, categories)

    return bdt


def predict(model, count_vectorizer, text):

    simplified_tokens = simplify(text)
    if not simplified_tokens:
        raise Exception('Could not simplify sentence.')

    vec_text = count_vectorizer.transform([token_to_sent(simplified_tokens)])

    prediction = model.predict(vec_text)
    print(prediction)
    return prediction


def train_pipeline(train_file='sample_questions', debug_tokens=None, debug_cats=None, verbose_output=False):
    # Reads train_file csv
    df = read_data(train_file)

    if not (debug_tokens or debug_cats):

        # Uses Google dependency parser to simplify the sentences
        print('Simplifying text ...')
        document_tokens, cats = simplify_parallel(df)
    else:
        document_tokens, cats = debug_tokens, debug_cats
    cats = np.array(cats)

    # Builds vocabulary from the training data_files
    vocab = build_dictionary(document_tokens)

    # Builds a vectorizer to convert sentences to BOW model
    # and transforms all the training data_files into a BOW matrix
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
    document_tokens, cats = simplify_parallel(df)
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


def simplify(root_token : Token, verbose=False):
    """
    Simplifies google tokens to few descriptive tokens that try to preserve the intent.
    :param root_token: head of the syntax tree
    :return: syntax.Token
    """
    # google_tokens = syntax_text(text)
    # root_token, _ = Token.build_tree_from_google_tokens(google_tokens)

    simple_dependents = []

    if root_token.lemma == 'be':

        # TODO: should we look at the nsubj?

        if verbose: print('root (be) at index {}'.format(root_token.edge_index))

        if 'ATTR' in root_token:
            if verbose: print('attr ({}) attached to root'.format(root_token['ATTR']))
            simple_dependents.append(root_token['ATTR'])

        if 'ACOMP' in root_token:
            if verbose: print('acomp ({}) attached to root'.format(root_token['ACOMP']))
            simple_dependents.append(root_token['ACOMP'])

        if 'ADVMOD' in root_token:
            if verbose: print('advmod ({}) attached to root'.format(root_token['ADVMOD']))
            simple_dependents.extend(root_token['ADVMOD'])

        if 'NSUBJ' in root_token:
            nsubj = root_token['NSUBJ']

            if verbose: print('nsubj ({}) attached to root'.format(nsubj))

            if 'NN' in nsubj:
                if verbose: print('nn(s) ({}) attached to nsubj'.format(nsubj['NN']))
                simple_dependents.extend(nsubj['NN'])

            if 'AMOD' in nsubj:
                if verbose: print('amod ({}) attached to nsubj'.format(nsubj['AMOD']))
                simple_dependents.append(nsubj['AMOD'])

            simple_dependents.append(nsubj)


    elif root_token.edge_index == 0 and root_token.part_of_speech == 'VERB':
        # This is a naive test for an imperative clause.

        if verbose: print('text begins with root verb ({})'.format(root_token))

        if 'DOBJ' in root_token:
            dobj = root_token['DOBJ']
            if verbose: print('dobj ({}) attached to root'.format(dobj))

            if 'NN' in dobj:
                if verbose: print('dobj ({}) has NN(s) attached to it'.format(dobj))
                simple_dependents.extend(dobj['NN'])

            simple_dependents.append(dobj)

            if verbose: print('dobj ({}) attached to root'.format(dobj))

    elif root_token.part_of_speech == 'VERB':
        # TODO: this is very experimental

        if verbose: print('root ({}) is verb'.format(root_token))
        simple_dependents.append(root_token)

        if 'DOBJ' in root_token:
            dobj = root_token['DOBJ']
            if verbose: print('dobj ({}) attached to root'.format(dobj))

            if 'NN' in dobj:
                if verbose: print('dobj ({}) has NN(s) attached to it'.format(dobj))
                simple_dependents.extend(dobj['NN'])

            simple_dependents.append(dobj)

        if 'ADVMOD' in root_token:
            advmods = root_token['ADVMOD']
            simple_dependents.extend(advmods)
            if verbose: print('advmod(s) ({}) attached to root'.format(advmods))

        if 'XCOMP' in root_token:
            xcomp = root_token['XCOMP']
            simple_dependents.append(xcomp)
            if verbose: print('xcomp ({}) attached to root'.format(xcomp))



    elif root_token.part_of_speech == 'NOUN':
        # If the root is Noun, keep all the noun compound modifiers 'NN'

        if verbose: print('root ({}) is noun'.format(root_token.lemma))

        if 'NN' in root_token:
            if verbose: print('nn(s) ({}) attached to root'.format(root_token['NN']))
            simple_dependents.extend(root_token['NN'])

        simple_dependents.append(root_token)


    # print('simplified2', [tk.lemma for tk in simple_dependents])
    return simple_dependents


def debug_simplify(text, model=None, vectorizer=None, prob=True):
    print_y('input: ' + text)
    tokens = simplify(text, verbose=True)

    print_y('----')
    print_g('simplified: ' + str(tokens))
    if model and vectorizer:
        bow_vec = vectorizer.transform([token_to_sent(tokens)])
        if prob:

            def print_(ind):
                print_g('prediction : {}'.format(model.classes_[ind]))
                print_g('confidence : {0:.4g}'.format(probs[ind]))

            probs = model.predict_proba(bow_vec)[0]  # Only one sentence is passed
            sorted_inds = np.argsort(probs)
            print_(sorted_inds[-1])
            print_(sorted_inds[-2])
            print_(sorted_inds[-3])
        else:
            prediction = model.predict(bow_vec)
            print_g('prediction: ' + str(prediction))


    return tokens



def debug_simplify_file(data_file='sample_questions', model=None, vectorizer=None):
    df = read_data(data_file)

    for i, row in df.iterrows():
        cat = row['CATEGORY']
        text = row['TEXT']

        print_y(str(i) + ' )')
        debug_simplify(text, model=model, vectorizer=vectorizer)
        print_g('Ground: ' + cat)

        print()

