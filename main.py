""" main.py - so far used for emulating FB messenger
"""

import os
import argparse
import pickle
from functools import partial

from nlp import *
from dialogue import *
from context import *
import res.strings as rs

def main(model, vectorizer):

    # Train a model
    if not (model or vectorizer):
        model, vectorizer = train_pipeline()

    print("Chatbot interactive environment")
    print("Type 'exit' to exit the program")
    print()

    # Creating a new context for each session
    user_context = Context()

    for user_message in iter(partial(input, '\033[92m\u03bb\033[0m '), 'exit'):

        # remove leading and trailing whitespace
        user_message = user_message.strip()

        if not user_message:
            continue

        try:
            process_message(model, vectorizer, user_context, user_message)
        except NLPError:
            print(rs.nlp_fail_message(user_message))


def train_and_save_model():
    print('Training new model')
    print('Please wait...')
    # Train new file if not found
    model, vectorizer = nlp.train_pipeline()
    if not os.path.exists('./models'):
        os.makedirs('models')
    clf_file = open('models/classifier', 'wb')
    vcx_file = open('models/vectorizer', 'wb')
    pickle.dump(model, clf_file)
    pickle.dump(vectorizer, vcx_file)
    clf_file.close()
    vcx_file.close()
    return model, vectorizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Debug environment for the chatbot')
    parser.add_argument('--force-train', '-f', action='store_true', help='retrains a new model and replaces the old one')

    args = parser.parse_args()

    if args.force_train:
        model, vectorizer = train_and_save_model()
    else:
        try:
            clf_file = open('models/classifier', 'rb')
            vcx_file = open('models/vectorizer', 'rb')
            model = pickle.load(clf_file)
            vectorizer = pickle.load(vcx_file)
            clf_file.close()
            vcx_file.close()
        except FileNotFoundError:
            model, vectorizer = train_and_save_model()

    main(model, vectorizer)
