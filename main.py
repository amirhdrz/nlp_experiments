""" main.py - so far used for emulating FB messenger
"""

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

    # Creating a new context for each session
    user_context = Context()

    for user_message in iter(partial(input, '\n\033[92m\u03bb\033[0m '), 'exit'):
        try:
            process_message(model, vectorizer, user_context, user_message)
        except NLPException:
            print(rs.nlp_fail(user_message))


if __name__ == "__main__":
    try:
        clf_file = open('debug_model/classifier', 'rb')
        vcx_file = open('debug_model/vectorizer', 'rb')
        model = pickle.load(clf_file)
        vectorizer = pickle.load(vcx_file)
        clf_file.close()
        vcx_file.close()
    except FileNotFoundError:

        # Train new file if not found
        model, vectorizer = nlp.train_pipeline()

        clf_file = open('debug_model/classifier', 'wb')
        vcx_file = open('debug_model/vectorizer', 'wb')
        pickle.dump(model, clf_file)
        pickle.dump(vectorizer, vcx_file)
        clf_file.close()
        vcx_file.close()

    main(model, vectorizer)
