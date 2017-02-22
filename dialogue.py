""" dialogue.py - Everything related to the dialogue
This file contains platform-dependent code and is the main layer
between the user and the rest of the chatbot.
"""

import nlp
from context import InteractionTopic, Context
from frame import Frame
from frame_models import create_frame, intent_to_slot
import utils

def process_message(intent_classifier, bow_vectorizer, user_context: Context, text: str):

    annotated_text = nlp.AnnotatedText(text)

    entities = annotated_text.entities

    cur_topic = user_context.current_topic

    if cur_topic:

        # searching for the expected entity
        for entity in entities:
            if entity.type in cur_topic.expecting_entity:
                # if expected entity found is found in user input,
                # removes the expected entity from the InteractionTopic
                cur_topic.remove_expected_entity(entity.type)

                # we're guessing here that what the user input is
                # related to the last topic
                cur_topic.add_conversation(annotated_text)

                # calls the requested slot
                fs_tuple = intent_to_slot(cur_topic.intent, cur_topic.frame)

                update_interaction(fs_tuple.slot, user_context)
                return

        # user probably did not respond with expected entity
        print("I'm confused (have current topic with no expected entity)")

    else:
        # There is no previous topic
        intent = nlp.predict(intent_classifier, bow_vectorizer, annotated_text.root_token)
        fs_tuple = create_frame(intent, user_context)
        user_context.current_topic = InteractionTopic(intent, fs_tuple.frame, annotated_text)
        update_interaction(fs_tuple.slot, user_context)


# TODO: need a better name for this function
def update_interaction(slot: Frame, context: Context):
    message = slot.user_message()
    if message:
        # TODO: need a better way to signal current slot cannot generate response
        print(message)

        # when a message is sent, we can assume that the current
        # topic is done, and can be deleted.
        del context.current_topic

def send_message(message: str):
    # DEBUG implementation
    print(message)
