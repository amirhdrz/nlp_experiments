""" dialogue.py - Everything related to the dialogue
This file contains platform-dependent code and is the main layer
between the user and the rest of the chatbot.
"""

import nlp
from context import InteractionTopic, Context
from frame_models import create_frame, intent_to_slot


def process_message(intent_classifier, bow_vectorizer, user_context: Context, text: str):

    annotated_text = nlp.AnnotatedText(text)

    # should look at the top 3 suggestions
    intent = nlp.predict(intent_classifier, bow_vectorizer, annotated_text.root_token)

    entities = annotated_text.entities

    cur_topic = user_context.current_topic

    if cur_topic:
        cur_topic.add_conversation(annotated_text)
        # searching for the expected entity
        for entity in entities:
            if entity.type in cur_topic.expecting_entity:
                # if found, removes the expected entity from the topic
                cur_topic.remove_expected_entity(entity.type)

                # calls the requested slot
                fs_tuple = intent_to_slot(cur_topic.intent, cur_topic.frame)

                # DEBUG codee
                print(fs_tuple.slot.user_message())
                return

        # user probably did not respond with expected entity
        # TODO: find a better way to handle this
        print("I'm confused")
    else:
        # There is no previous topic
        fs_tuple = create_frame(intent, user_context)
        user_context.current_topic = InteractionTopic(intent, fs_tuple.frame, annotated_text)

        # DEBUG code
        print(fs_tuple.slot.user_message())




