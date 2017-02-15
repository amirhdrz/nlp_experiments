""" Frames and and their corresponding slot fillters
"""


# Each frame contains slots, each slot has a slot filler
# PARENT is represented through class inheritance.

from abc import ABC, abstractmethod

from functools import wraps

from syntax import Token
from entity import Entity
from typing import List
from nlp import AnnotatedText

from intent import Intent

class Frame(ABC):
    """
    Abstract Frame class
    """
    def __init__(self, tokens : List[Token], entities : List[Entity]):
        self._tokens = tokens
        self._entities = entities

def slot(ident):
    def inner(slot_function):
        @wraps(slot_function)
        def wrapper(*args, **kwargs):
            if vars(args[0])[ident]:
                return slot_function(*args, **kwargs)
            else:
                return None
        return wrapper
    return inner


class InteractionTopic(object):
    """
    An InteractionTopic is a (limited) representation
    of the conversation revolving around a single topic,
    represented as a signle intent.

    This class does not deal with chatbot-platform specifics
    like the user id.
    """

    def __init__(self, intent : Intent, frame : Frame):
        self._intent = intent
        self._frame = frame
        self._conversation = []
        self._expecting = []

    @property
    def intent(self) -> Intent:
        return self._intent

    @property
    def frame(self):
        return self._frame

    @property
    def conversation(self) -> List[AnnotatedText]:
        """
        The conversation history
        :return:
        """
        return self._conversation

    @property
    def expecting(self) -> List[Entity]:
        return self._expecting

    def add_user_input(self, annotated_text : AnnotatedText):
        self._conversation.append(annotated_text)

    def add_expected_entity(self, entity : Entity):
        self._expecting.append(entity)
