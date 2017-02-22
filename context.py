""" context.py - Defines functions and classes that deal with user's context.
Classes and methods defined here should act mostly as dumb containers
and not worry about implementations of the other modules.
"""

from typing import List

from entity_recognition import Entity, EntityType
from intent import Intent
from nlp import AnnotatedText
import frame  # used for type hinting


class Context:

    def __init__(self):
        self._current_topic = None

    @property
    def current_topic(self) -> 'InteractionTopic':
        return self._current_topic

    @current_topic.setter
    def current_topic(self, value: 'InteractionTopic'):
        self._current_topic = value

    @current_topic.deleter
    def current_topic(self):
        self._current_topic = None


class InteractionTopic:
    """
    An InteractionTopic is a (limited) representation
    of the conversation revolving around a single topic,
    represented as a single intent.

    This class does not deal with chatbot-platform specifics,
    like the user id.

    Note that this class should always contain the first conversation
    that instantiated it.
    """

    def __init__(self, intent: Intent, frame: 'frame.Frame', first_conversation: AnnotatedText):
        self._intent = intent
        self._frame = frame
        self._conversation = [first_conversation]
        self._expecting_entity = []

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
        """
        return self._conversation

    @property
    def expecting_entity(self) -> List[EntityType]:
        return self._expecting_entity

    def add_conversation(self, annotated_text: AnnotatedText):
        self._conversation.append(annotated_text)

    def add_expected_entity(self, entity_type: EntityType):
        self._expecting_entity.append(entity_type)

    def remove_expected_entity(self, entity_type: EntityType):
        self._expecting_entity.remove(entity_type)
