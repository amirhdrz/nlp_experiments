""" frame.py - Frames and and their corresponding slot fillters

NOTE: Convention used with frame fillers and user_message functions,
is to return None explicitly.
"""

# Each frame contains slots, each slot has a slot filler
# PARENT is represented through class inheritance.

from abc import ABC, abstractmethod

import context  # used for type hinting
from entity_recognition import EntityType


class Frame(ABC):
    """
    Abstract Frame class
    """
    def __init__(self, context, parent_frame=None):
        self._context = context  # Should not be pickled
        self._parent = parent_frame
        self._frame_data = None

    @property
    def context(self) -> 'context.Context':
        return self._context

    @property
    def parent(self):
        return self._parent

    def frame_data(self):
        if not self._frame_data:
            self._frame_data = self._frame_data_filler()
        return self._frame_data

    def _frame_data_filler(self):
        """
        Returns data that fills this frame if empty.
        Subclasses should only return the filler data,
        and not set self._frame_data directly.
        """
        pass

    @abstractmethod
    def user_message(self):
        """
        User-oriented description of the frame.

        Current convention is that this function should return None,
        if it cannot generate the appropriate response.
        """
        pass


class LambdaFrame(Frame):

    def __init__(self, context, parent_frame,
                 key: str, default_message='{data}'):
        super().__init__(context, parent_frame=parent_frame)
        self.key = key
        self.default_message = default_message

    def _frame_data_filler(self):
        data = self.parent.frame_data()
        if data:
            return data[self.key]
        else: return None


    def user_message(self):
        data = self.frame_data()
        if data:
            return self.default_message.format(data=self.frame_data())
        else:
            return None

    @staticmethod
    def partial_constructor(context, parent_frame, parent_frame_data):
        def _constructor(key, default_message='{data}'):
            return LambdaFrame(context,
                               parent_frame,
                               key,
                               default_message=default_message)
        return _constructor


class UserFillingFrame(Frame):

    def __init__(self, context, parent_frame, prompt_message: str,
                 expected_entity: EntityType):
        super().__init__(context=context, parent_frame=parent_frame)
        self.prompt_message = prompt_message
        self.expected_entity = expected_entity

    def _frame_data_filler(self):

        # 1. Show user prompt message
        # TODO: this is only used for debugging
        print(self.prompt_message)

        # 2. Add expected entity to the context
        self.context.current_topic.add_expected_entity(self.expected_entity)
