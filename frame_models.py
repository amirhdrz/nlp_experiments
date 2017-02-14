from typing import List
from syntax import Token
from entity import Entity
from frame import *

#TODO: each frame needs to be given the entities and the tokens

# Each slot filler should know how to fill the its property

class CourseInfo(Frame):

    def __init__(self, tokens, entities):
        super().__init__(tokens, entities)
        self._coursecode = None
        self._data = None

    @property
    def coursecode(self):
        if self._coursecode:
            return self._coursecode
        else:
            #TODO: get the course code from the user
            return None

    @property
    def data(self):
        if self._data:
            return self._data

        elif self.coursecode:
            # DO API CALL
            # self.data = data returned
            return self._data

        else:
            return None


class Building(Frame):

    def __init__(self, tokens, entities):
        super().__init__(tokens, entities)
        self._id = None
        self._data = None

    @property
    def id(self):
        if self._id:
            return self._id
        else:
            #TODO: get the building id from the user
            pass

    @property
    def data(self):
        if self._data:
            return self._data




