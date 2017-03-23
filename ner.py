""" ner.py - Named Entity Recognition
"""

from typing import List
from enum import Enum, unique

import re
import data
from syntax import Token

_reg_course_coee = re.compile(r'[a-z]{3,4}[0-9]{3,4}((h?1?(f/s)?)|(y?1?y?))?')

_building_codes = data.building_codes()

@unique
class EntityType(Enum):
    COURSE_CODE = 'COURSE_CODE'
    BUILDING_CODE = 'BUILDING_CODE'


class Entity:

    def __init__(self, type: EntityType, *tokens):
        if type in EntityType:
            self._type = type
        else:
            raise ValueError('Unrecognized entity type ({})'.format(type))

        self._tokens = tokens

    @property
    def type(self):
        return self._type

    @property
    def tokens(self):
        return self._tokens

    def __str__(self):
        return str((self.type, self.tokens))

    def __repr__(self):
        return str((self.type, self.tokens))


class CourseCode(Entity):

    def __init__(self, *tokens):
        super().__init__(tokens)

def extract_entities(tokens: List[Token]):
    """ Returns a list of Entity objects
    :param tokens:
    :return:
    """
    # TODO: need a more robust implementation

    entities = []

    for tk in tokens:

        if tk.part_of_speech == 'NOUN' or tk.part_of_speech == 'X':

            if _reg_course_coee.match(tk.text_lower) is not None:
                entities.append(Entity(EntityType.COURSE_CODE, tk))

            # Have to turn these stuff into more advanced searches
            elif tk.text_content.upper() in _building_codes:
                entities.append(Entity(EntityType.BUILDING_CODE, tk))

    return entities


def _test():
    print('Entity Test')
    tokens = [Token(text) for text in "csc369,csc24020,ba,MC".split(',')]
    for tk in tokens:
        tk.part_of_speech = 'NOUN'
    entities = extract_entities(tokens)
    print(entities)

