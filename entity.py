""" Entity extraction.

"""

import re
import data
from syntax import Token

_reg_course_coee = re.compile(r'[a-z]{3,4}[0-9]{3,4}((h?1?(f/s)?)|(y?1?y?))?')

_building_codes = data.building_codes()


class Entity:

    _TYPES = ('COURSE_CODE',
              'BUILDING_CODE')

    def __init__(self, entity_type, *tokens):
        if entity_type in self._TYPES:
            self.type = entity_type
        else:
            raise Exception('Unrecognized entity type ({})'.format(entity_type))

        self.tokens = tokens

    def get_type(self):
        return self.type

    def get_tokens(self):
        return self.tokens

    def __str__(self):
        return str((self.type, self.tokens))

    def __repr__(self):
        return str((self.type, self.tokens))


def extract_entities(tokens):
    """ Returns a list of Entity objects
    :param tokens:
    :return:
    """
    # TODO: need a more robust implementation

    entities = []

    for tk in tokens:

        if tk.part_of_speech == 'NOUN':

            if _reg_course_coee.match(tk.text_lower) is not None:
                entities.append(Entity('COURSE_CODE', tk))

            # Have to turn these stuff into more advanced searches
            elif tk.text_content.upper() in _building_codes:
                entities.append(Entity('BUILDING_CODE', tk))

    return entities


def _test():
    print('Entity Test')
    tokens = [Token(text) for text in "csc369,csc24020,ba,MC".split(',')]
    for tk in tokens:
        tk.part_of_speech = 'NOUN'
    entities = extract_entities(tokens)
    print(entities)

