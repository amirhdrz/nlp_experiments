from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np

# letters = ['A', 'A', 'G', 'G', 'G', 'G', 'C', 'C', 'C']
#
# sequences = dict()
#
# for tup in permutations(range(9)):
#     seq = ''.join([letters[i] for i in tup])
#     if seq in sequences:
#         sequences[seq] += 1
#     else:
#         sequences[seq] = 1


## TESTING OF ONLINE ALGORITHMS

from functools import wraps

class T(object):

    def __init__(self):
        self._a = None

    @property
    def a(self) -> str:
        return self._a

    @a.setter
    def a(self, v : str):
        self._a = v

def make_pretty(a=0):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(a)
            return func(*args, **kwargs)
        return wrapper
    return inner

def ord():
    print('hi')

@make_pretty()
def nice():
    print('nicely decorated')

@make_pretty(a=3)
def nice2():
    print("i'm so nice")


