from itertools import permutations

letters = ['A', 'A', 'G', 'G', 'G', 'G', 'C', 'C', 'C']

sequences = dict()

for tup in permutations(range(9)):
    seq = ''.join([letters[i] for i in tup])
    if seq in sequences:
        sequences[seq] += 1
    else:
        sequences[seq] = 1

