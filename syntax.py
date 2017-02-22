import re
from nltk.tokenize import RegexpTokenizer

# TODO: remove entity field from Token class and create a new Entity class
#       this is because entities can sometimes span over multiple tokens

class Token(object):

    def __init__(self, text_content):
        self.text_content = text_content
        self.text_lower = text_content.lower()
        self.lemma = None
        self.part_of_speech = None
        self.edge_index = None
        self.edge_label = None
        self.dependents = {}

    def add_child(self, token):
        if token.edge_label in ['PREP', 'NN', 'P', 'ADVMOD']:
            # Multiple PREP dependencies are possible
            if token.edge_label in self.dependents:
                self.dependents[token.edge_label] += [token]
            else:
                self.dependents[token.edge_label] = [token]

        elif token.edge_label in self.dependents:
            # TODO: this is a silly assumption for a lot of modifiers
            raise Exception(token.edge_label, 'Token already contains a child with given edge label.')

        else:
            self.dependents[token.edge_label] = token

    def __getitem__(self, edge_label):
        return self.dependents[edge_label]

    def __contains__(self, edge_label):
        return edge_label in self.dependents

    def __str__(self):
        return self.lemma

    def __repr__(self):
        # TODO: Need a better representation
        if self.lemma:
            return self.lemma
        else:
            return self.text_content

    @classmethod
    def from_google_token(cls, token):
        ins = cls(token.text_content)
        ins.lemma = token.lemma.lower()
        ins.part_of_speech = token.part_of_speech
        ins.edge_index = token.edge_index
        ins.edge_label = token.edge_label
        return ins

    @classmethod
    def build_tree_from_google_tokens(cls, google_tokens):
        """
        :param tokens: Google tokens
        :return: Root token with all of it's children
        """
        root_index = [tk.edge_label for tk in google_tokens].index('ROOT')
        edge_indices = [tk.edge_index for tk in google_tokens]
        tokens = [Token.from_google_token(tk) for tk in google_tokens]
        for i, edge_index in enumerate(edge_indices):
            if i != edge_index:
                tokens[edge_index].add_child(tokens[i])

        return tokens[root_index], tokens


## DEBUGGING
def _test(text=None):
    if text is not None:
        text = "What are the prerequisites for csc369?\n"


    tokenizer = RegexpTokenizer('[\w\d]+')
    word_tokens = tokenizer.tokenize(query)
    # TODO: how to tokenize 'u of t' and 'uoft'
    # TODO: use Bing Spell Check API
    tokens = [Token(tk) for tk in word_tokens]
    ser = RegexpEntityRecognizer()
    ser.process(tokens)


