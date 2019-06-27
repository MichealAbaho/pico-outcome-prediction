# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 27/06/19 
# @Contact: michealabaho265@gmail.com
import spacy
from spacy.tokens import Doc

class customize_spacy:

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        #All tokens own a subsequent space character in this tokenize
        spaces = [True] * len(words)

        return Doc(self.vocab, words=words, spaces=spaces)
