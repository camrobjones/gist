"""
Extract key features from text data
"""

import spacy
nlp = spacy.load('en_core_web_sm')

s = "Sally hit Jenny because she was angry"
doc = nlp(s)
sent = list(doc.sents)[0]

[w.dep_ for w in sent]
tree_me(doc)
