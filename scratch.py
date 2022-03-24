from collections import Counter
from nltk.corpus import wordnet

attrs = ['also_sees',
 'attributes',
 'causes',
 'definition',
 'entailments',
 'examples',
 'frame_ids',
 'hypernym_distances',
 'hypernym_paths',
 'hypernyms',
 'hyponyms',
 'in_region_domains',
 'in_topic_domains',
 'in_usage_domains',
 'instance_hypernyms',
 'instance_hyponyms',
 'lemma_names',
 'lemmas',
 'lexname',
 'max_depth',
 'member_holonyms',
 'member_meronyms',
 'min_depth',
 'name',
 'offset',
 'part_holonyms',
 'part_meronyms',
 'pos',
 'region_domains',
 'root_hypernyms',
 'similar_tos',
 'substance_holonyms',
 'substance_meronyms',
 'topic_domains',
 'usage_domains',
 'verb_groups'
]

[('galaxy.n.03', 20),
 ('canada.n.01', 20),
 ('south.n.01', 20),
 ('war.n.01', 19),
 ('myanmar.n.01', 19),
 ('spain.n.01', 19),
 ('syria.n.01', 19),
 ('thailand.n.01', 19),
 ('tunisia.n.01', 19),
 ('united_states.n.01', 19)
]

synset = wordnet.synset('run.v.01')

for key in attrs:
    attr = getattr(synset, key)()
    if attr:
        print(f"{key}: {attr}")

synsets = list(wordnet.all_synsets())

sample = synsets

attr_counter = Counter()
synset_dict = Counter()

for synset in sample:
    
    valid_attrs = []

    for attr in attrs:
        if getattr(synset, attr)():
            valid_attrs.append(attr)

    attr_counter.update(valid_attrs)
    synset_dict[synset.name()] = len(valid_attrs)

attr_counter.most_common()
synset_dict.most_common(50)


def get_attr_examples(attr, n=10):
    """Get synsets with a valid attr"""
    found = 0
    for synset in sample:
        if getattr(synset, attr)():
            print(synset.name())
            found += 1

        if found >= n:
            break


get_attr_examples("causes")


def get_all_attrs(attr):
    """Get a counter of all attribute values for attr."""
    vals = Counter()
    for synset in synsets:
        vals.update([getattr(synset, attr)()])
    return vals


pos = get_all_attrs("pos")


def get_attr_val_examples(attr, val, n=10):
    """Get n synsets with attr=val."""
    found = 0
    for synset in sample:
        if getattr(synset, attr)() == val:
            print(synset.name())
            found += 1

        if found >= n:
            break


get_attr_val_examples("pos", "s")



relations = [
"closure",
"common_hypernyms",
"jcn_similarity",
"lch_similarity",
"lowest_common_hypernyms",
"shortest_path_distance",
"tree",
 'wup_similarity']


items = list(range(36))


def draw():
    return random.sample(items, 4)


def run(n):
    c = Counter()
    for r in range(n):
        c.update(draw())
    return c

cnt = run(30)

