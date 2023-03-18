"""gist views"""
import json
import time

from nltk.corpus import wordnet
from nltk import Tree
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
import spacy
from django_celery_results.models import TaskResult
from celery.result import AsyncResult

from gist import synspaces, tasks

"""
Constants
---------
"""

# nlp = spacy.load("en_core_web_sm")

# Create your views here.


"""
spacy
-----
"""


def get_spacy_data(token):
    """return spacy token data as dict"""
    token_data = {}
    token_data['orth'] = token.orth_
    token_data['idx'] = token.idx
    token_data['lemma'] = token.lemma_
    token_data['tag'] = token.tag_
    token_data['pos'] = token.pos_
    return token_data


"""
nltk
-----
Get syntax trees for sentences from nltk.
"""


def tok_format(node):
    """Format by which a token will be represented"""
    return f"{node.orth_}_{node.dep_}"


# Credit: https://stackoverflow.com/questions/36610179/how-to-get-the-dependency-tree-with-spacy
def to_nltk_tree(node):
    """Recursive function to create a tree from node and descendents"""
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
    else:
        return tok_format(node)


def tree_me(doc):
    """Create a tree from a spacy doc"""
    return [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]


# def get_tree(node):
#     """Recursive function to create a tree from node and descendents"""
#     if node.n_lefts + node.n_rights > 0:
#         return [[node.orth_], [get_tree(child) for child in node.children]]
#     else:
#         return node.orth_


def get_tree(node):
    """Recursive function to create a tree from node and descendents"""
    if node.n_lefts + node.n_rights > 0:
        return {'orth': node.orth_,
                'children': [get_tree(child) for child in node.children]}
    else:
        return {'orth': node.orth_, 'children': []}


"""
         is_ROOT
     _________|__________
    |              sentence_attr
    |                    |
this_nsubj             a_det

[is]
[this, sentence]


"""

# tr = tree(sent.root)


def get_syntax_tree(doc):
    trees = []
    for sent in doc.sents:
        trees.append(get_tree(sent.root))
    return trees


"""
Master functions
----------------
"""


def get_token_data(token):
    """Get all data from a spacy token"""
    token_data = get_spacy_data(token)
    wordnet_data = get_wordnet_data(token.orth_)
    token_data['wordnet'] = wordnet_data
    return token_data


"""
Views
-----
"""


def home(request):
    """gist homepage"""
    return render(request, 'gist/home.html')


"""
API Calls
---------
"""


def analyse(request):
    """Core API call: analyse text query"""

    # Return format
    doc_data = []

    # Retrieve input
    post = post = json.loads(request.body.decode('utf-8'))
    text = post.get('query', '')

    # Analyse Text
    doc = nlp(text)

    for sent in doc.sents:
        sent_data = []

        for token in sent:
            token_data = get_token_data(token)
            sent_data.append(token_data)

        doc_data.append(sent_data)

    # Tree data
    tree_data = get_syntax_tree(doc)

    data = {'doc': doc_data, 'tree': tree_data}

    return JsonResponse(data)


"""
Wordnet
-------

"""


def pos_readable(pos):
    """Return readable label for pos."""
    pos_dict = {"a": "Adjective",
                "r": "Adverb",
                "s": "Adjective",  # TODO: why s != a?
                "n": "Noun",
                "v": "Verb"}
    return pos_dict.get(pos, "")


def get_ancestors(synset):
    """Get (one set) of ancestors, back to a node with no parents"""
    # TODO: method for finding canonical parent node
    ancestors = []

    for i in range(100):  # Limit in case circularities
        hyponyms = synset.hypernyms()
        if not hyponyms:
            break
        parent = hyponyms[0]
        ancestors = [parent.name()] + ancestors
        synset = parent

    return ancestors


def get_synset_data(synset):
    """Extract data for a synset key."""
    # synset = wordnet.synset(synset_key)
    hypernyms = synset.hypernyms()
    hyponyms = synset.hyponyms()

    hypernym_data = [h.name() for h in hypernyms]
    hyponym_data = [h.name() for h in hyponyms]

    ancestors = get_ancestors(synset)

    data = {'hypernyms': hypernym_data,
            'hyponyms': hyponym_data,
            'name': synset.name(),
            'definition': synset.definition(),
            'examples': synset.examples(),
            'lemma': synset.lemma_names()[0],
            'lemmas': synset.lemma_names(),
            'pos': pos_readable(synset.pos()),
            'attributes': [s.name() for s in synset.attributes()],
            'causes': [s.name() for s in synset.causes()],
            'entailments': [s.name() for s in synset.entailments()],
            'similar_tos': [s.name() for s in synset.similar_tos()],
            'also_sees': [s.name() for s in synset.also_sees()],
            'topic_domains': [s.name() for s in synset.topic_domains()],
            'in_topic_domains': [s.name() for s in synset.in_topic_domains()],
            'usage_domains': [s.name() for s in synset.usage_domains()],
            'in_usage_domains': [s.name() for s in synset.in_usage_domains()],
            'region_domains': [s.name() for s in synset.region_domains()],
            'in_region_domains': [s.name() for s in
                                  synset.in_region_domains()],
            'member_holonyms': [s.name() for s in synset.member_holonyms()],
            'member_meronyms': [s.name() for s in synset.member_meronyms()],
            'part_holonyms': [s.name() for s in synset.part_holonyms()],
            'part_meronyms': [s.name() for s in synset.part_meronyms()],
            'instance_hypernyms': [s.name() for s in
                                   synset.instance_hypernyms()],
            'instance_hyponyms': [s.name() for s in
                                  synset.instance_hyponyms()],
            'substance_holonyms': [s.name() for s in
                                   synset.substance_holonyms()],
            'substance_meronyms': [s.name() for s in
                                   synset.substance_meronyms()],
            'max_depth': synset.max_depth(),
            'min_depth': synset.min_depth(),
            'ancestors': ancestors
            }

    return data


def wordnet_home(request):
    return render(request, 'gist/wordnet.html')


def synset_data(request):
    synset_key = request.GET.get('synset_key')
    print(synset_key)

    synset = wordnet.synset(synset_key)
    data = get_synset_data(synset)

    return HttpResponse(json.dumps(data),
                        content_type='application/json')


"""Search"""


def get_wordnet_data(lemma):
    """Search for a lemma in wordnet.

    Args:
        lemma (str): String to be searched

    Returns:
        results: List of dict of info about matching synsets
    """
    synsets = wordnet.synsets(lemma)
    synset_data = [get_synset_data(synset) for synset in synsets]
    return synset_data


def search_lemma(request):
    """Return JSON obj with info about lemma.

    Args:
        request (HttpRequest)
    """
    lemma = request.GET.get("lemma")
    print(lemma)
    data = get_wordnet_data(lemma)
    return HttpResponse(json.dumps(data),
                        content_type='application/json')


"""
Synspaces
"""

def get_embeddings_keys():
    """TODO: Store these somewhere special and walk to find them."""
    return ["glove_6B", "glove_42B", "glove_840B",
        "glove_6B_wordnet", "glove_42B_wordnet", "glove_840B_wordnet",
        "bert_l0_wordnet", "bert_l6_wordnet",
        "bert_l11_wordnet", "bert_l12_wordnet",]

def synspaces_home(request):
    """Return synspace home view."""
    embeddings_keys = get_embeddings_keys()

    context = {
        "embeddings_keys": embeddings_keys
    }

    return render(request, 'gist/synspaces.html', context)

def synspaces_search(request):
    """Return JSON obj with info about lemma.

    Args:
        request (HttpRequest)
    """
    queries = request.GET.get("queries[]")
    print(queries)
    embeddings_key = request.GET.get("embeddings_key", "bert_l11_wordnet")
    n = min(int(request.GET.get("n", 20)), 200)
    dimred = request.GET.get("dimred", "pca")
    metric = request.GET.get("metric", "cosine")

    # task = tasks.get_synspace.delay(
    #     queries=queries,
    #     embeddings_key=embeddings_key,
    #     n=n,
    #     dimred=dimred,
    #     metric=metric)
    data = synspaces.get_synspace(queries=queries, embeddings_key=embeddings_key, n=n, dimred=dimred, metric=metric)

    # print(task)
    # data = {"success": True, "task_id": task.id}
    return HttpResponse(json.dumps(data),
                        content_type='application/json')


def get_progress(request, task_id):
    print(task_id)
    result = TaskResult.objects.get(task_id=task_id)
    status = response.status

    if (status == "SUCCESS"):
        result = TaskResult.objects.get(task_id=task_id)
        response_data = {
            'status': result.status,
            'details': result.result
        }
    else:
        response_data = {
            'state': status,
            'details': result.result,
        }
    return HttpResponse(
        json.dumps(response_data),
        content_type='application/json'
    )
