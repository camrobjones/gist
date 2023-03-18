"""
Find similar words to query and plot
"""

from collections import Counter
from datetime import datetime
import re
import copy
import math

from nltk.corpus import brown, wordnet
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn import decomposition, manifold
import torchtext
from annoy import AnnoyIndex

from gist import twl

"""
Parameters
"""
MODELS = {}
EMBEDDINGS = {}

"""
Setup
"""


def build_wordlist(corpus="wordnet", lemmatize=0):
    """Create list of words for comparison."""
    if corpus == "brown":
        brown_words = brown.words()
        words = set([w.lower() for w in brown_words])

    if corpus == "scrabble":
        words = set(twl.iterator())

    if corpus == "wordnet":
        words =  set(wordnet.all_lemma_names())
        words =  [w for w in words if '_' not in str(w)]

    if lemmatize == "spacy":
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(" ".join(words))
        words = [w.lemma_ for w in doc]

    wordlist = list(words)
    word_df = pd.DataFrame(wordlist)
    word_df.to_csv(f"gist/data/wordlist_c-{corpus}_l-{lemmatize}.csv", index=False)
    return wordlist



def load_wordlist(corpus="wordnet", lemmatize=True):
    """load list of words for comparison."""
    try:
        word_df = pd.read_csv(f"gist/data/wordlist_{corpus}_{lemmatize}.csv")
        wordlist = word_df["0"].tolist()
    
    except FileNotFoundError:
        wordlist = build_wordlist(corpus=corpus, lemmatize=lemmatize)
    
    return np.array(wordlist)


"""
Notes:
- Glove POC before extend
- Assume wordnet until we need other wordlists
"""


class EmbeddingModel():
    """Word Embedding Model."""
    def __init__(self, name, version, dim):
        self.name = name
        self.version = version
        self.dim = dim
        self.load()

    def load(self):
        """Load Model"""
        raise NotImplementedError

    def embed(self, query):
        """Embed query."""
        raise NotImplementedError

class GloveEmbeddingModel(EmbeddingModel):
    """Glove Word Embedding Model."""
    def load(self):
        self.model = torchtext.vocab.GloVe(name=self.version, dim=self.dim, max_vectors=0)

    def embed(self, query):
        return self.model[query]


def load_model(name, version, dim):
    """Load model."""
    key = f"{name}_{version}"
    if key not in MODELS:
        model = GloveEmbeddingModel(name, version, dim)
        model.load()
        MODELS[key] = model

    return MODELS[key]


class Embeddings():
    """Matrix of word embeddings."""
    def __init__(self, model_name, model_version, model_dim):
        """Initialise Embeddings."""
        self.wordlist = load_wordlist()
        self.model = load_model(model_name, model_version, model_dim)
        self.create_filepath()

    def create_filepath(self):
        """Create a filepath to store the model index."""
        self.filepath = f"gist/data/embeddings/{self.model.name}_{self.model.version}_wordnet.annoy"

    def build(self, n_trees=100):
        """Build index for embeddings."""
        print("Building index...")
        self.index = AnnoyIndex(self.model.dim, 'angular')
        for i in tqdm(range(len(self.wordlist))):
            v = self.model.embed(wordlist[i])
            self.index.add_item(i, v)

        self.index.build(n_trees)
        self.index.save(self.filepath)

    def load(self):
        """Load Index."""
        self.index = AnnoyIndex(self.model.dim, "angular")
        self.index.load(self.filepath)

    def nn(self, query, n=20):
        """Get nearest neighbours to query."""
        v = self.model.embed(query)
        ids, dists = self.index.get_nns_by_vector(v, n, include_distances=True)
        words = [self.wordlist[i] for i in ids]
        vecs = [self.index.get_item_vector(i) for i in ids]

        return (words, dists, vecs)

def create_embeddings_tensor(wordlist, model_name, layer):
    """Create a matrix containing model embeddings for all words in wordlist."""

    embeddings = []

    # test cosine sense check
    for word in tqdm(wordlist):
        embedding = word_embedding(word, model_name, layer)
        embeddings.append(np.array(embedding))

    embeddings_matrix = np.array(embeddings)
    embeddings_tensor = torch.tensor(embeddings_matrix)  # TODO: avoid going via numpy

    return embeddings_tensor

def save_bert_embeddings(words, layer):
    bert, bert_tokenizer = get_bert()
    embeddings_tensor = create_embeddings_tensor(list(words), "bert", layer)

    # Save embeddings tensor
    embeddings_df = pd.DataFrame(embeddings_tensor)
    embeddings_df.to_csv(f"gist/data/bert_l{layer}.csv", index=False)

def save_glove_embeddings(wordlist, key):
    words = load_wordlist(wordlist)
    glove = load_glove(key)
    embeddings_tensor = torch.stack([glove[w] for w in words])

    # Save embeddings tensor
    embeddings_df = pd.DataFrame(embeddings_tensor)
    embeddings_df.to_csv(f"gist/data/embeddings/{key}_{wordlist}.csv", index=False)

# Build BERT word matrix

def get_bert():
    """Load BERT model."""
    if "bert" not in MODELS.keys():
        print("loading bert...")
        bert = BertModel.from_pretrained('bert-base-uncased',
                                             output_hidden_states=True,
                                             output_attentions=True)

        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        MODELS["bert"] = (bert, bert_tokenizer)

    return MODELS["bert"]

def get_embeddings(key="bert_l0"):
    """Load embeddings."""
    if key not in EMBEDDINGS:
        embeddings_df = pd.read_csv(f"gist/data/embeddings/{key}.csv")
        embeddings = torch.tensor(embeddings_df.to_numpy())
        EMBEDDINGS[key] = embeddings

    return EMBEDDINGS[key]

def word_embedding(queries, model="bert", layer=11):
    """Get the `model`'s embedding of `word` at `layer`"""
    model, tokenizer = get_bert()

    embeddings = []

    for query in queries:

        # t0 = datetime.now()
        word_enc = tokenizer(query, return_tensors="pt", padding=True)

        with torch.no_grad():
            output = model(**word_enc)
            hidden_states = output.hidden_states

        selected_layer = hidden_states[layer]
        word_vecs = selected_layer[:, 1:-1]
        embedding = torch.mean(word_vecs, dim=1)

        # t1 = datetime.now()
        # s = (t1 - t0).total_seconds()
        # print(f"{len(words)} in {s}")

        embeddings.append(embedding)

    embeddings = torch.stack(embeddings)
    mean_embedding = torch.mean(embeddings, dim=0)

    return mean_embedding



def word_embeddings(words, model="bert", batch_size = 1000):
    """Get the `model`'s embedding of `word` at `layer`"""
    model, tokenizer = get_bert()

    t0 = datetime.now()

    embeddings = []

    batches = math.ceil(len(words) / batch_size)

    for ix in tqdm(range(batches)):

        batch = words[ix * batch_size:(ix+1) * batch_size].tolist()

        word_enc = tokenizer(batch, return_tensors="pt", padding=True)

        with torch.no_grad():
            output = model(**word_enc)
            hidden_states = torch.stack(output.hidden_states)

        # selected_layer = hidden_states[layer]
        # word_vecs = selected_layer[:, 1:-1]
        embedding = torch.mean(hidden_states, dim=2)

        embeddings.append(embedding)

    # for layer in range(ems.shape[0]):
    #     layer_embeds = ems[layer]
    #     embeddings_matrix = np.array(layer_embeds)
    #     embeddings_tensor = torch.tensor(embeddings_matrix)  # TODO: avoid going via numpy
    #     embeddings_df = pd.DataFrame(embeddings_tensor)
    #     embeddings_df.to_csv(f"gist/data/bert_l{layer}_wordnet_batched.csv", index=False)

    t1 = datetime.now()
    s = (t1 - t0).total_seconds()
    print(f"{len(words)} in {s}")

    # mean_embedding = torch.mean(embedding, dim=1)

    return embedding

def most_similar(queries, embeddings, model, layer, wordlist, n=20, metric="cosine"):
    """Get the most similar n words to query."""

    query_embedding = word_embedding(queries, model)

    if metric == "cosine":
        distances = torch.cosine_similarity(query_embedding, embeddings)
        word_ids = torch.argsort(distances, descending=True)
    elif metric == "euclidean":
        distances = torch.norm(embeddings - query_embedding, dim=1)
        word_ids = torch.argsort(distances, descending=False)

    # Remove duplicates from query and results
    words = list(set(queries + wordlist[word_ids[:n]].tolist()))
    # embeddings = torch.stack([word_embedding(w, model) for w in words])
    result_embeddings = word_embeddings(np.array(words))[layer, :, :]
    result_embeddings = result_embeddings.double()

    if metric == "cosine":
        distances = torch.cosine_similarity(query_embedding, result_embeddings)
    elif metric == "euclidean":
        distances = torch.norm(result_embeddings - query_embedding, dim=1)

    distances = distances.tolist()

    # query_ids = torch.tensor([np.where(wordlist == q)[0].tolist()[0] for q in queries])
    
    # word_ids = torch.cat((query_ids, word_ids[~torch.isin(word_ids, query_ids)][:n]))

    # distances = [d.item() for d in distances[word_ids[:n]]]

    # words = 

    # result_embeddings = embeddings[word_ids[:n]]

    return (words, distances, result_embeddings)


def normalize_coords(coords, limits=(-9, 9)):
    """Normalize coord vals between min and max"""
    lim_range = limits[1] - limits[0]
    for dim in range(coords.shape[1]):
        v = coords[:, dim]   # foo[:, -1] for the last column
        coords[:, dim] = ((v - v.min()) / (v.max() - v.min()) * lim_range) - lim_range / 2

    return coords

def dimensionality_reduction(embeddings, dimensions=2, method="pca", normalize=True):
    """Create a reduced dimensional space representing embeddings."""
    # TODO: Add more techniques
    embeddings_array = np.array(embeddings)

    if method == "pca":
        pca = decomposition.PCA(n_components=dimensions)
        fit = pca.fit(embeddings_array)
        transform = pca.transform(embeddings_array)
    if method == "cosine_pca":
        pca = decomposition.KernelPCA(n_components=dimensions, kernel="cosine")
        fit = pca.fit(embeddings_array)
        transform = pca.transform(embeddings_array)

    if method == "fa":
        fa = decomposition.FactorAnalysis(n_components=dimensions)
        fit = fa.fit(embeddings_array)
        transform = fa.transform(embeddings_array)

    if method == "tsne":
        tsne = manifold.TSNE(n_components=dimensions, perplexity=3)
        transform = tsne.fit_transform(embeddings_array).astype(float)

    if method == "tsne_cosine":
        tsne = manifold.TSNE(n_components=dimensions, perplexity=3, metric="cosine")
        transform = tsne.fit_transform(embeddings_array).astype(float)

    if method == "mds":
        mds = manifold.MDS(n_components=dimensions)
        transform = mds.fit_transform(embeddings_array).astype(float)

    if method == "isomap":
        isomap = manifold.Isomap(n_components=dimensions)
        transform = isomap.fit_transform(embeddings_array).astype(float)

    if normalize:
        transform = normalize_coords(transform)

    return transform

def most_similar_glove(queries, n=20, key="glove_42B", metric="cosine"):
    """Get the most similar words in glove."""
    glove = load_glove(key=key)
    if key in ["glove_6B", "glove_42B", "glove_840B"]:
        embeddings = glove.vectors
        wordlist = [glove.itos[i] for i in range(len(glove.vectors))]
    else:
        embeddings = get_embeddings(key)
        wordlist = load_wordlist(key.split("_")[-1])

    vecs = []
    for query in queries:
        vecs.append(glove[query])
    vecs = torch.stack(vecs)
    vec = torch.mean(vecs, dim=0)

    if metric == "euclidean":
        dists = 1 / (0.3 * torch.norm(embeddings - vec, dim=1))
        q_dists = 1 / (0.3 * torch.norm(vecs - vec, dim=1))
        q_dists = torch.min(q_dists, torch.tensor([1]))  # deal with div/0
        lst = sorted(enumerate(dists.numpy()), key=lambda x: -x[1]) # sort by distance
    elif metric == "cosine":
        dists = torch.cosine_similarity(vec, embeddings)
        q_dists = torch.cosine_similarity(vec, vecs)
        lst = sorted(enumerate(dists.numpy()), key=lambda x: -x[1]) # sort by distance

    words = copy.deepcopy(queries)
    result_embeddings = [glove[w] for w in queries]
    distances = q_dists.tolist()

    for idx, difference in lst:
        w = wordlist[idx]
        if w not in words:
            words.append(w)
            distances.append(difference.item())
            result_embeddings.append(glove[w])

        if len(words) >= (n + len(queries)):
            break

    result_embeddings = torch.stack(result_embeddings).double()

    return (words, distances, result_embeddings)

def get_synspace(queries, model_name="bert", embeddings_key="glove_840B_wordnet", n=20,
    dimred="pca", metric="cosine"):
    """Get most similar words to query with 2d coordinates."""
    queries = queries.split(',')
    print(queries)
    if embeddings_key.startswith("glove"):
        embedding = Embeddings("glove", "840B", 300)
        embedding.load()
        words, similarities, embeddings = embedding.nn(queries[0], n=n)
    else:
        embeddings = get_embeddings(embeddings_key)
        wordlist = load_wordlist()
        layer = int(re.match("bert_l([0-9]+)", embeddings_key).groups()[0])
        words, similarities, embeddings = most_similar(
            queries, embeddings, model_name, layer, wordlist, n=n, metric=metric)
    coords = dimensionality_reduction(embeddings, method=dimred)
    result = []
    print(words)
    print(queries)
    for i in range(len(words)):
        color = "red" if words[i] in queries else "blue"
        word_data = {
                        "word": words[i],
                        "similarity": similarities[i],
                        "x": coords[i][0],
                        "y": coords[i][1],
                        "color": color
        }
        result.append(word_data)
    return result


