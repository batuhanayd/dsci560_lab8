import argparse
import json
import warnings
from collections import Counter, namedtuple

import nltk
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score, silhouette_score)
from sklearn.metrics.pairwise import cosine_similarity

# downloading NLTK resources gere so the script can run directly
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
# hiding the warning
warnings.filterwarnings('ignore')

# getting the common english stopwords to remove them for tokenization.
STOP_WORDS = set(stopwords.words('english'))

# top number of words per cluster to inspect later on.
TOP_K_WORDS = 10

# we skip coherence if a cluster has fewer than 2 valid words.
MIN_WORDS_COHERENCE = 2
Tagged = namedtuple('Tagged', ['words'])


#----------------------------------------------------------------
# Top-words coherence
def top_words_coherence(model, labels, tagged, top_k=TOP_K_WORDS):
    # safety check in case the model somehow has no vocab.
    if not hasattr(model, "wv") or len(model.wv) == 0:
        return float("nan")
    n_c = int(labels.max()) + 1
    coh_list = []
    for c in range(n_c):
        idx = np.where(labels == c)[0]
        cnt = Counter()
        for i in idx:
            if i < len(tagged):
                cnt.update(tagged[i].words)
        top = [w for w, _ in cnt.most_common(top_k) if w in model.wv]
        if len(top) < MIN_WORDS_COHERENCE:
            continue
        vecs = np.array([model.wv[w] for w in top], dtype=np.float64)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs_n = vecs / norms
        sim = np.dot(vecs_n, vecs_n.T)
        triu = np.triu_indices(len(top), k=1)
        coh_list.append(float(np.mean(sim[triu])))
    # if no valid cluster had enough number of words, return nan
    if not coh_list:
        return float("nan")
    return float(np.mean(coh_list))

#----------------------------------------------------------------
#CLI arguments for choosing different configs
def parse_args():
    parser = argparse.ArgumentParser(description='Word2Vec + KMeans document embedding for Reddit posts')
    parser.add_argument('--w2v_dim', type=int, default=100)
    parser.add_argument('--min_count', type=int, default=2,)
    parser.add_argument('--k_bins', type=int, default=3)
    return parser.parse_args()

#----------------------------------------------------------------
# Tokenization
# Followed the url link from the instructions
def tokenize_post(text):
    sentences   = []
    flat_tokens = []
    for sent in sent_tokenize(text):
        tokens = [
            w for w in word_tokenize(sent.lower())
            if w.isalpha() and w not in STOP_WORDS and len(w) > 2
           ]
        if tokens:
            sentences.append(tokens)
            flat_tokens.extend(tokens)
    return sentences, flat_tokens


#----------------------------------------------------------------
# Doc vectorization
def post_to_vector(tokens, word_to_bin, k):
    """Convert a token list into a normalized K-dim bin-frequency vector."""
    counts  = np.zeros(k, dtype=float)
    matched = 0
    for word in tokens:
        if word in word_to_bin:
            counts[word_to_bin[word]] += 1
            matched += 1
    if len(tokens) > 0:
        counts /= len(tokens)
    return counts

#----------------------------------------------------------------
# Main
def main():
    args = parse_args()
    W2V_DIM   = args.w2v_dim
    MIN_COUNT = args.min_count
    K_BINS    = args.k_bins

    print(f"Parameters -> W2V_DIM={W2V_DIM}, min_count={MIN_COUNT}, K_BINS={K_BINS}\n")

    with open('reddit_merged.json', 'r', encoding='utf-8') as f:
        posts = json.load(f)
    print(f"Loaded {len(posts)} posts")
    post_tokens   = []
    all_sentences = []
    # extract the reddit posts
    # create all_sentences which is title + the text where text is content + ocr output.
    for post in posts:
        raw = (post.get('title') or '') + ' ' + (post.get('text') or '')
        sents, flat = tokenize_post(raw)
        all_sentences.extend(sents)
        post_tokens.append(flat)

    tagged = [Tagged(words=tokens) for tokens in post_tokens]
    print(f"Total sentences for Word2Vec training: {len(all_sentences)}")

    # Train the Word2Vec for CBOW
    model = Word2Vec( sentences=all_sentences, vector_size=W2V_DIM, window=5,
        min_count=MIN_COUNT,
        sg=0, #Using CBOW
        workers=4
    )

    print(f"Vocabulary size: {len(model.wv)} unique words")

    # Similarity demonstration for validation of the model
    test_pairs = [
        ('roommate', 'landlord'),
        ('dirty',    'clean'),
        ('rent',     'lease'),
        ('fight',    'argument'),
    ]
    print("\n---Cosine similarity between word pairs--------------")
    print(f"  {'Pair':<28}  similarity")
    print("  " + "-" * 38)
    for w1, w2 in test_pairs:
        try:
            # get the similarity between word 1 and word 2.
            sim = model.wv.similarity(w1, w2)
            print(f"  {w1!r:12} ↔ {w2!r:12}  {sim:.4f}")
        except KeyError as e:
            print(f"  {w1!r} ↔ {w2!r}  - not in vocab: {e}")

    print(f"\nClosest words to 'roommate': {[w for w,_ in model.wv.most_similar('roommate', topn=5)]}")
    print(f"Closest words to 'dirty'   : {[w for w,_ in model.wv.most_similar('dirty',     topn=5)]}")
    
    #------------------------------------
    #KMeans word clustering
    # Get all vocabulary words
    vocab_words  = list(model.wv.index_to_key)
    word_vectors = model.wv[vocab_words]

    kmeans = KMeans(n_clusters=K_BINS, random_state=22, n_init=10)
    kmeans.fit(word_vectors)

    word_to_bin = {word: int(kmeans.labels_[i]) for i, word in enumerate(vocab_words)}

    print(f"\nSample words per bin:")
    for bin_id in range(K_BINS):
        sample = [w for w, b in word_to_bin.items() if b == bin_id][:8]
        print(f"  Bin {bin_id}: {sample}")
    
    # building the document vectors
    doc_vectors = np.array([
        post_to_vector(tok, word_to_bin, K_BINS)
        for tok in post_tokens
    ])
    print(f"\nDocument matrix shape: {doc_vectors.shape}")
    print(f"Post 0 vector: {doc_vectors[0].round(3)}")


    valid_mask   = doc_vectors.sum(axis=1) > 0
    X_valid      = doc_vectors[valid_mask]
    labels_valid = np.argmax(X_valid, axis=1)

    print(f"\nNon-empty posts used for scoring: {valid_mask.sum()} / {len(doc_vectors)}")
    print(f"Label distribution: { {i: int((labels_valid==i).sum()) for i in range(K_BINS)} }\n")

    sil = silhouette_score(X_valid, labels_valid)
    ch  = calinski_harabasz_score(X_valid, labels_valid)
    db  = davies_bouldin_score(X_valid, labels_valid)

    print(f"Silhouette Score          : {sil:.4f}")
    print(f"  {'Good separation' if sil > 0.5 else 'Moderate separation' if sil > 0.25 else 'Weak / overlapping clusters'}\n")
    print(f"Calinski-Harabasz Score   : {ch:.2f}")
    print(f"  Higher is better\n")
    print(f"Davies-Bouldin Index      : {db:.4f}")
    print(f"  Lower is better (0 means perfect, and >2 = poor)\n")

    labels_all = np.argmax(doc_vectors, axis=1)
    coh = top_words_coherence(model, labels_all, tagged)
    if not np.isnan(coh):
        print(f"Top-words Coherence Score : {coh:.4f}")
        print(f"  Higher is better (avg cosine similarity of top-{TOP_K_WORDS} words per cluster)\n")
    else:
        print("Top-words Coherence Score : N/A\n")

    # Getting and printing the top 10 words per bin
    print("=" * 55)
    print("Top 10 most representative words per bin")
    print("=" * 55)
    centroids    = kmeans.cluster_centers_
    vocab_matrix = model.wv[vocab_words]
    for bin_id in range(K_BINS):
        centroid  = centroids[bin_id].reshape(1, -1)
        sims      = cosine_similarity(vocab_matrix, centroid).flatten()
        top10_idx = sims.argsort()[::-1][:10]
        top10     = [(vocab_words[i], round(float(sims[i]), 3)) for i in top10_idx]
        print(f"\nBin {bin_id} - top 10 words:")
        for word, sim in top10:
            print(f"  {word:<20} {sim:.3f}")

    # Printing the final summary for report.
    print("\n" + "=" * 55)
    print("Summary")
    print("=" * 55)
    print(f"  W2V_DIM            : {W2V_DIM}")
    print(f"  min_count          : {MIN_COUNT}")
    print(f"  K_BINS             : {K_BINS}")
    print(f"  Silhouette Score   : {sil:.4f}")
    print(f"  Calinski-Harabasz  : {ch:.2f}")
    print(f"  Davies-Bouldin Index: {db:.4f}")
    print(f"  Top-words Coherence : {coh:.4f}" if not np.isnan(coh) else "  Top-words Coherence  : N/A")


if __name__ == '__main__':
    main()
