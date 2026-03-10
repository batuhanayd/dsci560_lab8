import json
import os
import re
from collections import Counter

import numpy as np
import nltk
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# paths and constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed_data")
DOCSTRING_JSON = os.path.join(DATA_DIR, "postid_to_docstring.json")
STATS_JSON = os.path.join(DATA_DIR, "embedding_cluster_stats.json")
N_CLUSTERS = 8
TOP_K_WORDS = 10
MIN_WORDS_COHERENCE = 2

CONFIGS = [
    {"name": "config_small", "vector_size": 50, "min_count": 2, "epochs": 10, "window": 5, "dm": 1},
    {"name": "config_medium", "vector_size": 100, "min_count": 2, "epochs": 20, "window": 5, "dm": 1},
    {"name": "config_large", "vector_size": 200, "min_count": 5, "epochs": 15, "window": 5, "dm": 1},
]


def tokenize(text):
    if not text or not isinstance(text, str):
        return []
    text = text.lower().strip()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = nltk.word_tokenize(text)
    return [w for w in words if len(w) > 1 and (w.isalnum() or "_" in w)]


def load_docs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = list(data.keys())
    docs = [data[pid] for pid in ids]
    return ids, docs


def tagged_docs(docstrings):
    out = []
    for i, doc in enumerate(docstrings):
        out.append(TaggedDocument(words=tokenize(doc), tags=[i]))
    return out


def train_model(tagged, cfg):
    return Doc2Vec(
        documents=tagged,
        vector_size=cfg["vector_size"],
        min_count=cfg["min_count"],
        epochs=cfg["epochs"],
        window=cfg["window"],
        dm=cfg["dm"],
        seed=42,
        workers=1,
    )


def doc_vectors(model, n):
    return np.array([model.dv[i] for i in range(n)], dtype=np.float64)


def cluster_cosine(X, k=N_CLUSTERS):
    ac = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    return ac.fit_predict(X)


def top_words_coherence(model, labels, tagged, top_k=TOP_K_WORDS):
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
    if not coh_list:
        return float("nan")
    return float(np.mean(coh_list))


def get_top_words_and_sims(model, labels, tagged, top_k=TOP_K_WORDS):
    """Per cluster: (top_words_list, mean_similarity, pairwise_list, word_to_sim_dict)."""
    if not hasattr(model, "wv") or len(model.wv) == 0:
        return []
    n_c = int(labels.max()) + 1
    out = []
    for c in range(n_c):
        idx = np.where(labels == c)[0]
        cnt = Counter()
        for i in idx:
            if i < len(tagged):
                cnt.update(tagged[i].words)
        top = [w for w, _ in cnt.most_common(top_k) if w in model.wv]
        if len(top) < MIN_WORDS_COHERENCE:
            out.append((top, None, [], {}))
            continue
        vecs = np.array([model.wv[w] for w in top], dtype=np.float64)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs_n = vecs / norms
        sim = np.dot(vecs_n, vecs_n.T)
        triu = np.triu_indices(len(top), k=1)
        pairwise = [round(float(sim[triu][i]), 4) for i in range(sim[triu].size)]
        # per-word: mean similarity to other top words
        n_w = len(top)
        word_to_sim = {}
        for i in range(n_w):
            others = (sim[i].sum() - 1.0) / (n_w - 1) if n_w > 1 else 0.0
            word_to_sim[top[i]] = round(float(others), 4)
        out.append((top, float(np.mean(sim[triu])), pairwise, word_to_sim))
    return out


def print_top_words_and_similarities(name, model, labels, tagged, top_k=TOP_K_WORDS):
    data = get_top_words_and_sims(model, labels, tagged, top_k)
    n_c = len(data)
    print(f"\n-------- Top words and similarity scores: {name} --------", flush=True)
    for c in range(n_c):
        top, mean_sim, pairwise, word_to_sim = data[c]
        print(f"\n  Cluster {c}:", flush=True)
        print(f"    Top words (word -> mean similarity to other top words): {word_to_sim}", flush=True)
        if mean_sim is not None:
            print(f"    Mean pairwise cosine similarity: {mean_sim:.4f}", flush=True)
            print(f"    All pairwise similarities: {pairwise}", flush=True)
        else:
            print(f"    Mean pairwise cosine similarity: (fewer than 2 words in vocab)", flush=True)


def metrics(vectors, labels, model, tagged):
    n_c = int(labels.max()) + 1
    if n_c < 2:
        return {"silhouette_score": None, "calinski_harabasz_score": None,
                "davies_bouldin_index": None, "top_words_coherence": None}
    sil = silhouette_score(vectors, labels, metric="cosine")
    ch = calinski_harabasz_score(vectors, labels)
    db = davies_bouldin_score(vectors, labels)
    coh = top_words_coherence(model, labels, tagged)
    return {
        "silhouette_score": float(sil),
        "calinski_harabasz_score": float(ch),
        "davies_bouldin_index": float(db),
        "top_words_coherence": float(coh) if not np.isnan(coh) else None,
    }


def show_clusters(name, labels, docstrings, post_ids, n_sample=4, max_len=150):
    n_c = int(labels.max()) + 1
    print(f"\n-------- Cluster examination: {name} --------", flush=True)
    for c in range(n_c):
        idx = np.where(labels == c)[0]
        print(f"\n  Cluster {c} (size={len(idx)})", flush=True)
        step = max(1, len(idx) // n_sample)
        for i in idx[::step][:n_sample]:
            if i < len(docstrings):
                s = docstrings[i][:max_len] + ("..." if len(docstrings[i]) > max_len else "")
                pid = post_ids[i] if i < len(post_ids) else i
                print(f"    [{pid}] {s}", flush=True)


def main():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    if not os.path.isfile(DOCSTRING_JSON):
        raise FileNotFoundError(DOCSTRING_JSON)

    post_ids, docstrings = load_docs(DOCSTRING_JSON)
    print(f"Loaded {len(docstrings)} documents.", flush=True)

    tagged = tagged_docs(docstrings)
    all_stats = {}

    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"\n--- {name}: vector_size={cfg['vector_size']}, min_count={cfg['min_count']}, epochs={cfg['epochs']} ---", flush=True)

        model = train_model(tagged, cfg)
        X = doc_vectors(model, len(docstrings))
        print(f"  Vectors: {X.shape}", flush=True)

        labels = cluster_cosine(X, N_CLUSTERS)
        m = metrics(X, labels, model, tagged)

        all_stats[name] = {
            "config": {k: v for k, v in cfg.items() if k != "dm"},
            "n_documents": len(docstrings),
            "n_clusters": N_CLUSTERS,
            **m,
        }

        print(f"  Silhouette: {m['silhouette_score']:.4f}  CH: {m['calinski_harabasz_score']:.2f}  DB: {m['davies_bouldin_index']:.4f}  Coherence: {m['top_words_coherence']}", flush=True)
        show_clusters(name, labels, docstrings, post_ids, 4, 150)
        print_top_words_and_similarities(name, model, labels, tagged, TOP_K_WORDS)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved to {STATS_JSON}", flush=True)

    print("\n-------- Summary --------", flush=True)
    for name, s in all_stats.items():
        print(f"  {name}: sil={s['silhouette_score']:.4f} CH={s['calinski_harabasz_score']:.2f} DB={s['davies_bouldin_index']:.4f} coh={s['top_words_coherence']}", flush=True)
    best_sil = max(all_stats, key=lambda k: all_stats[k]["silhouette_score"] or -2)
    best_ch = max(all_stats, key=lambda k: all_stats[k]["calinski_harabasz_score"] or -1)
    best_db = min(all_stats, key=lambda k: all_stats[k]["davies_bouldin_index"] or 999)
    best_coh = max(all_stats, key=lambda k: all_stats[k]["top_words_coherence"] or -2)
    print(f"  Best: silhouette={best_sil}, CH={best_ch}, DB={best_db}, coherence={best_coh}", flush=True)


if __name__ == "__main__":
    main()
