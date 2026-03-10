## DSCI 560 Lab 8 – Document Embeddings and Clustering

This repo contains code for **Lab 8**, split into:

- **Part 1**: `doc2vec_embeddings_clustering.py` – Doc2Vec + Agglomerative Clustering on Reddit docstrings.
- **Part 2**: `lab_8_part2.py` – Word2Vec + KMeans binning on merged Reddit posts.

Both parts rely on `gensim`, `nltk`, `numpy`, and `scikit-learn`.

---

## Environment setup

Recommended: create a fresh virtual environment (Python 3.8–3.12).

```bash
py -3.12 -m venv .venv           # or: python -m venv .venv
.\.venv\Scripts\Activate.ps1     # PowerShell on Windows
pip install gensim nltk numpy scikit-learn
```

Download NLTK data (the scripts will also try to download what they need):

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

Run all commands **from the project root** (directory containing `doc2vec_embeddings_clustering.py` and `lab_8_part2.py`).

---

## Part 1 – `doc2vec_embeddings_clustering.py`

Script that builds **Doc2Vec** embeddings from Reddit post docstrings, clusters them with cosine distance, and writes evaluation metrics to a JSON file.

### What it does

1. **Loads** `data/processed_data/postid_to_docstring.json` (post ID → full docstring text).
2. **Tokenizes** each docstring (NLTK, lowercase, URLs stripped).
3. **Trains** three Doc2Vec models with different settings (vector size, `min_count`, `epochs`).
4. **Clusters** document vectors with `AgglomerativeClustering`, cosine distance, 8 clusters.
5. **Computes** four metrics per config: silhouette score, Calinski–Harabasz, Davies–Bouldin, top-words coherence.
6. **Writes** `data/processed_data/embedding_cluster_stats.json` and prints cluster samples and a short summary.

### Input

| File | Description |
|------|-------------|
| `data/processed_data/postid_to_docstring.json` | JSON object: `{ "post_id": "docstring text", ... }`. Must exist. |

### Output

| Output | Description |
|--------|-------------|
| `data/processed_data/embedding_cluster_stats.json` | Per-config stats: config params, `n_documents`, `n_clusters`, `silhouette_score`, `calinski_harabasz_score`, `davies_bouldin_index`, `top_words_coherence`. |
| Console | Progress, metrics per config, sample docstrings per cluster, and which config is best on each metric. |

### Doc2Vec configs

| Config | vector_size | min_count | epochs |
|--------|-------------|-----------|--------|
| config_small | 50 | 2 | 10 |
| config_medium | 100 | 2 | 20 |
| config_large | 200 | 5 | 15 |

All use `window=5` and distributed memory (`dm=1`).

### Metrics

- **Silhouette score** (higher better): how well points match their cluster vs others; uses cosine.
- **Calinski–Harabasz** (higher better): ratio of between-cluster to within-cluster variance.
- **Davies–Bouldin** (lower better): average similarity between each cluster and its most similar cluster.
- **Top-words coherence** (higher better): per cluster, top-10 words by frequency; mean pairwise cosine similarity of their word vectors in the Doc2Vec model, then averaged over clusters.

### How to run (Part 1)

From the project root:

```bash
python doc2vec_embeddings_clustering.py
```

---

## Part 2 – `lab_8_part2.py`

Script that trains a **Word2Vec** model on merged Reddit posts, clusters word vectors with **KMeans**, builds simple document vectors from bin frequencies, and evaluates the resulting clusters.

### What it does

1. **Parses CLI arguments** (all optional, with defaults):
   - `--w2v_dim` (int, default `100`): Word2Vec embedding dimension.
   - `--min_count` (int, default `2`): minimum word frequency to keep in the vocab.
   - `--k_bins` (int, default `3`): number of KMeans clusters / bins.
2. **Loads** `reddit_merged.json` – list of Reddit posts with at least `title` and `text` fields.
3. **Tokenizes** each post:
   - Uses NLTK sentence and word tokenization.
   - Lowercases, removes non-alphabetic tokens, stopwords, and very short words.
   - Returns:
     - `all_sentences` for Word2Vec training.
     - `post_tokens` (flat token list per post) for document vectors.
4. **Trains Word2Vec (CBOW)** with:
   - `vector_size = W2V_DIM`, `window = 5`, `min_count = MIN_COUNT`, `sg = 0` (CBOW).
5. **KMeans clustering on word vectors**:
   - Clusters the Word2Vec vocabulary into `K_BINS` bins.
   - Builds a `word_to_bin` mapping (word → bin id).
6. **Builds document vectors**:
   - For each post, converts its tokens into a length-`K_BINS` vector of normalized bin frequencies.
7. **Evaluates clusters**:
   - For non-empty doc vectors:
     - **Silhouette score**
     - **Calinski–Harabasz score**
     - **Davies–Bouldin index**
   - For all docs:
     - **Top-words coherence** based on most frequent words per bin.
8. **Prints**:
   - Sample words per bin.
   - Top 10 most representative words per bin (closest to each centroid).
   - Final summary section with all metrics and the chosen hyperparameters.

### Input

| File | Description |
|------|-------------|
| `reddit_merged.json` | List of objects: each post with `title` and `text` (and possibly other fields). |

Ensure this file is in the **project root** (same directory as `lab_8_part2.py`).

### Output

All results are printed to the console:

- **Word2Vec validation**: cosine similarities for some word pairs and nearest neighbors.
- **Cluster quality**: silhouette, Calinski–Harabasz, Davies–Bouldin, and top-words coherence.
- **Interpretability**: sample words per bin and top-10 representative words per bin.
- **Summary**: hyperparameters and final metric values for use in the lab report.

### How to run (Part 2)

From the project root:

```bash
python lab_8_part2.py
```
