# CEFR-Directed Lexical Simplification

UNSW COMP6713 (Natural Language Processing), Term 1 2026 — individual assignment (report dated 20 March 2026).

**中文简介**：给定句子、源 CEFR 等级和目标等级（A1–C2），通过词汇替换把句子的词汇难度调整到目标等级。仅用 CPU、不使用 Transformer：从语料构建 CEFR 词表，结合 WordNet 同义词、GloVe 相似度和二元语言模型筛选并排序候选词。

The solution module `z5565492.py` and `report.pdf` are my own work. `main.py` and `unit_tests.csv` are the course-provided test harness and public tests; the training corpus `data.csv` and the specification are course material and are not included.

## Approach

1. **CEFR vocabulary from the corpus** – tokens are normalised by per-level corpus size (B2 is about 38× larger than A1) and a weighted CEFR index is rounded to a level, giving 20,726 labelled words after intersecting with GloVe. A hand-written table of 36 corrections covers words the corpus mislabels.
2. **Bigram language model** – Laplace-smoothed, 294,331 bigrams; candidates are scored in the partially transformed context.
3. **Candidates** – WordNet same-synset synonyms plus GloVe nearest neighbours (disabled for nouns), filtered by direction of level change and part of speech.
4. **Similarity gate** – five thresholds depending on POS and candidate source (e.g. 0.70 for WordNet nouns, 0.55 for manner adverbs).
5. **Ranking** – `0.65 × GloVe similarity + 0.35 × normalised LM score`, then inflection with pyinflect using the original Penn Treebank tag.

Approaches tried and dropped (spaCy vectors as the only similarity source, WordNet hypernyms/hyponyms, a single global threshold, GloVe noun neighbours) are documented in the report, together with its limitations (sparse A1/A2 data, nouns rarely replaced, one-word context in the bigram model, no multi-word expressions).

## Output on the public tests

Output of `python main.py z5565492` in a fresh Python 3.13 environment on 11 October 2026 with GloVe loaded (about 100 s in total):

| # | Input (source → target level) | Output |
|---|---|---|
| 1 | I purchased a magnificent house yesterday. (C1 → A2) | I **bought** a magnificent house yesterday . |
| 2 | The children were extremely delighted with the performance. (B2 → A2) | … were **quite** delighted … |
| 3 | She attempted to resolve the complicated problem. (B2 → B1) | She **tried** to **answer** the **difficult** problem . |
| 4 | The professor explained the concept very clearly. (B1 → A2) | The professor **told** the concept very clearly . |
| 5 | They constructed a large building near the river. (B2 → A2) | They **built** a **big** building near the river . |
| 6 | He quickly realised his mistake. (B2 → A2) | He quickly **saw** his mistake . |
| 7 | The committee will evaluate the proposal tomorrow. (C1 → B1) | The committee will **examine** the proposal tomorrow . |
| 8 | She was exhausted after completing the assignment. (B2 → A2) | She was **tired** after **finishing** the assignment . |
| 9 | The results demonstrate a significant improvement. (C1 → B1) | The results **show** a **certain** improvement . |
| 10 | The scientist conducted an experiment. (B2 → A2) | unchanged |

These match the examples in `report.pdf` for tests 2, 4, 8 and 9; in test 3 the run also replaces "resolve" by "answer", which the report says was not replaced. The printed vocabulary size is 20,729 words versus 20,726 in the report. If GloVe cannot be downloaded the code falls back to spaCy vectors (and prints a message), and the outputs then differ.

## How to reproduce

Tested with Python 3.13 on Windows in a fresh virtual environment (the pinned versions need Python ≥ 3.11):

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
# place the course corpus data.csv next to z5565492.py (not included)
python main.py z5565492        # about 100 s: builds the vocabulary and bigram model, loads GloVe
```

`gensim` is required (`z5565492.py` imports `gensim.downloader`). On first run it downloads `glove-wiki-gigaword-100` (about 130 MB) and caches it; a few hundred MB of free RAM are needed to load the vectors.

## Files

| File | Description |
|---|---|
| `z5565492.py` | Solution module |
| `report.pdf` | Report with design decisions and abandoned approaches |
| `main.py`, `unit_tests.csv` | Course-provided test harness and public tests |
| `requirements.txt` | Pinned Python dependencies |
