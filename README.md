# CEFR-Directed Lexical Simplification

UNSW COMP6713 (Natural Language Processing), Term 1 2026 — individual assignment.

**中文简介**：给定句子、源 CEFR 等级和目标等级（A1–C2），通过词汇替换把句子的词汇难度调整到目标等级。仅用 CPU、不使用 Transformer：从语料构建 CEFR 词表，结合 WordNet 同义词、GloVe 相似度和二元语言模型筛选并排序候选词。

## Approach

1. **CEFR vocabulary from the corpus** – tokens are normalised by per-level corpus size (B2 is about 38× larger than A1) and a weighted CEFR index is rounded to a level, giving 20,726 labelled words after intersecting with GloVe. A small hand-written table of 36 corrections covers words the corpus mislabels.
2. **Bigram language model** – Laplace-smoothed, 294,331 bigrams; candidates are scored in the partially transformed context.
3. **Candidates** – WordNet same-synset synonyms plus GloVe nearest neighbours (disabled for nouns), filtered by direction of level change and part of speech.
4. **Similarity gate** – five thresholds depending on POS and source (e.g. 0.70 for WordNet nouns, 0.55 for manner adverbs).
5. **Ranking** – `0.65 × GloVe similarity + 0.35 × normalised LM score`, then inflection with pyinflect using the original Penn Treebank tag.

Approaches that were tried and dropped (spaCy vectors as the only similarity source, WordNet hypernyms/hyponyms, a single global threshold, noun neighbours from GloVe) are documented in the report.

## Files

| File | Description |
|---|---|
| `z5565492.py` | Solution module |
| `report.pdf` | Report with design decisions and abandoned approaches |
| `main.py`, `unit_tests.csv` | Course-provided test harness and public tests |
| `requirements.txt` | Python dependencies |

## Running

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
python main.py z5565492
```

GloVe (`glove-wiki-gigaword-100`) is downloaded once through gensim; if it cannot be loaded the code falls back to spaCy vectors. The training corpus `data.csv` (expected next to `z5565492.py`) and the specification are course material and are not included.
