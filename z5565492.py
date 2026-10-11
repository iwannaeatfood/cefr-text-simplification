# z5565492.py  COMP6713 Assignment 1
# GloVe used for similarity if internet/cache available; falls back to spaCy vectors

from pathlib import Path
import re
from collections import Counter

import numpy as np
import pandas as pd
import spacy
import pyinflect          # must import to register ._.inflect on tokens
import nltk
from nltk.corpus import wordnet
import gensim.downloader as gensim_api

# download wordnet if not already cached
for _res, _pkg in [("corpora/wordnet", "wordnet"), ("corpora/omw-1.4", "omw-1.4")]:
    try:
        nltk.data.find(_res)
    except LookupError:
        nltk.download(_pkg, quiet=True)


CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_INDEX  = {lvl: i for i, lvl in enumerate(CEFR_LEVELS)}

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}

_WN_POS = {
    "NOUN": wordnet.NOUN,
    "VERB": wordnet.VERB,
    "ADJ":  wordnet.ADJ,
    "ADV":  wordnet.ADV,
}

# similarity thresholds. noun threshold is stricter because nouns in wordnet
# have many obscure synsets that produce wrong replacements (e.g. house->theater,
# problem->job). was using 0.40 for everything before and got bad noun outputs.
MIN_SIM_WORDNET      = 0.40
MIN_SIM_WORDNET_NOUN = 0.70
MIN_SIM_WORDNET_ADV  = 0.55   # -ly manner adverbs need stricter gate; really≈clearly is ~0.45
MIN_SIM_GLOVE        = 0.72
MIN_SIM_GLOVE_ADJ    = 0.65

# words blocked after testing -- passed all filters but were semantically wrong
# or produced bad grammar (carry an experiment, babies for children, etc.)
_REPLACEMENT_BLOCKLIST = {
    "baby", "babies", "infant", "infants", "toddler", "toddlers",
    "construction", "carry", "carrying", "carries",
    # intensifiers that GloVe scores as similar to other intensifiers
    # but are not simpler (incredible ≈ magnificent in GloVe space)
    "incredible", "unbelievable", "extraordinary",
}

REPLACE_THRESHOLD = 1   # min level gap before we bother replacing
TOP_K  = 30             # max candidates to rank
ALPHA  = 0.65           # glove sim weight vs LM score

_STOP_WORDS = {
    "a","an","the","and","or","but","if","in","on","at","to","for","of",
    "with","by","from","as","is","was","are","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","shall","can","not","no","nor","so","yet","both","either",
    "whether","because","since","while","although","though","unless","until",
    "after","before","when","where","who","which","that","this","these",
    "those","what","how","why","there","here","their","they","them","he",
    "she","it","we","you","i","me","my","his","her","its","our","your",
    "than","then","too","very","just","about","above","across","again",
    "against","all","also","always","among","any","back","between","each",
    "every","few","into","more","most","much","now","off","only","other",
    "out","over","own","same","some","such","through","under","up","upon",
    "well","without","one","two","three",
}


def load_training_data(path="data.csv"):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError("data.csv not found.")
    df = pd.read_csv(file_path)
    if not {"text", "cefr_level"}.issubset(df.columns):
        raise ValueError("data.csv must contain columns: text, cefr_level")
    return df


def build_cefr_vocab(df):
    """Assign a CEFR level to each word using normalised corpus frequency.

    Raw counts would label everything B2 (that corpus is ~38x larger than A1).
    Dividing by level total fixes this. Weighted average index then rounded.
    """
    level_counts: dict[str, Counter] = {lvl: Counter() for lvl in CEFR_LEVELS}
    level_total:  dict[str, int]     = {lvl: 0 for lvl in CEFR_LEVELS}

    for _, row in df.iterrows():
        lvl = row["cefr_level"]
        if lvl not in CEFR_LEVELS:
            continue
        for tok in re.findall(r"[a-z]+", row["text"].lower()):
            if tok not in _STOP_WORDS and len(tok) >= 3:
                level_counts[lvl][tok] += 1
                level_total[lvl] += 1

    all_words: set[str] = set()
    for c in level_counts.values():
        all_words.update(c.keys())

    word_cefr: dict[str, str] = {}
    for word in all_words:
        total_raw = sum(level_counts[lvl].get(word, 0) for lvl in CEFR_LEVELS)
        if total_raw < 2:
            continue
        norm_freqs = {
            lvl: (level_counts[lvl].get(word, 0) / level_total[lvl])
            if level_total[lvl] > 0 else 0.0
            for lvl in CEFR_LEVELS
        }
        total_norm = sum(norm_freqs.values())
        if total_norm == 0:
            continue
        avg_idx   = sum(CEFR_INDEX[lvl] * norm_freqs[lvl]
                        for lvl in CEFR_LEVELS) / total_norm
        level_idx = max(0, min(len(CEFR_LEVELS) - 1, int(round(avg_idx))))
        word_cefr[word] = CEFR_LEVELS[level_idx]

    return word_cefr


def build_bigram_model(df):
    """Bigram LM with Laplace smoothing: P(w2|w1) = (C(w1,w2)+1)/(C(w1)+|V|)."""
    bigram_counts:  Counter = Counter()
    unigram_counts: Counter = Counter()

    for text in df["text"]:
        tokens = ["<s>"] + re.findall(r"[a-z]+", text.lower()) + ["</s>"]
        for w1, w2 in zip(tokens, tokens[1:]):
            bigram_counts[(w1, w2)] += 1
            unigram_counts[w1]      += 1

    return bigram_counts, unigram_counts, len(unigram_counts)


def bigram_log_prob(tokens, bigram_counts, unigram_counts, vocab_size):
    """Sum of Laplace-smoothed bigram log probs for a token sequence."""
    seq      = ["<s>"] + tokens + ["</s>"]
    log_prob = 0.0
    for w1, w2 in zip(seq, seq[1:]):
        num = bigram_counts.get((w1, w2), 0) + 1
        den = unigram_counts.get(w1, 0) + vocab_size
        log_prob += np.log(num / den)
    return log_prob


def glove_similarity(word1: str, word2: str) -> float:
    """Cosine similarity via GloVe. Returns 0.0 for OOV words."""
    try:
        return float(_glove.similarity(word1, word2))
    except KeyError:
        return 0.0


def glove_most_similar(word: str, topn: int = TOP_K):
    """Top-n GloVe neighbours. Returns [] if word is OOV."""
    try:
        return _glove.most_similar(word, topn=topn)
    except KeyError:
        return []


def word_primary_pos(word: str, wn_pos) -> bool:
    """True if wn_pos accounts for >50% of this word's WordNet synsets.

    Plurality wasn't strict enough -- 'meeting' kept slipping in as a verb.
    """
    all_synsets = wordnet.synsets(word)
    if not all_synsets:
        return False
    counts = Counter(s.pos() for s in all_synsets)
    total  = sum(counts.values())
    return counts.get(wn_pos, 0) / total > 0.5


print("Loading spaCy model")
_nlp = spacy.load("en_core_web_md")

print("Loading training data from data.csv")
_df = load_training_data("data.csv")

print("Building CEFR word vocabulary")
_word_cefr = build_cefr_vocab(_df)

# Many high-frequency simple words are excluded by _STOP_WORDS or get
# pushed to B1/B2 because they appear uniformly across all corpus levels.
# This supplement corrects the most common cases so simplification to A1/A2
# actually finds candidates (e.g. extremely->very, delighted->happy).
# Only overrides when the derived label is harder than the known level.
_SIMPLE_SUPPLEMENT = {
    # adjectives
    "happy":    "A1", "glad":     "A1", "sad":      "A1", "pleased":  "A2",
    "cheerful": "B1", "joyful":   "B1", "easy":     "A1", "hard":     "A1",
    "quick":    "A1", "fast":     "A1", "clear":    "A2", "sure":     "A1",
    "young":    "A1", "early":    "A1", "open":     "A1", "free":     "A2",
    "real":     "A1", "true":     "A1", "right":    "A1", "wrong":    "A2",
    "huge":     "A2", "tiny":     "A2", "angry":    "A1", "scared":   "A2",
    # adverbs -- most are in _STOP_WORDS so never get a corpus label
    "very":     "A1", "really":   "A1", "quite":    "A2", "pretty":   "A2",
    "nearly":   "A2", "almost":   "A2", "often":    "A2", "always":   "A1",
    # verbs that are commonly simpler than their formal synonyms
    "run":      "A1", "keep":     "A2", "show":     "A1", "feel":     "A1",
    "seem":     "A2", "look":     "A1", "manage":   "B1", "perform":  "B1",
    "handle":   "B1", "finish":   "A2", "start":    "A1", "stop":     "A1",
}
for _sw, _slvl in _SIMPLE_SUPPLEMENT.items():
    if _sw not in _word_cefr or CEFR_INDEX[_word_cefr[_sw]] > CEFR_INDEX[_slvl]:
        _word_cefr[_sw] = _slvl

print("Training bigram language model")
_bigram_counts, _unigram_counts, _vocab_size = build_bigram_model(_df)

print("Loading GloVe embeddings")
print("This downloads once and is cached automatically by gensim.")


class _SpacyFallback:
    """Used when GloVe is unavailable (no internet, no local cache).

    similarity() uses spaCy en_core_web_md vectors.
    most_similar() returns [] -- WordNet candidates still work fine.
    """
    def similarity(self, w1, w2):
        v1, v2 = _nlp.vocab[w1], _nlp.vocab[w2]
        if v1.has_vector and v2.has_vector:
            return float(v1.similarity(v2))
        return 0.0

    def most_similar(self, word, topn=30):
        return []

    def __contains__(self, word):
        return _nlp.vocab[word].has_vector

#if glove could not work
try:
    _glove = gensim_api.load("glove-wiki-gigaword-100")
except Exception:
    print("!!!GloVe unavailable, falling back to spaCy vectors.")
    _glove = _SpacyFallback()

# index CEFR words by level, keeping only those also in GloVe
_level_words: dict[str, list[str]] = {lvl: [] for lvl in CEFR_LEVELS}
for _w, _lvl in _word_cefr.items():
    if _w in _glove:
        _level_words[_lvl].append(_w)

print(
    f"Initialisation complete. "
    f"Vocab: {len(_word_cefr):,} words | "
    f"Bigrams: {len(_bigram_counts):,} | "
    f"GloVe level sizes: { {k: len(v) for k, v in _level_words.items()} }"
)


def get_word_level(word: str):
    return _word_cefr.get(word.lower())


_KNOWN_FLAT_ADVS = {
    "fast", "hard", "well", "soon", "late", "still", "just", "quite",
    "almost", "nearly", "very", "really", "pretty", "rather", "often",
    "always", "never", "ever", "now", "then", "here", "there", "already",
}

# Degree/intensifier adverbs substitute freely across -ly and flat forms.
# extremely=>quite is fine; clearly=>really is not.
_DEGREE_ADVS = {
    "very", "really", "quite", "pretty", "rather", "fairly", "extremely",
    "highly", "greatly", "deeply", "truly", "fully", "mostly", "largely",
    "nearly", "almost", "barely", "hardly", "somewhat", "slightly",
}

def inflect_to_match(replacement_word: str, original_token):
    """Inflect replacement to match the original token's Penn Treebank tag.

    Returns None when a valid form cannot be produced.
    ADV sub-type rules:
      - degree adverb (extremely, quite...) => accept any adverb form freely
      - manner -ly adverb (clearly, quickly) => only accept other -ly adverbs
      - flat adverb (well, fast) => accept -ly or flat
    """
    penn_tag = original_token.tag_

    if penn_tag in ("RB", "RBR", "RBS"):
        orig = original_token.lemma_.lower()
        if orig not in _DEGREE_ADVS:
            # manner or flat adverb: reject degree adverb replacements entirely
            if replacement_word in _DEGREE_ADVS:
                return None
            if orig.endswith("ly"):
                if not replacement_word.endswith("ly"):
                    return None
            else:
                if not (replacement_word.endswith("ly") or
                        replacement_word in _KNOWN_FLAT_ADVS):
                    return None

    rep_doc = _nlp(replacement_word)
    if not rep_doc:
        return None
    inflected = rep_doc[0]._.inflect(penn_tag)
    return inflected if inflected else replacement_word


def preserve_case(original: str, replacement: str) -> str:
    if not replacement:
        return replacement
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def get_wordnet_synonyms(token, source_level: str,
                          target_level: str, simplifying: bool):
    """Same-synset synonyms that move in the right CEFR direction.

    Only same-synset, not hypernyms/hyponyms -- tried hyponyms earlier and got
    'monkeys' for 'children' via the primate chain. Not worth it.
    Directional filter: accept anything easier than source (not just at target),
    because the A2 vocab is too small (~990 words) to match exactly.
    """
    wn_pos = _WN_POS.get(token.pos_)
    if wn_pos is None:
        return []

    lemma      = token.lemma_.lower()
    synsets    = wordnet.synsets(lemma, pos=wn_pos)
    source_idx = CEFR_INDEX.get(source_level, CEFR_INDEX[target_level])
    target_idx = CEFR_INDEX[target_level]
    n_levels   = len(CEFR_LEVELS)

    seen: dict[str, float] = {}
    for syn in synsets:
        for lm in syn.lemmas():
            w = lm.name().lower().replace("_", "")
            if w == lemma or " " in w or w not in _word_cefr:
                continue
            cand_idx = CEFR_INDEX[_word_cefr[w]]
            if simplifying     and cand_idx >= source_idx:
                continue
            if not simplifying and cand_idx <= source_idx:
                continue
            dist  = abs(cand_idx - target_idx)
            score = 1.0 - dist / n_levels
            seen[w] = max(seen.get(w, 0.0), score)

    return list(seen.items())


def get_glove_candidates(token, target_level: str, simplifying: bool,
                          source_level: str):
    """GloVe nearest-neighbour candidates filtered to CEFR vocab + POS match."""
    lemma = token.lemma_.lower()
    if lemma not in _glove:
        return []

    wn_pos     = _WN_POS.get(token.pos_)
    source_idx = CEFR_INDEX.get(source_level, CEFR_INDEX[target_level])

    results = []
    glove_thresh = MIN_SIM_GLOVE_ADJ if token.pos_ == "ADJ" else MIN_SIM_GLOVE
    for word, sim in glove_most_similar(lemma, topn=200):
        if word not in _word_cefr:
            continue
        cand_idx = CEFR_INDEX[_word_cefr[word]]
        if simplifying     and cand_idx >= source_idx:
            continue
        if not simplifying and cand_idx <= source_idx:
            continue
        if wn_pos is not None and wordnet.synsets(word):
            if not word_primary_pos(word, wn_pos):
                continue
        if sim >= glove_thresh:
            results.append((word, sim))
        if len(results) >= TOP_K:
            break

    return results


def gather_candidates(token, target_level: str, simplifying: bool,
                       source_level: str):
    """Merge wordnet + glove candidates, apply similarity gates and blocklist.

    GloVe search is disabled for nouns -- noun neighbours are topic-similar
    not synonymous (committee/house, results/times all score 0.70+ in GloVe
    because they share political/academic context). Kept breaking noun outputs
    so turned it off entirely for that POS.
    """
    seen: dict[str, float] = {}
    lemma = token.lemma_.lower()

    if token.pos_ == "NOUN":
        wn_threshold = MIN_SIM_WORDNET_NOUN
    elif token.pos_ == "ADV":
        orig = token.lemma_.lower()
        wn_threshold = MIN_SIM_WORDNET_ADV if orig.endswith("ly") else MIN_SIM_WORDNET
    else:
        wn_threshold = MIN_SIM_WORDNET

    for word, _proximity in get_wordnet_synonyms(
            token, source_level, target_level, simplifying):
        sim = glove_similarity(lemma, word)
        if sim >= wn_threshold:
            seen[word] = max(seen.get(word, 0.0), sim)

    if token.pos_ != "NOUN":
        for word, sim in get_glove_candidates(
                token, target_level, simplifying, source_level):
            seen[word] = max(seen.get(word, 0.0), sim)

    for bl_word in _REPLACEMENT_BLOCKLIST:
        seen.pop(bl_word, None)

    return sorted(seen.items(), key=lambda x: -x[1])[:TOP_K]


def find_best_replacement(token, target_level: str, simplifying: bool,
                           token_strings: list, token_idx: int,
                           source_level: str):
    """Score candidates by ALPHA*glove_sim + (1-ALPHA)*normalised_LM, pick best."""
    candidates = gather_candidates(token, target_level, simplifying, source_level)
    if not candidates:
        return None

    lm_scores: list[float] = []
    for word, _ in candidates:
        ctx            = token_strings.copy()
        ctx[token_idx] = word
        lm_scores.append(
            bigram_log_prob(ctx, _bigram_counts, _unigram_counts, _vocab_size)
        )

    min_lm   = min(lm_scores)
    max_lm   = max(lm_scores)
    lm_range = (max_lm - min_lm) if max_lm != min_lm else 1.0

    # rank candidates, try in order until one inflects to the right POS
    ranked = sorted(
        zip(candidates, lm_scores),
        key=lambda x: -(ALPHA * x[0][1] + (1.0 - ALPHA) * (x[0][1] - min_lm) / lm_range
                        if lm_range else ALPHA * x[0][1]),
    )
    # recompute combined scores properly for sorting
    scored = []
    for (word, sim), lm_score in zip(candidates, lm_scores):
        norm_lm  = (lm_score - min_lm) / lm_range
        scored.append((ALPHA * sim + (1.0 - ALPHA) * norm_lm, word))
    scored.sort(reverse=True)

    for _, best_word in scored:
        inflected = inflect_to_match(best_word, token)
        if inflected is not None:
            return inflected

    return None


def transform_sentence(sentence: str, source_level: str, target_level: str) -> str:
    """Replace content words to shift vocabulary from source_level to target_level.

    Only replaces a word if it's at least REPLACE_THRESHOLD levels away from
    target and on the wrong side. Updates token_strings after each replacement
    so LM scoring uses the partially-transformed context.
    """
    if source_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid source CEFR level: {source_level}")
    if target_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid target CEFR level: {target_level}")

    if source_level == target_level:
        return sentence

    target_idx  = CEFR_INDEX[target_level]
    source_idx  = CEFR_INDEX[source_level]
    simplifying = source_idx > target_idx

    doc           = _nlp(sentence)
    token_strings = [tok.text.lower() for tok in doc]
    output_tokens = [tok.text         for tok in doc]

    for i, token in enumerate(doc):
        if token.pos_ not in CONTENT_POS:
            continue
        if token.is_punct or token.is_space or token.like_num:
            continue

        word_level = get_word_level(token.text)
        if word_level is None:
            continue

        word_idx = CEFR_INDEX[word_level]

        if abs(word_idx - target_idx) < REPLACE_THRESHOLD:
            continue
        if simplifying     and word_idx <= target_idx:
            continue
        if not simplifying and word_idx >= target_idx:
            continue

        replacement = find_best_replacement(
            token, target_level, simplifying,
            token_strings, i, source_level
        )
        if replacement is None:
            continue

        replacement      = preserve_case(token.text, replacement)
        output_tokens[i] = replacement
        token_strings[i] = replacement.lower()

    return " ".join(output_tokens)