# ============================================================
# PATTERN RETENTION — RÍTMICO (COMPATIBLE 100% CON TU MÓDULO)
# ============================================================

from transfolk_patterns.rhythmicPatternSearcher import (
    get_duration_tokens,
    generate_duration_combinations,
    count_duration_combinations
)


# ------------------------------------------------------------
# 1. Extraer patrones rítmicos del corpus (usando TU módulo)
# ------------------------------------------------------------
def extract_rhythmic_patterns(corpus_sequences, vocab, n_min=3, n_max=6, min_count=5):
    combos = generate_duration_combinations(vocab, n_min=n_min, n_max=n_max)
    counts = count_duration_combinations(corpus_sequences, vocab, combos)
    longest_counts = counts[0]["counts"]
    corpus_patterns = {
        tuple(p): freq
        for p, freq in longest_counts.items()
        if freq >= min_count
    }
    return corpus_patterns


# ------------------------------------------------------------
# 2. Reducir secuencia a DUR_ONLY (igual que tu módulo)
# ------------------------------------------------------------
def extract_dur_sequence(seq, vocab):
    dur_ids = set(get_duration_tokens(vocab).values())
    return [tok for tok in seq if tok in dur_ids]


# ------------------------------------------------------------
# 3. Contar patrones en secuencias generadas
# ------------------------------------------------------------
def count_patterns_in_generated(generated_sequences, corpus_patterns, vocab):
    gen_counts = {p: 0 for p in corpus_patterns}
    for seq in generated_sequences:
        dur_seq = extract_dur_sequence(seq, vocab)
        L = len(dur_seq)
        for pattern in corpus_patterns:
            n = len(pattern)
            for i in range(L - n + 1):
                if tuple(dur_seq[i:i+n]) == pattern:
                    gen_counts[pattern] += 1
    return gen_counts


# ------------------------------------------------------------
# 4. Métrica final Pattern-Retention Rate
# ------------------------------------------------------------
def compute_rhythmic_retention_rate(corpus_counts, gen_counts):
    if not corpus_counts:
        return 0.0, {}
    R_i = {}
    for pattern, freq_corpus in corpus_counts.items():
        freq_gen = gen_counts.get(pattern, 0)
        R_i[pattern] = freq_gen / freq_corpus if freq_corpus > 0 else 0.0
    R_global = sum(R_i.values()) / len(R_i)
    return R_global, R_i


# ------------------------------------------------------------
# 5. Pipeline completo que debes llamar desde main.py
# ------------------------------------------------------------
def rhythmic_pattern_retention(corpus_sequences, generated_sequences, vocab,
                               n_min=3, n_max=6, min_count=5):
    corpus_counts = extract_rhythmic_patterns(
        corpus_sequences,
        vocab,
        n_min=n_min,
        n_max=n_max,
        min_count=min_count
    )
    gen_counts = count_patterns_in_generated(
        generated_sequences,
        corpus_counts,
        vocab
    )
    return compute_rhythmic_retention_rate(corpus_counts, gen_counts)


# ============================================================
# PATTERN RETENTION — MELÓDICO
# ============================================================

from transfolk_patterns.melodicPatternSearcher import (
    sequence_to_intervals,
    count_melodic_patterns
)

def extract_melodic_patterns(corpus_sequences, vocab,
                             n_min=3, n_max=6, min_count=5):
    counts = count_melodic_patterns(corpus_sequences, vocab, n_min, n_max)
    longest = counts[0]["counts"]
    return {
        tuple(p): freq
        for p, freq in longest.items()
        if freq >= min_count
    }

def count_melodic_patterns_generated(generated_sequences, corpus_patterns, vocab):
    gen_counts = {p: 0 for p in corpus_patterns}
    for seq in generated_sequences:
        intervals = sequence_to_intervals(seq, vocab)
        L = len(intervals)
        for pattern in corpus_patterns:
            n = len(pattern)
            for i in range(L - n + 1):
                if tuple(intervals[i:i+n]) == pattern:
                    gen_counts[pattern] += 1
    return gen_counts

def compute_melodic_retention_rate(corpus_counts, gen_counts):
    if not corpus_counts:
        return 0.0, {}
    R_i = {}
    for pat, freq_corpus in corpus_counts.items():
        freq_gen = gen_counts.get(pat, 0)
        R_i[pat] = freq_gen / freq_corpus if freq_corpus > 0 else 0.0
    R_global = sum(R_i.values()) / len(R_i)
    return R_global, R_i

def melodic_pattern_retention(corpus_sequences, generated_sequences, vocab,
                              n_min=3, n_max=6, min_count=5):
    corpus_counts = extract_melodic_patterns(
        corpus_sequences,
        vocab,
        n_min=n_min,
        n_max=n_max,
        min_count=min_count
    )
    gen_counts = count_melodic_patterns_generated(
        generated_sequences,
        corpus_counts,
        vocab
    )
    return compute_melodic_retention_rate(corpus_counts, gen_counts)
