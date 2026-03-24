from transfolk_patterns.melodicPatternSearcher import count_melodic_patterns, sequence_to_intervals

def extract_melodic_patterns(corpus_sequences, vocab,
                             n_min=1, n_max=3, min_count=3):

    counts = count_melodic_patterns(corpus_sequences, vocab, n_min, n_max)
    longest = counts[0]["counts"]

    corpus_patterns = {
        tuple(p): freq
        for p, freq in longest.items()
        if freq >= min_count
    }
    return corpus_patterns


def extract_intervals_generated(seq, vocab):
    return sequence_to_intervals(seq, vocab)


def count_melodic_patterns_generated(generated_sequences, corpus_patterns, vocab):
    gen_counts = {p: 0 for p in corpus_patterns}

    for seq in generated_sequences:
        intervals = extract_intervals_generated(seq, vocab)
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
                              n_min=1, n_max=3, min_count=3):

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
