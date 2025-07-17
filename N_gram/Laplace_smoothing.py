from collections import Counter
import re

def laplace_smoothing(corpus, prev_word, word):
    tokens = re.findall(r"\b\w+\b", corpus.lower())
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    bigram_counts = Counter(bigrams)
    unigram_counts = Counter(tokens)
    vocab_size = len(set(tokens))

    numerator = bigram_counts[(prev_word.lower(), word.lower())] + 1
    denominator = unigram_counts[prev_word.lower()] + vocab_size

    return numerator / denominator

corpus = "tom studies linguistics tom reads tom talks tom eats"
prev_word = input('Previous word: ')
word = input('Target word: ')

prob = laplace_smoothing(corpus, prev_word, word)
print(f"Laplace smoothing probability P({word}|{prev_word}) = {prob:.4f}")
