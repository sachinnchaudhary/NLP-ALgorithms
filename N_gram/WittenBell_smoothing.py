from collections import Counter

import re 



from collections import Counter
import re

def unigram(word, corpus): 
    tokens = re.findall(r"\b\w+\b", corpus.lower())
    word_count = Counter(tokens)
    total_words = len(tokens)
    return word_count[word.lower()] / total_words 

def witten_bell_smoothing(corpus, prev_word, word):
    tokens = re.findall(r"\b\w+\b", corpus.lower())
    bigrams = list(zip(tokens[:-1], tokens[1:]))

    bigram_counts = Counter(bigrams)
    unigram_counts = Counter(tokens)

    pair_count = bigram_counts[(prev_word.lower(), word.lower())]
    prev_word_count = unigram_counts[prev_word.lower()]

    # Number of unique words following prev_word
    T = len(set(w2 for w1, w2 in bigram_counts if w1 == prev_word.lower()))

    if pair_count > 0:
        prob = pair_count / (prev_word_count + T)
    else:
        prob = (T / (prev_word_count + T)) * unigram(word, corpus)

    return prob

corpus = "tom loves nlp tom loves code tom studies linguistics"
prev_word = input('Previous word: ')
word = input('Target word: ')

prob = witten_bell_smoothing(corpus, prev_word, word)
print(f"Witten-Bell smoothing probability P({word}|{prev_word}) = {prob:.4f}")
