from collections import defaultdict, Counter

corpus = ["tom loves nlp",
          "tom loves linguistics", 
          "tom studies nlp"]

class KneserNey_smoothing:
    def __init__(self, corpus, discount=0.75):
        self.discount = discount
        self.bigram = Counter()
        self.unigram = Counter()
        self.continuation_count = Counter()
        self.unique_continuation = defaultdict(set)

        self.train(corpus)

    def train(self, corpus):
        for sentence in corpus:
            words = sentence.split()
            for i in range(len(words)-1):
                self.bigram[(words[i], words[i+1])] += 1
                self.unigram[words[i]] += 1
                self.unique_continuation[words[i]].add(words[i+1])
                
                # For continuation counts - count unique contexts where word appears
                self.continuation_count[words[i+1]] += 1

            # Last word unigram
            self.unigram[words[-1]] += 1

        self.total_bigrams = len(self.bigram)

    def continuation_prob(self, word):
        # Number of unique contexts where word appears / total unique bigram types
        return self.continuation_count[word] / self.total_bigrams if self.total_bigrams > 0 else 0

    def bigram_prob(self, prev_word, word):
        bigram_count = self.bigram[(prev_word, word)]
        unigram_count = self.unigram[prev_word]
        
        if unigram_count == 0:
            return 0

        # If the bigram was seen
        if bigram_count > 0:
            prob = max(bigram_count - self.discount, 0) / unigram_count
        else:
            prob = 0

        # Lambda weight (interpolation coefficient)
        lambda_weight = (self.discount * len(self.unique_continuation[prev_word])) / unigram_count
        prob += lambda_weight * self.continuation_prob(word)

        return prob

# Create instance
kn = KneserNey_smoothing(corpus)

# Test the model
print("P(nlp|loves) =", kn.bigram_prob("loves", "nlp"))
print("P(code|loves) =", kn.bigram_prob("loves", "code"))
print("P(linguistics|studies) =", kn.bigram_prob("studies", "linguistics"))

# Unseen bigram
print("P(runs|tom) =", kn.bigram_prob("tom", "runs"))

# Interactive part
prev_word = input('Previous word: ')
word = input('Target word: ')
print(f"P({word}|{prev_word}) =", kn.bigram_prob(prev_word, word))
