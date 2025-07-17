from collections import Counter, defaultdict
import re

class BPE:
    def __init__(self, vocab_size=50):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}

    def train(self, corpus):
        tokens = corpus.strip().split()
        # Fix here clearly:
        vocab = Counter([' '.join(word) + ' </w>' for word in tokens])

        while len(self.vocab) < self.vocab_size:
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                symbols = word.split()
                for i in range(len(symbols)-1):
                    pairs[symbols[i], symbols[i+1]] += freq
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            pattern = re.escape(' '.join(best_pair))
            replacement = ''.join(best_pair)

            vocab = {re.sub(pattern, replacement, word): freq for word, freq in vocab.items()}
            self.vocab[replacement] = len(self.vocab)

        for word in vocab:
            for token in word.split():
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        self.inverse_vocab = {token: id for token, id in self.vocab.items()}

    def encode(self, text):
        tokens = text.strip().split()
        token_ids = []
        for token in tokens:
            if token in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[token])
            else:
                # Handle unknown tokens with BPE merges
                chars = list(token) + ['</w>']
                while len(chars) > 1:
                    pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
                    pair_strs = [''.join(pair) for pair in pairs]
                    found = False
                    for pair_str in pair_strs:
                        if pair_str in self.inverse_vocab:
                            idx = pair_strs.index(pair_str)
                            chars = chars[:idx] + [pair_str] + chars[idx+2:]
                            found = True
                            break
                    if not found:
                        break
                token_ids.extend([self.inverse_vocab[c] for c in chars if c in self.inverse_vocab])
        return tokens, token_ids

# Usage clearly:
corpus = "BPE is the standard algorithm for NLP"
tokenizer = BPE(vocab_size=10)
tokenizer.train(corpus.lower())

text = "NLP"
tokens, token_ids = tokenizer.encode(text.lower())

print("Tokens:", tokens)
print("Token IDs:", token_ids)
