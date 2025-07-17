import pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
import math

from google.colab import files

uploaded = files.upload()

# Load data
dt = pd.read_csv('spam.csv', encoding='latin-1')
x_train, x_test, y_train, y_test = train_test_split(dt['v2'], dt['v1'], test_size=0.2, random_state=42)
train_data = [*zip(x_train, y_train)]

class Multinomial():
    def __init__(self):
        self.class_prob = {}
        self.word_prob = {}
        self.vocab = set()

    def train(self, data):
        class_count = Counter()
        word_count = defaultdict(Counter)

        # FIXED: Process all data first, then calculate probabilities
        for text, label in data:
            class_count[label] += 1  # FIXED: Use actual variable, not string 'label'

            for word in text.split():
                word_count[label][word] += 1  # FIXED: Use actual variables
                self.vocab.add(word)

        # FIXED: Move probability calculations outside the loop
        total_docs = sum(class_count.values())
        self.class_prob = {cls: count / total_docs for cls, count in class_count.items()}

        self.word_prob = {}
        for cls in class_count:
            total_words = sum(word_count[cls].values())
            # FIXED: Create dictionary with word:probability pairs, not a set
            self.word_prob[cls] = {
                word: (word_count[cls][word] + 1) / (total_words + len(self.vocab))
                for word in self.vocab
            }

    def predict(self, text):
        class_scores = {cls: math.log(prob) for cls, prob in self.class_prob.items()}

        for word in text.split():
            if word in self.vocab:
                for cls in self.class_prob:

                    class_scores[cls] += math.log(self.word_prob[cls][word])
            else:
                for cls in self.class_prob:

                    total_words = sum(self.word_prob[cls].values())
                    class_scores[cls] += math.log(1 / (total_words + len(self.vocab)))

        return max(class_scores, key=class_scores.get)

# Create and train the model
nb = Multinomial()
nb.train(train_data)

predictions = [nb.predict(email) for email in x_test]

accuracy = sum(pred == actual for pred, actual in zip(predictions, y_test)) / len(y_test)
print(f"Accuracy: {accuracy:.2%}")

for email, actual, pred in zip(x_test[:5], y_test[:5], predictions[:5]):
    print(f"email: {email[:60]}")
    print(f"actual: {'spam' if actual == 1 else 'not spam' }, predicted: {'spam' if pred==1 else 'not spam'}\n")

#measuring the precison and recall

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions, target_names =["not spam", "spam"] ))
