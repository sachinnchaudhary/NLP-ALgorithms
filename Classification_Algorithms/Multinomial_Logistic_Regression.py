import pandas as pd

!git clone https://huggingface.co/datasets/dair-ai/emotion

train_dt = pd.read_parquet('./emotion/unsplit/train-00000-of-00001.parquet')
test_dt = pd.read_parquet('./emotion/split/test-00000-of-00001.parquet')
validation_dt = pd.read_parquet('./emotion/split/validation-00000-of-00001.parquet')


x_train = train_dt['text']
y_train= train_dt['label']

x_test = test_dt['text']
y_test= test_dt['label']

x_val = validation_dt['text']
y_val= validation_dt['label']


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


vectorizer = TfidfVectorizer(stop_words='english', max_features=40000, ngram_range=(1,3))

x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)
x_val_tfidf = vectorizer.transform(x_val)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(multi_class="multinomial" ,class_weight='balanced',solver = "lbfgs", max_iter=1000)

model.fit(x_train_tfidf, y_train)



from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(x_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4%}")

print(classification_report(y_val, y_pred))

sample_text = [
             "i love code",
             "i love NLP!",
             "i love math",
             "i am happy",
             "i feel angry",

  ]

sample_tfidf = vectorizer.transform(sample_text)

sample_pred = model.predict(sample_tfidf)

for text, pred in zip(sample_text, sample_pred):
    print(f'"{text}" → Predicted Emotion: {pred}')
