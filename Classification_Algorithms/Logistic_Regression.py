import math
import pandas as pd
import numpy as np


dt = pd.read_csv('spam.csv', encoding = "latin-1")

dt = dt[['v2', 'v1']]


dt['label'] = dt['v1'].apply(lambda x: 1 if x == 'spam' else 0)


dt.drop('v1', axis = 1, inplace=True)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#TF-IDF vector

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Split data correctly
x_train, x_test, y_train, y_test = train_test_split(dt['v2'], dt['label'], test_size=0.2, random_state=42)




x_train_vect = vectorizer.fit_transform(x_train)
x_test_vect = vectorizer.transform(x_test)




from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train_vect, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = log_reg.predict(x_test_vect)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2%}")

print(classification_report(y_test, y_pred, target_names=['not spam' , 'spam']))

feature_names = vectorizer.get_feature_names_out()

coeff = log_reg.coef_[0]

top_spam = np.argsort(coeff)[-10:]
top_not_spam = np.argsort(coeff)[:10]


for i in top_spam:

    print(f"feauturs name: {coeff[i]:.3f}")


for i in top_not_spam:

    print(f"feauturs name: {coeff[i]:.3f}")
