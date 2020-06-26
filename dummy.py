from sklearn import datasets
from sklearn import dummy
import sklearn
import numpy as np

digits = datasets.load_digits()

dummy_clf = dummy.DummyClassifier(strategy="uniform")
dummy_clf.fit(digits.data[:int(0.8 * len(digits.data))], digits.target[:int(0.8 * len(digits.data))])

# Get the accuracy
results = dummy_clf.predict(digits.data[int(0.8 * len(digits.data)):])
results = digits.target[int(0.8 * len(digits.data)):]
print(sum([1 for i in range(len(results)) if results[i] == digits.target[int(0.8 * len(digits.data)) + i]]) / len(results))
