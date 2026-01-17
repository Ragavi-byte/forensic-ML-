from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)
