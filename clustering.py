from sklearn.cluster import KMeans
import pandas as pd
from model.vectorizer import Vectorizer

class Clustering:
    @staticmethod
    def cluster_cases(cases_df, text_column='description', n_clusters=3):
        if cases_df.empty:
            cases_df['pattern_cluster'] = []
            return cases_df

        # Fill missing text
        texts = cases_df[text_column].fillna("").astype(str)

        # TF-IDF vectorization
        vec = Vectorizer()
        tfidf_matrix = vec.fit_transform(texts)

        # KMeans clustering
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(tfidf_matrix)

        # Add cluster labels to DataFrame
        cases_df['pattern_cluster'] = labels.tolist()
        return cases_df
