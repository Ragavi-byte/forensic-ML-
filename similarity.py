from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from model.vectorizer import Vectorizer

class Similarity:
    @staticmethod
    def compute_similarity(cases_df, text_column='description'):
        if cases_df.empty:
            return pd.DataFrame()
        
        # Replace NaN descriptions with empty string
        texts = cases_df[text_column].fillna("").astype(str)

        vec = Vectorizer()
        tfidf_matrix = vec.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
        sim_df = pd.DataFrame(sim_matrix, index=cases_df.index, columns=cases_df.index)
        return sim_df

    @staticmethod
    def get_top_matches(sim_df, top_n=3):
        results = {}
        if sim_df.empty:
            return results
        
        for idx in sim_df.index:
            sorted_idx = sim_df.loc[idx].sort_values(ascending=False).iloc[1:top_n+1].index.tolist()
            results[idx] = sorted_idx
        return results
