from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords
import io


patterns = "[!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()


def lemmatize(doc):
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    for token in doc.split():
        if token and token not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]

            tokens.append(token)
    if len(tokens) > 0:
        return ' '.join(tokens)
    return None


def classificate(text: str, ksr: pd.DataFrame, num: int):
    text = lemmatize(text)
    tfidf = TfidfVectorizer()
    mx_tf = tfidf.fit_transform(ksr['name_lemm'])
    new_entry = tfidf.transform([text])

    cosine_similarities = cosine_similarity(new_entry, mx_tf).flatten()
    matches = ksr[['code', 'name', 'unit']].copy()
    matches['possibility'] = cosine_similarities
    matches = matches.sort_values(by=['possibility'], ascending=False).drop_duplicates(subset=['possibility'])
    matches = matches[:num]

    return matches.values.tolist()


def classificate_file(df: pd.DataFrame, ksr: pd.DataFrame):
    tfidf = TfidfVectorizer()
    mx_tf = tfidf.fit_transform(ksr['name_lemm'])
    try:
        new_df = pd.DataFrame(columns=['code', 'record_name', 'ref_name'])
        for index, row in df.iterrows():
            new_entry = tfidf.transform([row[0]])
            cosine_similarities = cosine_similarity(new_entry, mx_tf).flatten()
            matches = ksr[['code', 'name']].copy()
            matches['possibility'] = cosine_similarities
            matches = matches.sort_values(by=['possibility'], ascending=False).drop_duplicates(subset=['possibility'])
            matches = matches.values.tolist()[0]
            new_df.loc[len(new_df)] = [matches[0], row[0], matches[1]]
        
        csv_data = new_df.to_csv(index=False)
        return csv_data
    except Exception as e:
        return None
