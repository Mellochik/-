from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords


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


def classificate(text: str, classes: List[str], num: int):
    text = lemmatize(text)
    tfidf = TfidfVectorizer()
    mx_tf = tfidf.fit_transform(classes)
    new_entry = tfidf.transform([text])

    cosine_similarities = cosine_similarity(new_entry, mx_tf).flatten()
    matches = pd.DataFrame({"name": classes, "possibility": cosine_similarities})
    matches = matches.sort_values(by=['possibility'], ascending=False).drop_duplicates(subset=['possibility'])
    matches = matches[:num]

    return {key: value for key, value in zip(matches['name'], matches['possibility'])}
