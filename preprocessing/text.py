"""Transparent count / relative TF / smoothed IDF; no labels or stop-word list."""
import re
import math
from collections import Counter
import numpy as np

TOKEN_PATTERN = r"[a-z]+(?:'[a-z]+)?"

def tokens(document):
    return re.findall(TOKEN_PATTERN, document.lower())

def fit(documents, references):
    counts = [Counter(tokens(d)) for d in documents]
    vocabulary = sorted(set().union(*(set(c) for c in counts)))
    if not vocabulary:
        raise ValueError("Reference collection has no English word tokens.")
    if len(vocabulary) > 2000:
        raise ValueError("Reference vocabulary exceeds the 2,000-term limit.")
    df = [sum(term in c for c in counts) for term in vocabulary]
    n = len(documents)
    return dict(schema_version=1, token_pattern=TOKEN_PATTERN, lowercase=True,
                vocabulary=vocabulary, document_frequency=df,
                idf=[math.log((1+n)/(1+d))+1 for d in df], reference_count=n,
                reference_sources=references, tf="count / all document tokens",
                normalization="L2 of TF-IDF across frozen vocabulary; zero vectors stay zero",
                stop_words=None)

def validate(model):
    if model.get("token_pattern") != TOKEN_PATTERN or model.get("schema_version") != 1:
        raise ValueError("Unsupported text reference settings.")
    if (model.get('lowercase') is not True or model.get('stop_words') is not None
            or model.get('tf') != 'count / all document tokens'
            or model.get('normalization') != 'L2 of TF-IDF across frozen vocabulary; zero vectors stay zero'):
        raise ValueError('Unsupported text transformation rules.')
    v = model.get("vocabulary", [])
    if not v or len(v)>2000 or len(set(v)) != len(v) or any(not isinstance(t,str) or not re.fullmatch(TOKEN_PATTERN,t) for t in v):
        raise ValueError("Invalid frozen vocabulary.")
    n=model.get("reference_count",0)
    df=model.get("document_frequency",[])
    idf=model.get("idf",[])
    if not isinstance(n,int) or n<1 or len(df)!=len(v) or len(idf)!=len(v):
        raise ValueError("Invalid reference dimensions.")
    if not isinstance(model.get('reference_sources'),list) or len(model['reference_sources'])!=n:
        raise ValueError('Reference source list must match the fit collection size.')
    if any(not isinstance(d,int) or not 1<=d<=n for d in df):
        raise ValueError("Invalid document frequencies.")
    if not np.allclose(idf,[math.log((1+n)/(1+d))+1 for d in df]):
        raise ValueError("IDF does not match the documented formula.")
    return model

def transform(document, model):
    c=Counter(tokens(document)); total=sum(c.values()); v=model['vocabulary']
    counts=np.array([c[t] for t in v]); tf=counts/max(total,1)
    tfidf=tf*np.asarray(model['idf']); norm=np.linalg.norm(tfidf)
    return dict(count=counts, tf=tf, tfidf=tfidf, normalized=tfidf/norm if norm else tfidf,
                token_count=total, unknown_token_count=sum(n for t,n in c.items() if t not in set(v)))
