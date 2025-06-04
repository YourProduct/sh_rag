import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag import load_docs, build_index, search


def test_retrieval():
    docs, names = load_docs('docs')
    vectorizer, embeddings = build_index(docs)
    results = search('retrieval', vectorizer, embeddings, docs, top_k=1)
    assert results
