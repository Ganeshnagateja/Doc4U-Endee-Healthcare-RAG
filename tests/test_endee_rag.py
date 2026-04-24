from endee_rag import chunk_text, LocalKeywordStore


def test_chunk_text_splits_long_text():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_local_keyword_store_returns_relevant_docs(tmp_path):
    store = LocalKeywordStore(storage_path=str(tmp_path / "store.json"))
    store.add_texts(
        ["Fever and dehydration need hydration and monitoring.", "Vector databases store embeddings."],
        [{"title": "fever", "source": "test"}, {"title": "vector", "source": "test"}],
    )
    results = store.similarity_search_with_score("fever hydration", k=1)
    assert results
    assert results[0][0]["metadata"]["title"] == "fever"
