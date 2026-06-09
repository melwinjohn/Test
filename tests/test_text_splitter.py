from rag_pipeline.text_splitter import merge_sentences, split_into_sentences


def test_split_into_sentences():
    text = "First sentence. Second sentence! Third one?"
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "First sentence."
    assert sentences[1] == "Second sentence!"
    assert sentences[2] == "Third one?"


def test_merge_sentences_respects_max_chars():
    sentences = ["aaa", "bbb", "cccccccc"]
    merged = merge_sentences(sentences, max_chars=7)
    assert merged == ["aaa bbb", "cccccccc"]
