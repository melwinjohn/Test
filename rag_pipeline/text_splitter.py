import re


_SENTENCE_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"'(\[])|(?<=[.!?][\"')\]])\s+(?=[A-Z\"'(\[])"
)


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences for semantic chunking."""
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []

    sentences = _SENTENCE_PATTERN.split(normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def merge_sentences(sentences: list[str], max_chars: int) -> list[str]:
    """Merge consecutive sentences without exceeding max_chars."""
    if not sentences:
        return []

    merged: list[str] = []
    current = sentences[0]

    for sentence in sentences[1:]:
        candidate = f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            merged.append(current)
            current = sentence

    merged.append(current)
    return merged
