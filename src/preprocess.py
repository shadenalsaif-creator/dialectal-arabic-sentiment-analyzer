"""Text preprocessing utilities for dialectal Arabic.

CAMeLBERT-DA was pre-trained on dialectal Arabic tweets, so we keep
preprocessing intentionally LIGHT: aggressive normalization (e.g. removing
emojis or diacritics the model has seen during pre-training) can hurt
performance. We only remove clear noise (URLs, mentions, tatweel) and
normalize characters that are pure orthographic variation.
"""

import re

# Compiled once at import time for speed on batch jobs.
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_MENTION_RE = re.compile(r"@\w+")
_TATWEEL_RE = re.compile("\u0640+")          # ـــ elongation character
_REPEAT_RE = re.compile(r"(.)\1{3,}")         # حلوووووو -> حلوو
_WHITESPACE_RE = re.compile(r"\s+")

# Orthographic normalization map (variation, not meaning).
_CHAR_MAP = str.maketrans({
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ى": "ي",
})


def clean_text(text: str) -> str:
    """Light cleaning pipeline for a single dialectal Arabic string.

    Steps:
        1. Remove URLs and @mentions (noise, no sentiment signal).
        2. Remove tatweel (كلمةـــ -> كلمة).
        3. Collapse character floods to 2 repeats (keeps emphasis signal).
        4. Normalize alef/yaa orthographic variants.
        5. Collapse whitespace.

    Hashtag WORDS are kept (only '#' is stripped) because hashtags in
    Gulf social media often carry the actual opinion (#خايس).
    Emojis are kept on purpose -- they are strong sentiment features.
    """
    text = str(text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = _TATWEEL_RE.sub("", text)
    text = _REPEAT_RE.sub(r"\1\1", text)
    text = text.translate(_CHAR_MAP)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def is_arabic(text: str, min_ratio: float = 0.5) -> bool:
    """Return True if at least `min_ratio` of alphabetic chars are Arabic.

    Used to warn the user when input is mostly non-Arabic, where the
    model's prediction would not be meaningful.
    """
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return False
    arabic = sum("\u0600" <= c <= "\u06FF" for c in letters)
    return arabic / len(letters) >= min_ratio
