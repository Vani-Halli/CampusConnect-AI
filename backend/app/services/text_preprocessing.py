import re
import unicodedata


_TOKEN_PATTERN = re.compile(r"(?u)\w+(?:\.\w+)*(?:'\w+)*")
_CURRENCY_NAMES = {
    "$": " usd ",
    "€": " eur ",
    "£": " gbp ",
    "₹": " inr ",
}


def normalize_text(text: object) -> str:
    """Normalize text for matching without dropping numbers or meaningful words."""
    if not isinstance(text, str):
        return ""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    for symbol, currency_name in _CURRENCY_NAMES.items():
        normalized = normalized.replace(symbol, currency_name)
    normalized = normalized.replace("%", " percent ")
    normalized = normalized.replace("’", "'")
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
    normalized = re.sub(r"(?<=\d)\.(?=\d)", "\ue000", normalized)
    normalized = "".join(
        character
        if unicodedata.category(character)[0] in {"L", "M", "N"}
        or character.isspace()
        or character in {"_", "'", "\ue000"}
        else " "
        for character in normalized
    )
    normalized = normalized.replace("\ue000", ".")
    normalized = re.sub(r"(?<!\w)'|'(?!\w)", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize_text(text: str) -> list[str]:
    """Tokenize normalized text, retaining single-letter abbreviations and negations."""
    return _TOKEN_PATTERN.findall(normalize_text(text))
