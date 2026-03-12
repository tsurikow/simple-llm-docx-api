import math
from itertools import product

import numpy as np
from rank_bm25 import BM25Okapi
from razdel import tokenize

_DENSE_WEIGHT = 0.8
_BM25_WEIGHT = 0.2
_METADATA_TOP_WINDOW = 4
_REFERENCE_NUMBER_WINDOW = 3
_MONTH_STEMS = (
    "январ",
    "феврал",
    "март",
    "апрел",
    "ма",
    "июн",
    "июл",
    "август",
    "сентябр",
    "октябр",
    "ноябр",
    "декабр",
)
_STOP_TOKENS = {"от", "к", "по", "на"}


def tokenize_text(text: str) -> list[str]:
    tokens: list[str] = []
    for item in tokenize(text):
        token = item.text.lower().strip("«»\"'()[]{}.,:;!?")
        if token:
            tokens.append(token)
    return tokens


def is_contract_metadata_question(question: str) -> bool:
    q = question.lower()
    return "договор" in q and any(token in q or token in question for token in ("№", "номер", "дат", "реквизит"))


def score_top_indices(
    chunks: list[str],
    vectors: np.ndarray,
    question_vector: np.ndarray,
    user_question: str,
    top_k: int,
) -> np.ndarray:
    dense_scores = vectors @ question_vector
    bm25_scores = _bm25_scores(chunks=chunks, query=user_question)
    scores = _hybrid_scores(dense_scores=dense_scores, bm25_scores=bm25_scores)
    if not is_contract_metadata_question(user_question):
        return np.argsort(scores)[::-1][:top_k]

    reference_number = _extract_reference_number(chunks[: min(len(chunks), _REFERENCE_NUMBER_WINDOW)])
    reranked = sorted(
        range(len(chunks)),
        key=lambda idx: scores[idx] + _metadata_score(chunks[idx], idx, reference_number),
        reverse=True,
    )
    selected = [0]
    for idx in reranked:
        if idx != 0:
            selected.append(idx)
        if len(selected) >= top_k:
            break
    return np.asarray(selected[:top_k], dtype=int)


def _bm25_scores(chunks: list[str], query: str) -> np.ndarray:
    query_tokens = tokenize_text(query)
    if not query_tokens:
        return np.zeros(len(chunks), dtype=np.float32)
    scores = BM25Okapi([tokenize_text(chunk) for chunk in chunks]).get_scores(query_tokens)
    return np.asarray(scores, dtype=np.float32)


def _hybrid_scores(dense_scores: np.ndarray, bm25_scores: np.ndarray) -> np.ndarray:
    return _DENSE_WEIGHT * _normalize_scores(dense_scores) + _BM25_WEIGHT * _normalize_scores(bm25_scores)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if math.isclose(min_score, max_score):
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - min_score) / (max_score - min_score)).astype(np.float32)


def _metadata_score(chunk: str, index: int, reference_number: str | None) -> float:
    tokens = tokenize_text(chunk)
    has_contract = any("договор" in token for token in tokens)
    has_number = "№" in tokens or any("номер" in token for token in tokens)
    date_positions = _date_positions(tokens)
    has_date = bool(date_positions)
    cluster_bonus = _nearby_metadata_bonus(tokens, date_positions)

    score = 0.0
    if has_contract and has_number:
        score += 0.35
    if has_date:
        score += 0.2
    if has_contract and has_number and has_date:
        score += 0.25
    score += cluster_bonus + _reference_number_bonus(tokens, reference_number)

    if any("приложен" in token for token in tokens) and cluster_bonus < 0.5:
        score -= 0.1
    if index == 0:
        score += 0.3
    elif index < _METADATA_TOP_WINDOW:
        score += 0.08 * (_METADATA_TOP_WINDOW - index)
    return score


def _nearby_metadata_bonus(tokens: list[str], date_positions: list[int]) -> float:
    contract_positions = [i for i, token in enumerate(tokens) if "договор" in token]
    number_positions = [i for i, token in enumerate(tokens) if token == "№"]
    from_positions = [i for i, token in enumerate(tokens) if token == "от"]

    score = 0.0
    if contract_positions and number_positions:
        score += 0.12
    if _min_span(contract_positions, number_positions, from_positions) <= 6:
        score += 0.18
    if _min_span(contract_positions, number_positions, from_positions, date_positions) <= 8:
        score += 0.28
    return score


def _date_positions(tokens: list[str]) -> list[int]:
    positions = [index for index, token in enumerate(tokens) if _is_numeric_date_token(token)]
    for index in range(len(tokens) - 2):
        if _is_day_token(tokens[index]) and _is_month_token(tokens[index + 1]) and _is_year_token(tokens[index + 2]):
            positions.append(index)
    return positions


def _extract_reference_number(chunks: list[str]) -> str | None:
    best_candidate: str | None = None
    best_score = -1.0
    for chunk in chunks:
        for candidate in _extract_number_candidates(tokenize_text(chunk)):
            score = float(len(_number_parts(candidate)))
            if score > best_score:
                best_candidate = candidate
                best_score = score
    return best_candidate


def _extract_number_candidates(tokens: list[str]) -> list[str]:
    candidates: list[str] = []
    for index, token in enumerate(tokens):
        if token != "№":
            continue
        parts: list[str] = []
        for next_token in tokens[index + 1 : index + 6]:
            if next_token in _STOP_TOKENS:
                break
            if next_token in {"/", "-"}:
                if parts:
                    parts.append(next_token)
                continue
            if not _looks_like_number_token(next_token):
                break
            parts.append(next_token)
        if parts:
            candidates.append("".join(parts))
    return candidates


def _reference_number_bonus(tokens: list[str], reference_number: str | None) -> float:
    if not reference_number:
        return 0.0
    reference_parts = set(_number_parts(reference_number))
    if not reference_parts:
        return 0.0

    best_ratio = 0.0
    normalized_reference = _normalize_number(reference_number)
    for candidate in _extract_number_candidates(tokens):
        candidate_parts = set(_number_parts(candidate))
        if not candidate_parts:
            continue
        ratio = len(reference_parts & candidate_parts) / len(reference_parts)
        if _normalize_number(candidate) == normalized_reference:
            ratio = 1.0
        best_ratio = max(best_ratio, ratio)

    if best_ratio >= 1.0:
        return 0.35
    if best_ratio >= 0.6:
        return 0.2
    if best_ratio >= 0.3:
        return 0.1
    return 0.0


def _min_span(*groups: list[int]) -> int:
    if any(not group for group in groups):
        return 10**6
    return min(max(combo) - min(combo) for combo in product(*groups))


def _is_numeric_date_token(token: str) -> bool:
    if not any(sep in token for sep in (".", "/", "-")):
        return False
    parts = [part for part in token.replace("/", ".").replace("-", ".").split(".") if part]
    return len(parts) == 3 and all(part.isdigit() for part in parts) and len(parts[0]) <= 2 and len(parts[1]) <= 2 and len(parts[2]) in (2, 4)


def _is_day_token(token: str) -> bool:
    return token.isdigit() and 1 <= len(token) <= 2


def _is_month_token(token: str) -> bool:
    return any(stem in token for stem in _MONTH_STEMS)


def _is_year_token(token: str) -> bool:
    return token.isdigit() and len(token) == 4


def _looks_like_number_token(token: str) -> bool:
    if token in _STOP_TOKENS:
        return False
    has_digit = any(char.isdigit() for char in token)
    has_alpha = any(char.isalpha() for char in token)
    has_joiner = any(char in "-/" for char in token)
    return bool((has_digit and (has_alpha or has_joiner or len(token) >= 2)) or (has_alpha and has_joiner))


def _number_parts(value: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    for char in value.lower():
        if char.isalnum():
            current.append(char)
            continue
        if current:
            part = "".join(current)
            if len(part) >= 2:
                parts.append(part)
            current = []
    if current:
        part = "".join(current)
        if len(part) >= 2:
            parts.append(part)
    return parts


def _normalize_number(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())
