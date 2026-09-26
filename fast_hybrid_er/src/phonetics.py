"""
High-Performance Phonetic Encoding & Blocking Super-Key Module.
Implements Soundex, Metaphone, Consonant Skeleton, and Phonetic Similarity Metrics.
Zero external library dependencies (pure Python, highly optimized).
"""

import re
from typing import List, Set, Tuple

SOUNDEX_MAP = {
    'B': '1', 'F': '1', 'P': '1', 'V': '1',
    'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
    'D': '3', 'T': '3',
    'L': '4',
    'M': '5', 'N': '5',
    'R': '6'
}

COMMON_SUFFIX_STOPWORDS = {
    "PVT", "LTD", "LIMITED", "PRIVATE", "INC", "INCORPORATED", "CORP", "CORPORATION",
    "CO", "COMPANY", "LLC", "LLP", "SA", "SAS", "SARL", "GMBH", "STORE", "SHOP", "ENTERPRISE",
    "ENTERPRISES", "SERVICES", "SOLUTIONS", "INDUSTRIES", "TRADING", "TRADERS"
}


def soundex(word: str) -> str:
    """Computes standard American Soundex code (e.g. 'Robert' -> 'R163')."""
    if not word:
        return ""
    clean = re.sub(r'[^A-Za-z]', '', word.upper())
    if not clean:
        return ""

    first_char = clean[0]
    tail = clean[1:]

    encoded = []
    prev = SOUNDEX_MAP.get(first_char, '')

    for char in tail:
        code = SOUNDEX_MAP.get(char, '')
        if code and code != prev:
            encoded.append(code)
            prev = code
        elif not code and char not in ('H', 'W'):
            prev = ''

    res = first_char + ''.join(encoded)
    res = res.replace('0', '')
    res = (res + '0000')[:4]
    return res


def metaphone_key(word: str) -> str:
    """Fast simplified Metaphone phonetic key algorithm."""
    if not word:
        return ""
    w = re.sub(r'[^A-Z]', '', word.upper())
    if not w:
        return ""

    # Drop initial silent consonants
    if w.startswith(('KN', 'GN', 'PN', 'WR', 'PS')):
        w = w[1:]
    elif w.startswith('X'):
        w = 'S' + w[1:]

    res = []
    i = 0
    n = len(w)

    while i < n:
        c = w[i]
        nxt = w[i+1] if i+1 < n else ''

        # Skip duplicates except 'C'
        if i > 0 and c == w[i-1] and c != 'C':
            i += 1
            continue

        if c in 'AEIOU':
            if i == 0:
                res.append(c)
        elif c == 'B':
            if not (i == n-1 and i > 0 and w[i-1] == 'M'):
                res.append('B')
        elif c == 'C':
            if nxt in ('I', 'E', 'Y'):
                res.append('S')
            elif nxt == 'H':
                res.append('X')
                i += 1
            else:
                res.append('K')
        elif c == 'D':
            if nxt == 'G' and i+2 < n and w[i+2] in ('E', 'I', 'Y'):
                res.append('J')
                i += 1
            else:
                res.append('T')
        elif c in ('F', 'V'):
            res.append('F')
        elif c == 'G':
            if nxt == 'H':
                i += 1
            elif nxt in ('E', 'I', 'Y'):
                res.append('J')
            else:
                res.append('K')
        elif c == 'H':
            if i == 0 or w[i-1] in 'AEIOU':
                if nxt in 'AEIOU':
                    res.append('H')
        elif c in ('J', 'Y'):
            res.append('J')
        elif c == 'K':
            res.append('K')
        elif c == 'L':
            res.append('L')
        elif c == 'M':
            res.append('M')
        elif c == 'N':
            res.append('N')
        elif c == 'P':
            if nxt == 'H':
                res.append('F')
                i += 1
            else:
                res.append('P')
        elif c == 'Q':
            res.append('K')
        elif c == 'R':
            res.append('R')
        elif c == 'S':
            if nxt == 'H':
                res.append('X')
                i += 1
            else:
                res.append('S')
        elif c == 'T':
            if nxt == 'H':
                res.append('0')
                i += 1
            elif nxt == 'I' and i+2 < n and w[i+2] in ('A', 'O'):
                res.append('X')
                i += 1
            else:
                res.append('T')
        elif c == 'W':
            if nxt in 'AEIOU':
                res.append('W')
        elif c == 'X':
            res.append('KS')
        elif c == 'Z':
            res.append('S')

        i += 1

    return ''.join(res)[:6]


def get_consonant_skeleton(word: str) -> str:
    """Extracts first 5 consonants of a word, creating a phonetic skeleton."""
    if not word:
        return ""
    w = re.sub(r'[^A-Z]', '', word.upper())
    if not w:
        return ""
    first = w[0]
    tail = re.sub(r'[AEIOUYHW]', '', w[1:])
    return (first + tail)[:5]


def get_phonetic_fingerprint(text: str) -> str:
    """Generates space-separated sequence of metaphone keys for tokens in text."""
    if not text:
        return ""
    tokens = text.split()
    keys = [metaphone_key(t) for t in tokens if len(t) >= 2]
    return " ".join([k for k in keys if k])


def generate_phonetic_blocking_keys(text: str) -> List[str]:
    """
    Generates multiple complementary phonetic blocking keys:
    1. First token Metaphone key (e.g. 'M123')
    2. First token Soundex key (e.g. 'M120')
    3. Most salient non-stopword Metaphone key
    4. Consonant skeleton key
    """
    if not text:
        return []

    tokens = [re.sub(r'[^A-Za-z]', '', t.upper()) for t in text.split()]
    tokens = [t for t in tokens if len(t) >= 2]
    if not tokens:
        return []

    keys = set()
    # 1. Primary token keys
    first_tok = tokens[0]
    m_first = metaphone_key(first_tok)
    s_first = soundex(first_tok)
    c_first = get_consonant_skeleton(first_tok)

    if m_first:
        keys.add(f"M:{m_first}")
    if s_first:
        keys.add(f"S:{s_first}")
    if c_first:
        keys.add(f"C:{c_first}")

    # 2. Salient non-stopword token (if first token was a generic prefix or stopword)
    content_tokens = [t for t in tokens if t not in COMMON_SUFFIX_STOPWORDS]
    if content_tokens and content_tokens[0] != first_tok:
        salient_tok = content_tokens[0]
        m_salient = metaphone_key(salient_tok)
        if m_salient:
            keys.add(f"MS:{m_salient}")

    return sorted(list(keys))


def phonetic_token_jaccard(text1: str, text2: str) -> float:
    """Computes Jaccard similarity across metaphone keys of tokens in two strings."""
    if not text1 or not text2:
        return 0.0
    set1 = set(metaphone_key(t) for t in text1.split() if len(t) >= 2)
    set2 = set(metaphone_key(t) for t in text2.split() if len(t) >= 2)
    set1.discard("")
    set2.discard("")
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return float(intersection / union) if union > 0 else 0.0
