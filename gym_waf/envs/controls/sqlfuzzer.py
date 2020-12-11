"""Strategies and fuzzer class module"""

from numpy import random
from functools import wraps
import re
from gym_waf.envs.controls.fuzz_utils import (
    replace_random,
    filter_candidates,
    random_string,
    num_tautology,
    string_tautology,
    num_contradiction,
    string_contradiction,
)


def reset_inline_comments(payload: str, seed=None):
    """Remove randomly chosen multi-line comment content.
    Arguments:
        payload: query payload string

    Returns:
        str: payload modified
    """
    rng = random.RandomState(seed)

    positions = list(re.finditer(r"/\*[^(/\*|\*/)]*\*/", payload))

    if not positions:
        return payload

    pos = rng.choice(positions).span()

    replacements = ["/**/"]

    replacement = rng.choice(replacements)

    new_payload = payload[: pos[0]] + replacement + payload[pos[1] :]

    return new_payload


def logical_invariant(payload, seed=None):
    """logical_invariant

    Adds an invariant boolean condition to the payload

    E.g., something OR False


    :param payload:
    """
    rng = random.RandomState(seed)

    pos = re.search("(#|-- )", payload)

    if not pos:
        # No comments found
        return payload

    pos = pos.start()

    replacement = rng.choice(
        [
            # AND True
            " AND 1",
            " AND True",
            " AND " + num_tautology(),
            " AND " + string_tautology(),
            # OR False
            " OR 0",
            " OR False",
            " OR " + num_contradiction(),
            " OR " + string_contradiction(),
        ]
    )

    new_payload = payload[:pos] + replacement + payload[pos:]

    return new_payload


def change_tautologies(payload, seed=None):
    rng = random.RandomState(seed)

    results = list(re.finditer(r'((?<=[^\'"\d\wx])\d+(?=[^\'"\d\wx]))=\1', payload))
    if not results:
        return payload
    candidate = rng.choice(results)

    replacements = [num_tautology(), string_tautology()]

    replacement = rng.choice(replacements)

    new_payload = (
        payload[: candidate.span()[0]] + replacement + payload[candidate.span()[1] :]
    )

    return new_payload


def spaces_to_comments(payload, seed=None):
    rng = random.RandomState(seed)

    # TODO: make it selectable (can be mixed with other strategies)
    symbols = {" ": ["/**/"], "/**/": [" "]}

    symbols_in_payload = filter_candidates(symbols, payload)

    if not symbols_in_payload:
        return payload

    # Randomly choose symbol
    candidate_symbol = rng.choice(symbols_in_payload)
    # Check for possible replacements
    replacements = symbols[candidate_symbol]
    # Choose one replacement randomly
    candidate_replacement = rng.choice(replacements)

    # Apply mutation at one random occurrence in the payload
    return replace_random(payload, candidate_symbol, candidate_replacement)


def spaces_to_whitespaces_alternatives(payload, seed=None):
    rng = random.RandomState(seed)

    symbols = {
        " ": ["\t", "\n", "\f", "\v", "\xa0"],
        "\t": [" ", "\n", "\f", "\v", "\xa0"],
        "\n": ["\t", " ", "\f", "\v", "\xa0"],
        "\f": ["\t", "\n", " ", "\v", "\xa0"],
        "\v": ["\t", "\n", "\f", " ", "\xa0"],
        "\xa0": ["\t", "\n", "\f", "\v", " "],
    }

    symbols_in_payload = filter_candidates(symbols, payload)

    if not symbols_in_payload:
        return payload

    # Randomly choose symbol
    candidate_symbol = rng.choice(symbols_in_payload)
    # Check for possible replacements
    replacements = symbols[candidate_symbol]
    # Choose one replacement randomly
    candidate_replacement = rng.choice(replacements)

    # Apply mutation at one random occurrence in the payload
    return replace_random(payload, candidate_symbol, candidate_replacement)


def random_case(payload, seed=None):
    rng = random.RandomState(seed)

    new_payload = []

    for c in payload:
        if rng.random() > 0.5:
            c = c.swapcase()
        new_payload.append(c)

    return "".join(new_payload)


def comment_rewriting(payload, seed=None):
    rng = random.RandomState(seed)

    p = rng.random()

    if p < 0.5 and ("#" in payload or "-- " in payload):
        return payload + random_string(2)
    elif p >= 0.5 and ("*/" in payload):
        return replace_random(payload, "*/", random_string() + "*/")
    else:
        return payload


def swap_int_repr(payload, seed=None):
    rng = random.RandomState(seed)

    candidates = list(re.finditer(r'(?<=[^\'"\d\wx])\d+(?=[^\'"\d\wx])', payload))

    if not candidates:
        return payload

    candidate_pos = rng.choice(candidates).span()

    candidate = payload[candidate_pos[0] : candidate_pos[1]]

    replacements = [
        hex(int(candidate)),
        "(SELECT {})".format(candidate),
        # "({})".format(candidate),
    ]

    replacement = rng.choice(replacements)

    return payload[: candidate_pos[0]] + replacement + payload[candidate_pos[1] :]


def swap_keywords(payload, seed=None):
    rng = random.RandomState(seed)

    symbols = {
        # OR
        "||": [" OR ", " || "],
        " || ": [" OR ", "||"],
        "OR": [" OR ", "||"],
        "  OR  ": [" OR ", "||", " || "],
        # AND
        "&&": [" AND ", " && "],
        " && ": ["AND", " AND ", " && "],
        "AND": [" AND ", "&&", " && "],
        "  AND  ": [" AND ", "&&"],
        # Not equals
        "<>": ["!=", " NOT LIKE "],
        "!=": [" != ", "<>", " <> ", " NOT LIKE "],
        # Equals
        " = ": [" LIKE ", "="],
        "LIKE": [" LIKE ", "="],
    }

    symbols_in_payload = filter_candidates(symbols, payload)

    if not symbols_in_payload:
        return payload

    # Randomly choose symbol
    candidate_symbol = rng.choice(symbols_in_payload)
    # Check for possible replacements
    replacements = symbols[candidate_symbol]
    # Choose one replacement randomly
    candidate_replacement = rng.choice(replacements)

    # Apply mutation at one random occurrence in the payload
    return replace_random(payload, candidate_symbol, candidate_replacement)


strategies = [
    spaces_to_comments,
    random_case,
    swap_keywords,
    swap_int_repr,
    spaces_to_whitespaces_alternatives,
    comment_rewriting,
    change_tautologies,
    logical_invariant,
    reset_inline_comments,
]
