"""Shared story-text helpers for the dataset loaders.  **Owner: Parsa.**

VIST ``text`` and VWP ``anonymised_story`` are tokenised in the Penn Treebank style:
``"he does n't know the man 's name ."``. :func:`detokenize` turns that back into
normal English (``"he doesn't know the man's name."``), so fine-tuned models do not
learn to write split words. Text stays lower-case (team decision, 26 Nov).
"""

from __future__ import annotations

import re

# One rule per line: (pattern, replacement). Order does not matter: no rule's
# output can be matched by another rule.
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # 1. Contractions and possessives: "does n't" -> "doesn't", "man 's" -> "man's".
    (re.compile(r"(?<=\w) (n't|'s|'re|'ll|'ve|'m|'d)(?!\w)", re.IGNORECASE), r"\1"),
    # 2. No space before punctuation, and none inside brackets.
    (re.compile(r"(?<=\S) (?=[.,!?;:)])"), ""),
    (re.compile(r"\( (?=\S)"), "("),
    # 3. Words PTB splits in two.
    (re.compile(r"\b(gon|wan) na\b", re.IGNORECASE), r"\1na"),
    (re.compile(r"\b(got) ta\b", re.IGNORECASE), r"\1ta"),
    (re.compile(r"\b(can) not\b", re.IGNORECASE), r"\1not"),
)


def detokenize(text: str) -> str:
    """Undo PTB tokenisation in ``text``.

    * Rejoins contractions and ``'s``, removes the space before ``. , ! ? ; :`` and
      inside ``( )``, and rejoins ``gon na`` / ``wan na`` / ``got ta`` / ``can not``.
    * Only single spaces next to these pieces are removed; other spacing (such as the
      two spaces VWP puts between shots) is kept.
    * Quote marks (``'`` or ``"`` on their own) are left alone: an opening and a closing
      quote cannot be told apart safely, and a lone ``'`` can also end a plural
      possessive (``"the horses ' hooves"``).
    """
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text
