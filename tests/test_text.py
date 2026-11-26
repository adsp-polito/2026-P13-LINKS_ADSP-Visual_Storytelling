"""Tests for p13.data.text.detokenize (undo PTB tokenisation in story text)."""

from __future__ import annotations

import pytest

from p13.data.text import detokenize


@pytest.mark.parametrize(
    ("tokenised", "expected"),
    [
        # Rule 1: contractions and possessives.
        ("he does n't know .", "he doesn't know."),
        ("she has n't left .", "she hasn't left."),
        ("i ca n't and i wo n't .", "i can't and i won't."),
        ("the man 's wife .", "the man's wife."),
        (
            "they 're here , we 'll go , i 've seen , i 'm sure , he 'd stay .",
            "they're here, we'll go, i've seen, i'm sure, he'd stay.",
        ),
        # Rule 2: punctuation and brackets.
        ("wait ! what ? yes ; no : maybe .", "wait! what? yes; no: maybe."),
        ("the boss ( a man ) left .", "the boss (a man) left."),
        ("it was ... strange .", "it was... strange."),
        # Rule 3: words split in two.
        (
            "we 're gon na win , i wan na go , you got ta see .",
            "we're gonna win, i wanna go, you gotta see.",
        ),
        ("he can not stay .", "he cannot stay."),
    ],
)
def test_detokenize_rules(tokenised: str, expected: str) -> None:
    assert detokenize(tokenised) == expected


def test_two_spaces_between_shots_are_kept():
    assert detokenize("a man ran .  the man fell .") == "a man ran.  the man fell."


def test_quotes_are_left_alone():
    # Opening and closing quotes cannot be told apart; a lone ' may end a plural possessive.
    assert detokenize("he sang ' happy birthday ' .") == "he sang ' happy birthday '."
    assert detokenize("their horses ' hooves .") == "their horses ' hooves."
    assert detokenize('she said " no " .') == 'she said " no ".'


def test_words_that_only_look_like_pieces_are_kept():
    assert detokenize("the nation dances .") == "the nation dances."
    assert detokenize("i can nottake .") == "i can nottake."
    assert detokenize("a canopy , not a tent .") == "a canopy, not a tent."


def test_detokenize_twice_changes_nothing():
    once = detokenize("he does n't know the man 's name .  it 's ok , ( really ) .")
    assert once == "he doesn't know the man's name.  it's ok, (really)."
    assert detokenize(once) == once
