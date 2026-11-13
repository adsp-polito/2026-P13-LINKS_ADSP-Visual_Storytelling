"""VIST tests. Only the placeholder helper exists so far – Parsa adds the rest with the loader."""

from p13.data.vist import rewrite_placeholders


def test_placeholders_rewritten():
    out = rewrite_placeholders("[male] and [female] went to [location] .")
    assert out == "a man and a woman went to a place ."
    assert "[" not in out


def test_repeated_placeholder_uses_definite_article():
    out = rewrite_placeholders("[male] arrived . [male] smiled at [organization] .")
    assert out == "a man arrived . the man smiled at an organization ."


def test_case_insensitive():
    assert rewrite_placeholders("[MALE] ran") == "a man ran"
