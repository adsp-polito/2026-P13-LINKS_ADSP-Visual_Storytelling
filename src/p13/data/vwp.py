"""VWP (Visual Writing Prompts) → unified JSONL.  **Owner: Parsa.**

Source: VWP v2.1 on Hugging Face (``tonyhong/vwp``); see ``data/README.md``.
Expected raw layout (``python data/download_vwp.py --split val`` creates it)::

    data/raw/vwp/vwp_v2.1_{train,val,test}.csv
    data/raw/vwp/images/{train,val}/<image_key>.jpg    e.g. tt0047396_0010_0_shot_0667_img_0.jpg

Pipeline for :func:`prepare`:

1. Read the CSV (:func:`read_csv`): one story per row. Its images are the row's own
   ``link0..k`` in that order (writers picked different shots of the same scene).
2. Story text = ``anonymised_story`` (lower-cased and tokenised, like VIST ``text``),
   with character names the anonymiser missed masked (:func:`mask_leaked_names`: its own
   placeholder for that name if the same story shows it, else "a person") and numbered
   placeholders rewritten (:func:`rewrite_placeholders`), then PTB tokenisation undone
   (``text.detokenize``: "does n't" -> "doesn't", "home ." -> "home.").
3. Drop a whole story if its links do not match ``img_id_list`` and the text segments
   (same count, no gaps),
   if it repeats an earlier story, if it has a bad number of images, no text, or any
   image that is missing or cannot be decoded (``drop_counts`` says why).
4. Resize images to long side ≤ ``cfg.data.max_image_side`` into ``data/images/vwp/<split>/``.
5. Write ``data/processed/vwp_<split>.jsonl`` and, for val, the dev-200 id list with
   at most one story per scene.

The test split is blind (the CSV has images but no stories), so :func:`prepare` refuses it.
"""

from __future__ import annotations

import ast
import csv
import difflib
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from p13.common.config import ExperimentConfig
from p13.common.jsonl import write_jsonl
from p13.common.schema import MAX_IMAGES, MIN_IMAGES, StoryRecord
from p13.common.subset import read_ids, sample_ids, subset_path, write_ids
from p13.data.images import find_image, is_valid_image, resize_long_side
from p13.data.text import detokenize

log = logging.getLogger(__name__)

# Edit here if the team decides on different wording.
NOUNS: dict[str, str] = {
    "male": "man",
    "female": "woman",
    "location": "place",
    "organization": "organization",
    "person": "person",  # character names the VWP anonymiser missed
}
_ORDINALS = (
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
)
_PLACEHOLDER_RE = re.compile(r"\[(male|female|location|organization|person)(\d*)\]", re.IGNORECASE)

# Character names that are also English words in VWP stories (checked on train + val
# v2.1: each one is used as a normal word in stories where it is not a character name).
WORD_NAMES = frozenset(
    {
        "will",
        "guy",
        "judge",
        "rob",
        "art",
        "lake",
        "gay",
        "bill",
        "ward",
        "mark",
        "wolf",
        "nick",
        "ray",
        "dean",
        "grace",
        "frank",
        "dash",
        "marshall",
        "max",
        "mike",
        "billy",
        "scott",
        "mickey",
    }
)

# Image URLs end in <folder>/<shot file>, the folder being tt<imdb>_<scene>[_<n>]. The same
# shot can sit in two scene folders (270 shots in train), so local files are named after
# folder + file (unique for every URL), not after the shot alone.
_URL_RE = re.compile(r"/(tt\d+_[^/]*)/(shot_\d+_img_\d+)\.\w+$")
_EMPTY = ("", "{}")  # how the VWP CSV marks an empty cell
_N_LINKS, _N_CHARS = 10, 5

# Why a whole story can be dropped (keys of the ``drop_counts`` dict).
DROP_REASONS = (
    "link_count_mismatch",
    "duplicate",
    "bad_length",
    "empty_story",
    "missing_image",
    "broken_image",
)


def csv_name(split: str) -> str:
    return f"vwp_v2.1_{split}.csv"


def _ordinal(rank: int) -> str:
    return _ORDINALS[rank] if rank < len(_ORDINALS) else f"no. {rank + 1}"


def rewrite_placeholders(text: str) -> str:
    """Replace numbered VWP placeholders (``[male0]``, ``[female1]`` ...) with noun phrases.

    One character of a kind: "a man", then "the man" (same as VIST).
    Several of a kind: "a man" / "another man" / "a third man" on first mention,
    then "the first man" / "the second man" / "the third man".
    """
    order: dict[str, list[str]] = defaultdict(list)
    for m in _PLACEHOLDER_RE.finditer(text):
        kind, idx = m.group(1).lower(), m.group(2)
        if idx not in order[kind]:
            order[kind].append(idx)
    seen: set[tuple[str, str]] = set()

    def sub(m: re.Match[str]) -> str:
        kind, idx = m.group(1).lower(), m.group(2)
        noun = NOUNS[kind]
        rank = order[kind].index(idx)
        first_mention = (kind, idx) not in seen
        seen.add((kind, idx))
        if first_mention:
            if rank == 0:
                return ("an " if noun[0] in "aeiou" else "a ") + noun
            return f"another {noun}" if rank == 1 else f"a {_ordinal(rank)} {noun}"
        return f"the {_ordinal(rank)} {noun}" if len(order[kind]) > 1 else f"the {noun}"

    return _PLACEHOLDER_RE.sub(sub, text)


def _name_re(name: str) -> re.Pattern[str]:
    """Whole-word, case-insensitive match of a name; a final dot is optional ("T.J.")."""
    core = re.escape(name.rstrip("."))
    return re.compile(rf"(?<!\w){core}\.?(?!\w)", re.IGNORECASE)


# Tokens of a lower-cased story, split the same way the VWP anonymiser splits its text, so
# that both versions of a story line up token by token: placeholders, titles ("dr.") and
# initials ("t.j", "t.j.", "w.") keep their dot; any other word does not ("around." ->
# "around" "."); "hasn't" -> "has" "n't"; clitics ("'s") and punctuation. Straight and
# curly apostrophes are both handled. Only used to line up two versions of a story.
_TOKEN_RE = re.compile(
    r"\[[a-z]+\d*\]"
    r"|(?:dr|mr|mrs|ms)\."
    r"|[a-z](?:\.[a-z])+\.?"
    r"|[a-z]\.(?!\w)"
    r"|[a-z0-9]+(?=n['\u2019]t)"
    r"|n['\u2019]t"
    r"|[a-z0-9]+"
    r"|['\u2019][a-z]+"
    r"|[^\w\s]"
)
_PERSON_PLACEHOLDER_RE = re.compile(r"\[(?:male|female)\d+\]")


def anonymiser_placeholder(name: str, original: str, anonymised: str) -> str | None:
    """The placeholder the VWP anonymiser used for ``name`` elsewhere in the same story.

    The anonymiser sometimes replaces a name in one sentence and misses it in another.
    Lining up the original and the anonymised story word by word (``difflib``) shows which
    ``[maleN]`` / ``[femaleN]`` stands where the original has the name. Returns it only if
    every such spot agrees on one placeholder, else ``None``.
    """
    orig = _TOKEN_RE.findall(original.lower())
    anon = _TOKEN_RE.findall(anonymised.lower())
    forms = {name.lower(), name.lower().rstrip(".")}
    votes: set[str] = set()
    matcher = difflib.SequenceMatcher(None, orig, anon, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace" and i2 - i1 == j2 - j1:
            for o, a in zip(orig[i1:i2], anon[j1:j2], strict=True):
                if o in forms and _PERSON_PLACEHOLDER_RE.fullmatch(a):
                    votes.add(a)
    return votes.pop() if len(votes) == 1 else None


def _sentence_ends(name: str, anonymised: str, original: str) -> list[bool]:
    """For each match of ``name`` in ``anonymised``: does its final dot also end a sentence?

    In "grab T.J. T.J. was", the dot after the first "T.J." ends the sentence too, and the
    name pattern would swallow it. True if the match ends with "." and is followed by two
    spaces (VWP joins the per-shot texts with two spaces) or by the end of the text; or, when
    the name occurs as often in ``original`` as in ``anonymised``, if the same occurrence in
    ``original`` is followed by a capital letter or the end of the text.
    """
    pattern = _name_re(name)
    matches = list(pattern.finditer(anonymised))
    in_original = [
        bool(re.match(r"\s+[A-Z\"']|\s*$", original[m.end() :])) for m in pattern.finditer(original)
    ]
    same_count = len(in_original) == len(matches)
    ends = []
    for k, m in enumerate(matches):
        after = anonymised[m.end() :]
        if not m.group(0).endswith(".") or after.lstrip().startswith("."):
            ends.append(False)  # no dot swallowed, or a full stop is already there
            continue
        ends.append(after.startswith("  ") or not after.strip() or (same_count and in_original[k]))
    return ends


def mask_leaked_names(anonymised: str, names: list[str], original: str) -> str:
    """Replace character names the VWP anonymiser missed.

    * If the same story shows which placeholder the anonymiser used for the name
      (:func:`anonymiser_placeholder`), that placeholder is reused, so the character stays
      one person with the right gender.
    * Otherwise the name becomes ``[personN]`` ("a person").
    * A dot swallowed with a dotted name ("T.J.") is put back when it also ended a sentence.

    A name that is also an English word (``WORD_NAMES``) is replaced only if the original
    story never uses it in lower case; otherwise masking could break a normal word
    ("a guy"), so the name is kept.
    """
    leaked = []
    for name in names:
        if not name.rstrip(".").strip():
            continue  # a name like "." would match every word boundary
        pattern = _name_re(name)
        if not pattern.search(anonymised):
            continue
        low = name.lower().rstrip(".")
        if low in WORD_NAMES and re.search(rf"(?<!\w){re.escape(low)}(?!\w)", original):
            continue
        leaked.append((name, pattern))
    # Longer names first, so "Alan" is not half-replaced by the pattern for "Al".
    leaked.sort(key=lambda x: len(x[0]), reverse=True)
    n_person = 0
    for name, pattern in leaked:
        placeholder = anonymiser_placeholder(name, original, anonymised)
        if placeholder is None:
            placeholder = f"[person{n_person}]"
            n_person += 1
        ends = iter(_sentence_ends(name, anonymised, original))
        anonymised = pattern.sub(
            lambda m, p=placeholder, e=ends: p + (" ." if next(e) else ""), anonymised
        )
    return anonymised


def story_text(anonymised: str, names: list[str], original: str) -> str:
    """Final story text: leaked names masked, placeholders rewritten, then detokenised."""
    return detokenize(rewrite_placeholders(mask_leaked_names(anonymised, names, original))).strip()


def image_key(url: str) -> str:
    """``.../tt0047396_0010_0/shot_0667_img_0.jpg`` → ``tt0047396_0010_0_shot_0667_img_0``."""
    m = _URL_RE.search(url)
    if m is None:
        raise ValueError(f"unexpected VWP image URL: {url}")
    return "_".join(m.groups())


@dataclass(frozen=True)
class RawStory:
    story_id: str
    scene_id: str
    image_keys: tuple[str, ...]  # one per link0..k, in that order (see image_key)
    urls: tuple[str, ...]
    n_listed: int  # len(img_id_list)
    n_texts: int  # non-empty text0..9 segments (one per shot)
    text: str  # final story text (see story_text)
    aligned: bool = True  # links and texts both fill slots 0..n-1 with no gap


def read_csv(path: Path) -> list[RawStory]:
    """Step 1–2: one :class:`RawStory` per CSV row, in file order.

    Raises if two rows share a ``story_id``, because record ids must be unique per split.
    """
    stories = []
    seen_ids: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            story_id = row["story_id"].strip()
            if story_id in seen_ids:
                raise ValueError(f"{path}: duplicate story_id {story_id!r}")
            seen_ids.add(story_id)
            link_slots = [i for i in range(_N_LINKS) if row[f"link{i}"].strip() not in _EMPTY]
            text_slots = [i for i in range(_N_LINKS) if row[f"text{i}"].strip()]
            urls = tuple(row[f"link{i}"].strip() for i in link_slots)
            names = [c for i in range(_N_CHARS) if (c := row[f"char{i}"].strip()) not in _EMPTY]
            stories.append(
                RawStory(
                    story_id=story_id,
                    scene_id=row["scene_full_id"].strip(),
                    image_keys=tuple(image_key(u) for u in urls),
                    urls=urls,
                    n_listed=len(ast.literal_eval(row["img_id_list"])),
                    n_texts=len(text_slots),
                    text=story_text(row["anonymised_story"], names, row["story"]),
                    aligned=link_slots == list(range(len(link_slots)))
                    and text_slots == list(range(len(text_slots))),
                )
            )
    return stories


def image_urls(stories: list[RawStory]) -> dict[str, str]:
    """Image key → URL for every image linked in ``stories`` (used by data/download_vwp.py)."""
    urls: dict[str, str] = {}
    for story in stories:
        for key, url in zip(story.image_keys, story.urls, strict=True):
            if urls.setdefault(key, url) != url:
                raise ValueError(f"image {key} has two URLs: {urls[key]} and {url}")
    return urls


def build_records(
    stories: list[RawStory],
    *,
    split: str,
    raw_image_dir: Path,
    out_image_dir: Path,
    max_image_side: int,
) -> tuple[list[StoryRecord], dict[str, int]]:
    """Steps 3–4: return ``(records, drop_counts)``; log how many stories were dropped and why.

    Checks run in ``DROP_REASONS`` order, so a story is counted under its first problem.
    For duplicates (same images and same text), the first story in the file is kept.
    If a story has both missing and broken images, it counts as ``missing_image``.
    Images are only resized for stories that are kept.
    """
    drops = dict.fromkeys(DROP_REASONS, 0)
    seen: set[tuple[tuple[str, ...], str]] = set()
    status: dict[str, Path | str] = {}  # image key -> valid raw path, or drop reason

    def check(key: str) -> Path | str:
        if key not in status:
            raw = find_image(raw_image_dir, key)
            if raw is None:
                status[key] = "missing_image"
            elif not is_valid_image(raw):
                status[key] = "broken_image"
            else:
                status[key] = raw
        return status[key]

    records: list[StoryRecord] = []
    for story in stories:
        n = len(story.image_keys)
        if not story.aligned or n != story.n_listed or n != story.n_texts:
            drops["link_count_mismatch"] += 1
            continue
        key = (story.image_keys, story.text)
        if key in seen:
            drops["duplicate"] += 1
            continue
        seen.add(key)
        if not MIN_IMAGES <= n <= MAX_IMAGES:
            drops["bad_length"] += 1
            continue
        if not story.text:
            drops["empty_story"] += 1
            continue
        checks = [check(k) for k in story.image_keys]
        failed = {c for c in checks if isinstance(c, str)}
        if failed:
            # "missing_image" wins: it means the download is incomplete (blocks dev-200 lists).
            drops["missing_image" if "missing_image" in failed else "broken_image"] += 1
            continue

        images = []
        for k, raw in zip(story.image_keys, checks, strict=True):
            dst = out_image_dir / f"{k}.jpg"
            resize_long_side(raw, dst, max_side=max_image_side)
            images.append(dst.as_posix())
        records.append(
            StoryRecord(
                id=story.story_id, dataset="vwp", split=split, images=images, story=story.text
            )
        )

    log.info("VWP %s: kept %d of %d stories, dropped %s", split, len(records), len(stories), drops)
    return records, drops


def _one_per_scene(ids: set[str], scene_of: dict[str, str], seed: int) -> list[str]:
    """One random kept story per scene (scenes and ids sorted first, so it is deterministic)."""
    by_scene: dict[str, list[str]] = defaultdict(list)
    for i in sorted(ids):
        by_scene[scene_of[i]].append(i)
    rng = random.Random(seed)
    return [rng.choice(by_scene[scene]) for scene in sorted(by_scene)]


def _ensure_dev_ids(
    cfg: ExperimentConfig,
    split: str,
    records: list[StoryRecord],
    drops: dict[str, int],
    scene_of: dict[str, str],
) -> None:
    """Write the committed dev-subset id list once; never overwrite an existing one.

    The list holds at most one story per scene, so no scene is evaluated twice.
    Raises if an existing list names stories that were not built here, because eval
    would then silently run on fewer stories than the committed subset.
    """
    size = cfg.data.dev_subset_size
    path = subset_path(cfg.data.splits_dir, "vwp", split, size)
    ids = {r.id for r in records}
    if path.exists():
        absent = [i for i in read_ids(path) if i not in ids]
        if absent:
            raise RuntimeError(
                f"{path} lists {len(absent)} ids that were not built here (e.g. {absent[0]}); "
                f"check the {split} images (drop counts: {drops})"
            )
        return
    if not ids:
        log.warning("not writing %s: no stories were built", path)
        return
    if drops["missing_image"]:
        log.warning(
            "not writing %s: %d stories lack images. Download all %s images first.",
            path,
            drops["missing_image"],
            split,
        )
        return
    picked = _one_per_scene(ids, scene_of, cfg.experiment.seed)
    if len(picked) < size:
        log.warning("%s: only %d scenes, so the list has %d ids", path, len(picked), len(picked))
    write_ids(path, sample_ids(picked, size, cfg.experiment.seed))
    log.info("wrote %s", path)


def prepare(cfg: ExperimentConfig, split: str) -> Path:
    """Run steps 1–5 for one split; return the written JSONL path.

    Fails (and writes nothing) for the blind test split, if the CSV or the split's raw
    image folder is missing, or if no story could be built, so a bad run never replaces
    a good JSONL with an empty one.
    """
    if split == "test":
        raise ValueError("VWP test is blind (its CSV has no stories); use --split train or val")
    raw_dir = Path(cfg.data.raw_dir) / "vwp"
    csv_path = raw_dir / csv_name(split)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{csv_path} not found; run: python data/download_vwp.py --split {split}"
        )
    raw_image_dir = raw_dir / "images" / split
    if not raw_image_dir.is_dir():
        raise FileNotFoundError(
            f"{raw_image_dir} not found; run: python data/download_vwp.py --split {split}"
        )
    stories = read_csv(csv_path)
    records, drops = build_records(
        stories,
        split=split,
        raw_image_dir=raw_image_dir,
        out_image_dir=Path(cfg.data.image_dir) / "vwp" / split,
        max_image_side=cfg.data.max_image_side,
    )
    out_path = Path(cfg.data.jsonl_dir) / f"vwp_{split}.jsonl"
    if not records:
        raise RuntimeError(
            f"VWP {split}: no stories were built (dropped {drops}); "
            f"is {raw_image_dir} complete? {out_path} was not changed."
        )
    n = write_jsonl(out_path, records)
    log.info("VWP %s: wrote %d records to %s", split, n, out_path)

    if split == "val":
        _ensure_dev_ids(cfg, split, records, drops, {s.story_id: s.scene_id for s in stories})
    return out_path
