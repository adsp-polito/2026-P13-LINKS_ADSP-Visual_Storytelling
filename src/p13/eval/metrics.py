from __future__ import annotations

import csv
import itertools
import logging
import math
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeAlias

from p13.common.config import ExperimentConfig
from p13.common.jsonl import read_jsonl
from p13.common.schema import GeneratedStory, StoryRecord

log = logging.getLogger(__name__)

Key: TypeAlias = tuple[str, str, str]
Scores: TypeAlias = dict[Key, float]
Refs: TypeAlias = dict[Key, StoryRecord]
Group: TypeAlias = tuple[str, str | None, str, str, str]

_WORD_RE = re.compile(r"\w+(?:'\w+)*")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
NR_BLOCK = 4


def _sentences(text: str) -> list[list[str]]:
    sentences = (_words(s) for s in _SENTENCE_END_RE.split(text.strip()))
    return [s for s in sentences if s]


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def _non_redundancy(text: str) -> float:
    sentences = _sentences(text)
    if len(sentences) < 2:
        return math.nan
    between = [_jaccard(a, b) for a, b in itertools.combinations(sentences, 2)]
    within = [
        _jaccard(s[i : i + NR_BLOCK], s[i + NR_BLOCK : i + 2 * NR_BLOCK])
        for s in sentences
        for i in range(0, len(s) - NR_BLOCK, NR_BLOCK)
    ]
    parts = [sum(between) / len(between)]
    if within:
        parts.append(sum(within) / len(within))
    return 1 - sum(parts) / len(parts)


def _todo(name: str) -> NotImplementedError:
    return NotImplementedError(f"metric {name!r} not implemented yet (Nioosha)")


# ---------------------------------------------------------------- headline


def groovist(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """GROOViST (Surikuchi et al., EMNLP 2023): visual grounding. Reference-free.

    CLIP matches the story's noun phrases to image regions across the **whole** sequence
    (order-independent); phrases under the threshold count negatively; scores are weighted
    by word **concreteness**. The threshold is fixed per dataset from the **reference**
    stories, read from config (never hard-coded) and reused for every model. Keep the
    per-phrase scores for error analysis.
    """
    raise _todo("groovist")


def rovist_c(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """RoViST-C (Wang et al., Findings NAACL 2022): coherence from ALBERT **sentence-order
    prediction** over consecutive sentences. Reference-free."""
    raise _todo("rovist_c")


def rovist_nr(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """RoViST-NR: non-redundancy in [0, 1]; higher = less repetition. Reference-free.

    Clean version of RoViST's idea (not a copy of the authors' code): words are lower-cased
    with punctuation ignored, as in ``length``; overlap is set-based Jaccard.

    * Between sentences: mean Jaccard over all pairs of sentences.
    * Within a sentence: mean Jaccard between consecutive blocks of ``NR_BLOCK`` (4) words.
    * NR = 1 − mean(between, within); only the between part if no sentence has two blocks.

    Sentences end at ``.``, ``!`` or ``?`` followed by a space. A story with fewer than two
    sentences gets no score (empty cell).
    """
    scores: Scores = {}
    for g in gens:
        value = _non_redundancy(g.story)
        if not math.isnan(value):
            scores[g.key] = value
    return scores


DHM_GAPS: dict[str, str] = {"dhm_g": "groovist", "dhm_c": "rovist_c", "dhm_nr": "rovist_nr"}
DERIVED_METRICS: tuple[str, ...] = ("dhm_g", "dhm_c", "dhm_nr", "dhm")


def derive_dhm(
    keys: list[Key], model: dict[str, Scores], human: dict[str, Scores]
) -> dict[str, Scores]:
    """dHM per story (paper Eq. 1–2), from scores already computed in this run.

    ``model`` / ``human`` map ``groovist``, ``rovist_c``, ``rovist_nr`` to per-story scores of
    a model group and of the ``model="human"`` group on the same dataset split. For each key:

    * ``dhm_g``, ``dhm_c``, ``dhm_nr`` = model − human (**signed**; positive = above the
      human story). No normalisation: GROOViST is in [-1, 1], RoViST-C/NR in [0, 1].
    * ``dhm`` = mean of the three absolute gaps (the paper's number, for comparability).

    A gap is NaN if either side has no score for that story, and ``dhm`` is NaN if any of
    the three gaps is, so it becomes an empty cell rather than a guess.
    """
    out: dict[str, Scores] = {name: {} for name in DERIVED_METRICS}
    for key in keys:
        gaps = []
        for name, base in DHM_GAPS.items():
            m, h = model[base].get(key), human[base].get(key)
            gap = math.nan if m is None or h is None else m - h
            out[name][key] = gap
            gaps.append(gap)
        out["dhm"][key] = sum(abs(g) for g in gaps) / len(gaps)
    return out


def halluc_any(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """1 if ``halluc_ner`` or the object check finds at least one hallucination, else 0.
    Its mean over a group is the **hallucination rate** used in the decision rule."""
    raise _todo("halluc_any")


def halluc_ner(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """Count of invented names / places / specific dates per story (spaCy NER: PERSON;
    GPE / LOC / FAC; DATE only for specific dates such as years or calendar dates, not vague
    phrases like "the weekend")."""
    raise _todo("halluc_ner")


def halluc_obj_share(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """Share of mentioned objects (spaCy nouns) that an open-vocabulary detector
    (OWLv2 / Grounding DINO) doesn't find in any image of the sequence (CHAIR-style).

    **Empty** (no key / NaN), not 0, when a story mentions no objects: 0 would reward saying
    nothing. The report pools it per group as in CHAIR: all hallucinated objects / all
    mentioned objects, i.e. sum(share × ``n_objects``) / sum(``n_objects``).
    """
    raise _todo("halluc_obj_share")


def n_objects(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """Number of objects (spaCy nouns) the story mentions: the denominator of
    ``halluc_obj_share``. A count the report needs for pooling, not a score."""
    raise _todo("n_objects")


# ---------------------------------------------------------------- supporting


def clipscore(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """CLIPScore, a second grounding signal. Reference-free.

    Split the story into sentences, score each sentence against its **best-matching** image
    in the sequence (CLIPScore = 2.5 · max(cos, 0)) and average over sentences. Never pair
    sentence i with image i: stories rarely have one sentence per image.
    """
    raise _todo("clipscore")


def distinct_2(gens: list[GeneratedStory]) -> float:
    """Unique bigrams / all bigrams over **all** stories of one system on one dataset split.

    Corpus-level on purpose (a generic-story detector): a model that tells
    the same story for every sequence scores low here even if each story looks varied on
    its own. Bigrams never cross story boundaries. Lower-cased words, punctuation ignored.
    Returns 0.0 if there are no bigrams at all.
    """
    total = 0
    unique: set[tuple[str, str]] = set()
    for g in gens:
        bigrams = list(itertools.pairwise(_words(g.story)))
        total += len(bigrams)
        unique.update(bigrams)
    return len(unique) / total if total else 0.0


def length(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """Story length in words (P1 asks for 50–100). Punctuation is not counted."""
    return {g.key: float(len(_words(g.story))) for g in gens}


def reading_grade(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """Flesch–Kincaid grade level (e.g. via ``textstat``)."""
    raise _todo("reading_grade")


# ---------------------------------------------------------------- comparability


def bleu4(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """BLEU-4 vs the cleaned reference (one ``pycocoevalcap`` call with METEOR, CIDEr,
    ROUGE-L). Comparability only."""
    raise _todo("bleu4")


def meteor(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """METEOR vs the cleaned reference. Comparability only."""
    raise _todo("meteor")


def cider(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """CIDEr-D vs the cleaned reference. Needs the whole group for its IDF. Comparability only."""
    raise _todo("cider")


def rouge_l(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """ROUGE-L F1 vs the cleaned reference. Comparability only."""
    raise _todo("rouge_l")


def bertscore(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """BERTScore F1 vs the cleaned reference (roberta-large, baseline rescaling).
    Comparability only."""
    raise _todo("bertscore")


ID_COLUMNS = ("dataset", "split", "id", "model", "adapter", "prompt_id")
GROUP_COLUMNS = ("dataset", "split", "model", "adapter", "prompt_id", "n_stories")


def human_stories(keys: Iterable[Key], refs: Refs) -> list[GeneratedStory]:
    """The human baseline: one ``model="human"`` story per key, its text = the reference story.

    ``adapter`` is null and ``prompt_id`` is ``"reference"``, so the human stories form one
    system per dataset split and go through the same pipeline as the model stories.
    """
    return [
        GeneratedStory(
            id=refs[key].id,
            dataset=refs[key].dataset,
            split=refs[key].split,
            model=HUMAN_MODEL,
            adapter=None,
            prompt_id=HUMAN_PROMPT_ID,
            story=refs[key].story,
        )
        for key in sorted(set(keys))
    ]


def evaluate(cfg: ExperimentConfig, story_files: list[Path], *, with_human: bool = False) -> Path:
    """Run every metric in ``cfg.eval.metrics`` on the given GeneratedStory files.

    Steps:

    1. Check the metric names (fails before any work if one is unknown or listed twice, or
       if ``halluc_obj_share`` is asked without ``n_objects``, which the report needs to pool it).
    2. Read the stories and group them by system ``(model, adapter, prompt_id)`` **and**
       ``(dataset, split)``: datasets are never pooled. The same story twice
       in one system is an error, even across files. ``model`` / ``adapter`` are only used
       for grouping, never to compute a score.
    3. Load the cleaned references from ``<jsonl_dir>/<dataset>_<split>.jsonl``. A story
       without a reference is an error: eval must never quietly run on fewer stories.
       The test split is refused unless ``eval.allow_test`` is true (final numbers only).
       With ``with_human``, add the human baseline (:func:`human_stories`): one
       ``model="human"`` group per dataset split, covering every story id of the model
       groups there. The story files must then not contain ``model="human"`` stories.
    4. If ``dhm*`` is asked for, check (before any scoring) that ``groovist``, ``rovist_c``
       and ``rovist_nr`` are asked for too, and that every model group has exactly one
       ``model="human"`` group on the same dataset split covering all its story ids.
    5. Run each per-story metric once per group. For the human baseline
       (``model == "human"``) the metrics in :data:`REFERENCE_METRICS` are not called and
       their cells stay empty: a story compared with itself scores perfectly, which means
       nothing. This is the only place ``model`` affects scoring.
    6. Derive ``dhm*`` (:func:`derive_dhm`): pair each model row with the human row of the
       same ``(dataset, split, id)``. Human rows keep these cells empty.
    7. Write ``cfg.run_dir / "scores.csv"``: one row per (system, story), columns
       :data:`ID_COLUMNS` + one per per-story metric. An empty cell means the metric gave no
       score for that story (e.g. a reference-based metric for ``model="human"``);
       ``adapter`` is empty for zero-shot.
    8. If corpus-level metrics (:data:`CORPUS_METRICS`, e.g. distinct-2) were asked for,
       also write ``cfg.run_dir / "system_scores.csv"``: one row per group,
       :data:`GROUP_COLUMNS` + one column per corpus metric. Otherwise an old
       ``system_scores.csv`` is removed, so the two files always come from the same run.

    Returns the path of ``scores.csv``.
    """
    names = _check_metric_names(cfg.eval.metrics)
    metric_names = [n for n in names if n in METRICS]
    derived_names = [n for n in names if n in DERIVED_METRICS]
    story_names = [n for n in names if n in METRICS or n in DERIVED_METRICS]
    corpus_names = [n for n in names if n in CORPUS_METRICS]
    groups = _group_stories(story_files)
    dataset_splits = {g[3:] for g in groups}
    if not cfg.eval.allow_test and any(split == "test" for _, split in dataset_splits):
        raise ValueError(
            "refusing to score the test split: it stays untouched until the final numbers "
            "(set eval.allow_test=true for those)"
        )
    refs = _load_refs(Path(cfg.data.jsonl_dir), dataset_splits)
    for group, gens in groups.items():
        _check_refs_exist(group, gens, refs)
    if with_human:
        groups = _add_human_groups(groups, refs)
    human_of = _check_dhm_inputs(names, groups) if derived_names else {}

    all_scores: dict[Group, dict[str, Scores]] = {}
    for group, gens in groups.items():
        log.info("scoring %d stories for %s", len(gens), _group_label(group))
        is_human = group[0] == HUMAN_MODEL
        all_scores[group] = {
            name: {} if is_human and name in REFERENCE_METRICS else METRICS[name](gens, refs)
            for name in metric_names
        }
    for group, human_group in human_of.items():
        keys = [g.key for g in groups[group]]
        all_scores[group].update(derive_dhm(keys, all_scores[group], all_scores[human_group]))
    for group in groups:
        for name in derived_names:
            all_scores[group].setdefault(name, {})

    rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for group, gens in groups.items():
        scores = all_scores[group]
        for g in gens:
            row: dict[str, object] = {
                "dataset": g.dataset,
                "split": g.split,
                "id": g.id,
                "model": g.model,
                "adapter": g.adapter or "",
                "prompt_id": g.prompt_id,
            }
            for name in story_names:
                row[name] = _cell(scores[name].get(g.key))
            rows.append(row)
        if corpus_names:
            model, adapter, prompt_id, dataset, split = group
            group_row: dict[str, object] = {
                "dataset": dataset,
                "split": split,
                "model": model,
                "adapter": adapter or "",
                "prompt_id": prompt_id,
                "n_stories": len(gens),
            }
            for name in corpus_names:
                group_row[name] = _cell(CORPUS_METRICS[name](gens))
            group_rows.append(group_row)

    out = cfg.run_dir / "scores.csv"
    _write_csv(out, [*ID_COLUMNS, *story_names], rows)
    log.info("wrote %d rows (%d groups) to %s", len(rows), len(groups), out)
    group_out = cfg.run_dir / "system_scores.csv"
    if not corpus_names:
        group_out.unlink(missing_ok=True)
    else:
        _write_csv(group_out, [*GROUP_COLUMNS, *corpus_names], group_rows)
        log.info("wrote %d rows to %s", len(group_rows), group_out)
    return out


def _cell(value: float | None) -> object:
    return "" if value is None or math.isnan(value) else value


def _check_metric_names(names: list[str]) -> list[str]:
    if not names:
        raise ValueError("eval.metrics is empty; nothing to compute")
    known = {*METRICS, *DERIVED_METRICS, *CORPUS_METRICS}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown metric(s) {unknown}; choose from {sorted(known)}")
    if len(set(names)) != len(names):
        raise ValueError(f"eval.metrics lists a metric twice: {names}")
    if "halluc_obj_share" in names and "n_objects" not in names:
        raise ValueError(
            "halluc_obj_share needs n_objects in eval.metrics too "
            "(the report pools the share as sum(share × n_objects) / sum(n_objects))"
        )
    return list(names)


def _add_human_groups(
    groups: dict[Group, list[GeneratedStory]], refs: Refs
) -> dict[Group, list[GeneratedStory]]:
    if any(group[0] == HUMAN_MODEL for group in groups):
        raise ValueError(
            f"with_human: the story files already contain model={HUMAN_MODEL!r} stories; "
            "pass either those files or --with-human, not both"
        )
    keys = [g.key for gens in groups.values() for g in gens]
    humans = human_stories(keys, refs)
    merged = dict(groups)
    for h in humans:
        merged.setdefault(_group_of(h), []).append(h)
    return dict(sorted(merged.items(), key=lambda kv: _group_label(kv[0])))


def _check_dhm_inputs(
    names: list[str], groups: dict[Group, list[GeneratedStory]]
) -> dict[Group, Group]:
    """Fail before any scoring if ``dhm*`` can't be derived; return model group -> human group.

    Needs ``groovist``, ``rovist_c`` and ``rovist_nr`` in ``eval.metrics`` and, for every
    model group, exactly one ``model="human"`` group on the same dataset split that has a
    story for every id of the model group (eval never quietly scores fewer stories).
    """
    missing = [m for m in DHM_GAPS.values() if m not in names]
    if missing:
        raise ValueError(f"dhm* needs {missing} in eval.metrics too (dHM is derived from them)")
    humans: dict[tuple[str, str], list[Group]] = {}
    for group in groups:
        if group[0] == HUMAN_MODEL:
            humans.setdefault(group[3:], []).append(group)
    human_of: dict[Group, Group] = {}
    for group, gens in groups.items():
        if group[0] == HUMAN_MODEL:
            continue
        dataset, split = group[3:]
        found = humans.get((dataset, split), [])
        if len(found) != 1:
            what = "no" if not found else f"{len(found)} (prompt_ids differ)"
            raise ValueError(
                f"dhm* for {_group_label(group)} needs exactly one model={HUMAN_MODEL!r} group "
                f"on {dataset}/{split} in this run; found {what}"
            )
        human_ids = {g.key for g in groups[found[0]]}
        absent = [g.key for g in gens if g.key not in human_ids]
        if absent:
            raise ValueError(
                f"dhm* for {_group_label(group)}: {len(absent)} of {len(gens)} stories have no "
                f"human story in this run (e.g. {absent[0]})"
            )
        human_of[group] = found[0]
    return human_of


def _group_of(g: GeneratedStory) -> Group:
    return (g.model, g.adapter, g.prompt_id, g.dataset, g.split)


def _group_label(group: Group) -> str:
    model, adapter, prompt_id, dataset, split = group
    return f"{model} + {adapter or 'no adapter'} + {prompt_id} on {dataset}/{split}"


def _group_stories(story_files: list[Path]) -> dict[Group, list[GeneratedStory]]:
    """Read every file; return stories per group in a stable order (sorted by key)."""
    groups: dict[Group, dict[Key, GeneratedStory]] = {}
    for path in story_files:
        for g in read_jsonl(path, GeneratedStory):
            stories = groups.setdefault(_group_of(g), {})
            if g.key in stories:
                raise ValueError(
                    f"{path}: story {g.key} appears twice for {_group_label(_group_of(g))}"
                )
            stories[g.key] = g
    if not groups:
        raise ValueError(f"no generated stories in {[str(p) for p in story_files]}")
    return {
        grp: [stories[k] for k in sorted(stories)]
        for grp, stories in sorted(groups.items(), key=lambda kv: _group_label(kv[0]))
    }


def _load_refs(jsonl_dir: Path, dataset_splits: set[tuple[str, str]]) -> Refs:
    refs: Refs = {}
    for dataset, split in sorted(dataset_splits):
        path = jsonl_dir / f"{dataset}_{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found: the reference stories for {dataset}/{split} are needed "
                f"(run p13-prepare, or set data.jsonl_dir)"
            )
        for r in read_jsonl(path, StoryRecord):
            refs[(r.dataset, r.split, r.id)] = r
    return refs


def _check_refs_exist(group: Group, gens: list[GeneratedStory], refs: Refs) -> None:
    absent = [g.key for g in gens if g.key not in refs]
    if absent:
        raise ValueError(
            f"{len(absent)} of {len(gens)} stories for {_group_label(group)} have no "
            f"reference story (e.g. {absent[0]}); is data.jsonl_dir the right folder?"
        )


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    """Write to ``<path>.tmp`` then rename, so a crash never leaves a half-written CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


METRICS: dict[str, Callable[[list[GeneratedStory], Refs], Scores]] = {
    "groovist": groovist,
    "rovist_c": rovist_c,
    "rovist_nr": rovist_nr,
    "halluc_any": halluc_any,
    "halluc_ner": halluc_ner,
    "halluc_obj_share": halluc_obj_share,
    "n_objects": n_objects,
    "clipscore": clipscore,
    "length": length,
    "reading_grade": reading_grade,
    "bleu4": bleu4,
    "meteor": meteor,
    "cider": cider,
    "rouge_l": rouge_l,
    "bertscore": bertscore,
}


CORPUS_METRICS: dict[str, Callable[[list[GeneratedStory]], float]] = {
    "distinct_2": distinct_2,
}

HUMAN_MODEL = "human"
HUMAN_PROMPT_ID = "reference"

REFERENCE_METRICS: frozenset[str] = frozenset(
    {"dhm_g", "dhm_c", "dhm_nr", "dhm", "bleu4", "meteor", "cider", "rouge_l", "bertscore"}
)
