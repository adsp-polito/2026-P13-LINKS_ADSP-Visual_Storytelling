"""GROOViST (Surikuchi et al., EMNLP 2023): visual grounding of a story, reference-free.

Clean reimplementation of https://github.com/akskuchi/groovist with our own region source:

1. Noun phrases: spaCy ``noun_chunks`` of the story, lower-cased, VIST placeholders such as
   ``[male]`` turned into plain words, single-pronoun chunks dropped, duplicates dropped.
2. Regions: per image, the whole image plus the ``eval.regions_per_image`` boxes with the
   highest OWLv2 objectness (after non-maximum suppression).
3. Alignment of a phrase = 2.5 · the highest CLIP cosine between the phrase and any region of
   any image of the sequence (order-independent).
4. Penalty: alignment ``a`` stays ``a`` if ``a >= θ`` and becomes ``a − θ`` otherwise, with
   θ fixed per dataset (``eval.groovist_threshold``).
5. Concreteness: each phrase is weighted by the Brysbaert concreteness of its head noun
   (spaCy ``chunk.root``: the word, else its lemma, else 1.0). The authors average every word,
   so "the dog" weighed less than "dogs"; the head alone removes that style effect.
6. Story score = tanh(mean over phrases of weight · penalised alignment), in (−1, 1).
   A story without noun phrases gets no score.

Two extra columns from the same phrases: ``groovist_raw`` (the mean before tanh; tanh squeezes
well-grounded stories together near 1) and ``groovist_grounded_share`` (share of phrases whose
alignment reaches θ).

Every phrase of every scored story is also written to ``<run_dir>/groovist_phrases.jsonl``
for error analysis.
"""

from __future__ import annotations

import csv
import dataclasses
import functools
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from p13.common.config import ExperimentConfig
from p13.common.jsonl import read_jsonl
from p13.common.schema import GeneratedStory, StoryRecord
from p13.common.subset import read_ids, subset_path

log = logging.getLogger(__name__)

SCALE = 2.5
DEFAULT_WEIGHT = 1.0
NMS_IOU = 0.5
PHRASES_FILE = "groovist_phrases.jsonl"
PLACEHOLDERS = {
    "[male]": "male",
    "[female]": "female",
    "[location]": "location",
    "[organization]": "organization",
}

Token = tuple[str, str, str, bool]


@dataclass(frozen=True)
class Phrase:
    text: str
    weight: float
    alignment: float


class Backend(Protocol):
    device: str

    def noun_chunks(self, text: str) -> list[list[Token]]: ...

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...

    def embed_regions(self, image_path: str) -> np.ndarray: ...


def clean_text(text: str) -> str:
    for placeholder, word in PLACEHOLDERS.items():
        text = text.replace(placeholder, word)
    return text


def select_phrases(chunks: list[list[Token]]) -> list[list[Token]]:
    """Drop chunks that are a single pronoun and repeated chunks (first one kept)."""
    seen: set[str] = set()
    kept = []
    for chunk in chunks:
        if not chunk or (len(chunk) == 1 and chunk[0][2] == "PRON"):
            continue
        text = _phrase_text(chunk)
        if text in seen:
            continue
        seen.add(text)
        kept.append(chunk)
    return kept


def concreteness(chunk: list[Token], lexicon: dict[str, float]) -> float:
    """Rating of the chunk's head noun: the word, else its lemma, else ``DEFAULT_WEIGHT``.

    Only the head counts, so articles and possessives ("the", "his") don't change the weight.
    """
    heads = [t for t in chunk if t[3]] or chunk[-1:]
    word, lemma, _, _ = heads[0]
    return lexicon.get(word, lexicon.get(lemma, DEFAULT_WEIGHT))


def _phrase_text(chunk: list[Token]) -> str:
    return " ".join(t[0] for t in chunk)


def max_alignments(phrase_embs: np.ndarray, region_embs: list[np.ndarray]) -> np.ndarray:
    """2.5 · max cosine of each phrase over every region of every image."""
    regions = np.concatenate(region_embs, axis=0)
    return SCALE * (_normalise(phrase_embs) @ _normalise(regions).T).max(axis=1)


def penalise(alignment: float, theta: float) -> float:
    return alignment if alignment >= theta else alignment - theta


def raw_score(phrases: list[Phrase], theta: float) -> float:
    """Mean of weight · penalised alignment over the phrases (GROOViST before tanh)."""
    if not phrases:
        return math.nan
    values = [p.weight * penalise(p.alignment, theta) for p in phrases]
    return sum(values) / len(values)


def story_score(phrases: list[Phrase], theta: float) -> float:
    return math.tanh(raw_score(phrases, theta))


def grounded_share(phrases: list[Phrase], theta: float) -> float:
    """Share of phrases whose alignment reaches θ."""
    if not phrases:
        return math.nan
    return sum(p.alignment >= theta for p in phrases) / len(phrases)


def story_phrases(
    story: str, images: list[str], backend: Backend, lexicon: dict[str, float]
) -> list[Phrase]:
    return story_phrases_on(story, [images], backend, lexicon)[0]


def story_phrases_on(
    story: str, image_sets: list[list[str]], backend: Backend, lexicon: dict[str, float]
) -> list[list[Phrase]]:
    """The story's phrases aligned with each image set in turn (the story is parsed once)."""
    chunks = select_phrases(backend.noun_chunks(clean_text(story)))
    if not chunks:
        return [[] for _ in image_sets]
    texts = [_phrase_text(chunk) for chunk in chunks]
    weights = [concreteness(chunk, lexicon) for chunk in chunks]
    embs = backend.embed_texts(texts)
    out = []
    for images in image_sets:
        alignments = max_alignments(embs, [backend.embed_regions(path) for path in images])
        out.append(
            [
                Phrase(text, w, float(a))
                for text, w, a in zip(texts, weights, alignments, strict=True)
            ]
        )
    return out


COLUMNS: dict[str, Any] = {
    "groovist": story_score,
    "groovist_raw": raw_score,
    "groovist_grounded_share": grounded_share,
}

_last: tuple[list[GeneratedStory], ExperimentConfig, dict[str, dict[Any, float]]] | None = None


def score(
    name: str, gens: list[GeneratedStory], refs: dict[Any, StoryRecord], cfg: ExperimentConfig
) -> dict[Any, float]:
    """One of :data:`COLUMNS` for a group. Within one ``evaluate`` run every metric of a group
    gets the same ``gens`` list, so the phrases are computed once for all columns; a new run
    reads the files again and recomputes."""
    global _last
    if _last is None or _last[0] is not gens or _last[1] is not cfg:
        _last = (gens, cfg, _score_group(gens, refs, cfg))
    return _last[2][name]


def _score_group(
    gens: list[GeneratedStory], refs: dict[Any, StoryRecord], cfg: ExperimentConfig
) -> dict[str, dict[Any, float]]:
    datasets = sorted({g.dataset for g in gens})
    missing = [d for d in datasets if d not in cfg.eval.groovist_threshold]
    if missing:
        raise ValueError(
            f"groovist needs eval.groovist_threshold for {missing}: run "
            "p13-eval --calibrate-groovist and paste the line it prints into the config"
        )
    mismatches = {d: settings_mismatches(cfg, d) for d in datasets}
    check_mismatches(mismatches, cfg.eval.ignore_threshold_mismatch)
    backend = load_backend(cfg)
    lexicon = load_lexicon(cfg.eval.concreteness_file)
    record_settings(cfg, mismatches, backend.device)
    out: dict[str, dict[Any, float]] = {name: {} for name in COLUMNS}
    rows: list[dict[str, Any]] = []
    for g in gens:
        theta = cfg.eval.groovist_threshold[g.dataset].theta
        phrases = story_phrases(g.story, refs[g.key].images, backend, lexicon)
        for name, fn in COLUMNS.items():
            value = fn(phrases, theta)
            if not math.isnan(value):
                out[name][g.key] = value
        rows.extend(phrase_rows(g, phrases, theta))
    append_phrase_rows(cfg.run_dir / PHRASES_FILE, rows)
    return out


CHECKED_SETTINGS = ("clip_model", "region_model", "regions_per_image", "spacy_model", "dtype")
SETTINGS_FILE = "groovist_settings.json"


def current_settings(cfg: ExperimentConfig) -> dict[str, Any]:
    return {name: getattr(cfg.eval, name) for name in CHECKED_SETTINGS}


def settings_mismatches(cfg: ExperimentConfig, dataset: str) -> list[str]:
    """Settings that differ between θ's calibration and this run (the device isn't checked)."""
    calibrated = cfg.eval.groovist_threshold[dataset]
    return [
        f"{name}: θ was calibrated with {getattr(calibrated, name)!r}, "
        f"this run has eval.{name}={value!r}"
        for name, value in current_settings(cfg).items()
        if getattr(calibrated, name) != value
    ]


def check_mismatches(mismatches: dict[str, list[str]], ignore: bool) -> None:
    found = {d: m for d, m in mismatches.items() if m}
    if not found:
        return
    lines = [f"{d} {m}" for d, ms in found.items() for m in ms]
    if not ignore:
        raise ValueError(
            "groovist settings differ from the ones θ was calibrated with:\n  "
            + "\n  ".join(lines)
            + "\nrecalibrate with p13-eval --calibrate-groovist, or pass "
            "--ignore-threshold-mismatch for a quick local test"
        )
    banner = "!" * 72
    log.warning(
        "%s\nIGNORING GROOViST THRESHOLD MISMATCH (--ignore-threshold-mismatch): these scores "
        "are not comparable with properly calibrated runs\n  %s\n%s",
        banner,
        "\n  ".join(lines),
        banner,
    )


def record_settings(cfg: ExperimentConfig, mismatches: dict[str, list[str]], device: str) -> None:
    """Add this group's datasets to ``<run_dir>/groovist_settings.json`` (cleared every run)."""
    path = cfg.run_dir / SETTINGS_FILE
    record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for dataset, found in mismatches.items():
        record[dataset] = {
            "calibrated": dataclasses.asdict(cfg.eval.groovist_threshold[dataset]),
            "used": {**current_settings(cfg), "device": device},
            "mismatches": found,
            "ignored_mismatch": bool(found) and cfg.eval.ignore_threshold_mismatch,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def phrase_rows(g: GeneratedStory, phrases: list[Phrase], theta: float) -> list[dict[str, Any]]:
    """One row per noun phrase, for error analysis (lowest ``value`` = least grounded)."""
    return [
        {
            "dataset": g.dataset,
            "split": g.split,
            "id": g.id,
            "model": g.model,
            "adapter": g.adapter,
            "prompt_id": g.prompt_id,
            "phrase": p.text,
            "weight": p.weight,
            "alignment": p.alignment,
            "grounded": p.alignment >= theta,
            "value": p.weight * penalise(p.alignment, theta),
        }
        for p in phrases
    ]


def append_phrase_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


CALIBRATION_FILE = "groovist_threshold.json"


def calibrate(cfg: ExperimentConfig) -> Path:
    """Fix θ per dataset from the reference stories and check that GROOViST sees grounding.

    For every dataset in ``data.datasets`` on ``data.split`` (the eval set: the dev subset
    if ``data.subset`` is ``dev200``), θ = mean alignment (2.5 · max cosine) of all noun
    phrases of the reference stories. Check: each reference story is also aligned with the
    images of the next story (sorted by id) that shares no image with it (for VWP: no movie);
    its own images should score clearly higher. Writes ``<run_dir>/groovist_threshold.json``
    and logs the ``eval.groovist_threshold`` value to put in the config.
    """
    if cfg.data.split == "test":
        raise ValueError("calibrate GROOViST on val, never on the test split")
    backend = load_backend(cfg)
    lexicon = load_lexicon(cfg.eval.concreteness_file)
    settings = {**current_settings(cfg), "device": backend.device}
    report = {
        dataset: {
            **calibrate_dataset(eval_references(cfg, dataset), backend, lexicon),
            "settings": settings,
        }
        for dataset in cfg.data.datasets
    }
    out = cfg.run_dir / CALIBRATION_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for dataset, r in report.items():
        log.info(
            "groovist %s: θ=%.4f from %d phrases in %d stories; own images %.3f vs other "
            "%.3f (own wins %.0f%% of %d stories)",
            dataset,
            r["theta"],
            r["n_phrases"],
            r["n_stories"],
            r["own_mean_score"],
            r["other_mean_score"],
            100 * r["own_wins_share"],
            r["n_compared"],
        )
    log.info("put in the config: eval.groovist_threshold: %s", threshold_line(report))
    log.info("wrote %s", out)
    return out


def threshold_line(report: dict[str, dict[str, Any]]) -> str:
    """The ``eval.groovist_threshold`` value as one YAML flow mapping, ready to paste."""
    entries = []
    for dataset, r in report.items():
        fields = {"theta": r["theta"], **r["settings"]}
        entries.append(f"{dataset}: {{" + ", ".join(f"{k}: {v}" for k, v in fields.items()) + "}")
    return "{" + ", ".join(entries) + "}"


def calibrate_dataset(
    refs: list[StoryRecord], backend: Backend, lexicon: dict[str, float]
) -> dict[str, Any]:
    refs = sorted(refs, key=lambda r: r.id)
    sources = [distractor_sources(r) for r in refs]
    partners = [_partner(refs, sources, i) for i in range(len(refs))]
    own: list[list[Phrase]] = []
    other: list[list[Phrase] | None] = []
    for ref, partner in zip(refs, partners, strict=True):
        image_sets = [ref.images] if partner is None else [ref.images, partner.images]
        aligned = story_phrases_on(ref.story, image_sets, backend, lexicon)
        own.append(aligned[0])
        other.append(aligned[1] if partner is not None else None)
    alignments = [p.alignment for phrases in own for p in phrases]
    if not alignments:
        raise ValueError(f"no noun phrases in {len(refs)} reference stories: cannot fix θ")
    theta = sum(alignments) / len(alignments)
    pairs = [
        (story_score(o, theta), story_score(m, theta))
        for o, m in zip(own, other, strict=True)
        if o and m
    ]
    return {
        "theta": theta,
        "n_stories": len(refs),
        "n_phrases": len(alignments),
        "n_compared": len(pairs),
        "own_mean_score": _mean([o for o, _ in pairs]),
        "other_mean_score": _mean([m for _, m in pairs]),
        "own_wins_share": _mean([float(o > m) for o, m in pairs]),
    }


def eval_references(cfg: ExperimentConfig, dataset: str) -> list[StoryRecord]:
    path = Path(cfg.data.jsonl_dir) / f"{dataset}_{cfg.data.split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: the reference stories of {dataset} are needed "
            "(run p13-prepare, or limit data.datasets)"
        )
    refs = read_jsonl(path, StoryRecord)
    if cfg.data.subset != "dev200":
        return refs
    ids_path = subset_path(cfg.data.splits_dir, dataset, cfg.data.split, cfg.data.dev_subset_size)
    if not ids_path.is_file():
        raise FileNotFoundError(f"{ids_path} not found: use --set data.subset=full")
    by_id = {r.id: r for r in refs}
    ids = read_ids(ids_path)
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"{ids_path}: {len(missing)} ids not in {path} (e.g. {missing[0]})")
    return [by_id[i] for i in ids]


VWP_MOVIE_RE = re.compile(r"^(tt\d+)_")


def distractor_sources(ref: StoryRecord) -> set[str]:
    """What a distractor must not share with ``ref``: its movies for VWP (from image file
    names such as ``tt0047396_0010_0_shot_0667_img_0.jpg``), its images otherwise."""
    if ref.dataset != "vwp":
        return set(ref.images)
    movies = set()
    for path in ref.images:
        match = VWP_MOVIE_RE.match(Path(path).name)
        if match is None:
            raise ValueError(
                f"vwp story {ref.id}: image {path} doesn't start with a movie id (tt<digits>_)"
            )
        movies.add(match.group(1))
    return movies


def _partner(refs: list[StoryRecord], sources: list[set[str]], i: int) -> StoryRecord | None:
    for step in range(1, len(refs)):
        j = (i + step) % len(refs)
        if sources[i].isdisjoint(sources[j]):
            return refs[j]
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def select_boxes(
    boxes_cxcywh: np.ndarray, objectness: np.ndarray, width: int, height: int, k: int
) -> list[tuple[int, int, int, int]]:
    """Top-``k`` OWLv2 boxes by objectness after NMS, in pixels of the original image.

    OWLv2 pads the image to a square of side ``max(width, height)`` (bottom / right), so the
    normalised boxes are scaled by that side and then clipped to the image.
    """
    side = max(width, height)
    cx, cy, w, h = (boxes_cxcywh * side).T
    xyxy = np.stack(
        [
            np.clip(cx - w / 2, 0, width),
            np.clip(cy - h / 2, 0, height),
            np.clip(cx + w / 2, 0, width),
            np.clip(cy + h / 2, 0, height),
        ],
        axis=1,
    )
    kept: list[np.ndarray] = []
    for i in np.argsort(-objectness, kind="stable"):
        if len(kept) == k:
            break
        box = xyxy[i]
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            continue
        if all(_iou(box, other) <= NMS_IOU for other in kept):
            kept.append(box)
    return [(int(b[0]), int(b[1]), math.ceil(b[2]), math.ceil(b[3])) for b in kept]


@functools.lru_cache(maxsize=4)
def load_lexicon(path: str) -> dict[str, float]:
    """Brysbaert et al. (2014) concreteness ratings (1–5): ``Word`` → ``Conc.M``."""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(
            f"{file} not found: download the concreteness ratings with "
            "python data/download_concreteness.py (or set eval.concreteness_file)"
        )
    with file.open(encoding="utf-8", newline="") as f:
        return {
            row["Word"].strip().lower(): float(row["Conc.M"])
            for row in csv.DictReader(f, delimiter="\t")
        }


def load_backend(cfg: ExperimentConfig) -> Backend:
    e = cfg.eval
    return _load_backend(
        e.spacy_model,
        e.clip_model,
        e.region_model,
        e.regions_per_image,
        resolve_device(e.device),
        e.dtype,
    )


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@functools.lru_cache(maxsize=1)
def _load_backend(
    spacy_model: str,
    clip_model: str,
    region_model: str,
    regions_per_image: int,
    device: str,
    dtype: str,
) -> Backend:
    return _ModelBackend(spacy_model, clip_model, region_model, regions_per_image, device, dtype)


class _ModelBackend:
    def __init__(
        self,
        spacy_model: str,
        clip_model: str,
        region_model: str,
        regions_per_image: int,
        device: str,
        dtype: str,
    ) -> None:
        import spacy
        import torch
        from transformers import (
            CLIPModel,
            CLIPProcessor,
            Owlv2ForObjectDetection,
            Owlv2Processor,
        )

        log.info(
            "groovist: loading %s, %s, %s on %s in %s",
            spacy_model,
            clip_model,
            region_model,
            device,
            dtype,
        )
        self._torch = torch
        self.device = device
        self._device = device
        self._dtype = getattr(torch, dtype)
        self._k = regions_per_image
        self._nlp = spacy.load(spacy_model)
        self._clip = CLIPModel.from_pretrained(clip_model, dtype=self._dtype).to(device).eval()
        self._clip_proc = CLIPProcessor.from_pretrained(clip_model)
        self._owl = (
            Owlv2ForObjectDetection.from_pretrained(region_model, dtype=self._dtype)
            .to(device)
            .eval()
        )
        self._owl_proc = Owlv2Processor.from_pretrained(region_model)
        self._regions: dict[str, np.ndarray] = {}

    def noun_chunks(self, text: str) -> list[list[Token]]:
        doc = self._nlp(text)
        return [
            [(t.lower_, t.lemma_.lower(), t.pos_, t.i == chunk.root.i) for t in chunk]
            for chunk in doc.noun_chunks
        ]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self._clip_proc(text=texts, padding=True, truncation=True, return_tensors="pt")
        with self._torch.no_grad():
            pooled = self._clip.text_model(
                input_ids=inputs["input_ids"].to(self._device),
                attention_mask=inputs["attention_mask"].to(self._device),
            ).pooler_output
            return self._clip.text_projection(pooled).float().cpu().numpy()

    def embed_regions(self, image_path: str) -> np.ndarray:
        if image_path not in self._regions:
            from PIL import Image

            with Image.open(image_path) as im:
                image = im.convert("RGB")
            crops = [image] + [image.crop(box) for box in self._boxes(image)]
            inputs = self._clip_proc(images=crops, return_tensors="pt")
            with self._torch.no_grad():
                pooled = self._clip.vision_model(
                    pixel_values=inputs["pixel_values"].to(self._device, self._dtype)
                ).pooler_output
                embs = self._clip.visual_projection(pooled).float().cpu().numpy()
            self._regions[image_path] = embs
        return self._regions[image_path]

    def _boxes(self, image: Any) -> list[tuple[int, int, int, int]]:
        if self._k == 0:
            return []
        inputs = self._owl_proc(text=[["an object"]], images=image, return_tensors="pt")
        with self._torch.no_grad():
            out = self._owl(
                **{
                    k: v.to(self._device, self._dtype)
                    if v.is_floating_point()
                    else v.to(self._device)
                    for k, v in inputs.items()
                }
            )
        objectness = out.objectness_logits[0].reshape(-1).float().cpu().numpy()
        boxes = out.pred_boxes[0].float().cpu().numpy()
        return select_boxes(boxes, objectness, image.width, image.height, self._k)


def _normalise(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0
