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
5. Concreteness: each phrase is weighted by the mean Brysbaert concreteness of its words
   (the word, else its lemma; words not in the list are skipped; 1.0 if none is found).
6. Story score = tanh(mean over phrases of weight · penalised alignment), in (−1, 1).
   A story without noun phrases gets no score.
"""

from __future__ import annotations

import csv
import functools
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from p13.common.config import ExperimentConfig
from p13.common.schema import GeneratedStory, StoryRecord

log = logging.getLogger(__name__)

SCALE = 2.5
DEFAULT_WEIGHT = 1.0
NMS_IOU = 0.5
PLACEHOLDERS = {
    "[male]": "male",
    "[female]": "female",
    "[location]": "location",
    "[organization]": "organization",
}

Token = tuple[str, str, str]


@dataclass(frozen=True)
class Phrase:
    text: str
    weight: float
    alignment: float


class Backend(Protocol):
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
        text = " ".join(word for word, _, _ in chunk)
        if text in seen:
            continue
        seen.add(text)
        kept.append(chunk)
    return kept


def concreteness(chunk: list[Token], lexicon: dict[str, float]) -> float:
    """Mean rating of the words found in the lexicon (word first, then lemma)."""
    found = []
    for word, lemma, _ in chunk:
        rating = lexicon.get(word, lexicon.get(lemma))
        if rating is not None:
            found.append(rating)
    return sum(found) / len(found) if found else DEFAULT_WEIGHT


def max_alignments(phrase_embs: np.ndarray, region_embs: list[np.ndarray]) -> np.ndarray:
    """2.5 · max cosine of each phrase over every region of every image."""
    regions = np.concatenate(region_embs, axis=0)
    return SCALE * (_normalise(phrase_embs) @ _normalise(regions).T).max(axis=1)


def penalise(alignment: float, theta: float) -> float:
    return alignment if alignment >= theta else alignment - theta


def story_score(phrases: list[Phrase], theta: float) -> float:
    if not phrases:
        return math.nan
    values = [p.weight * penalise(p.alignment, theta) for p in phrases]
    return math.tanh(sum(values) / len(values))


def story_phrases(
    story: str, images: list[str], backend: Backend, lexicon: dict[str, float]
) -> list[Phrase]:
    chunks = select_phrases(backend.noun_chunks(clean_text(story)))
    if not chunks:
        return []
    texts = [" ".join(word for word, _, _ in chunk) for chunk in chunks]
    alignments = max_alignments(
        backend.embed_texts(texts), [backend.embed_regions(path) for path in images]
    )
    return [
        Phrase(text, concreteness(chunk, lexicon), float(a))
        for text, chunk, a in zip(texts, chunks, alignments, strict=True)
    ]


def score(
    gens: list[GeneratedStory], refs: dict[Any, StoryRecord], cfg: ExperimentConfig
) -> dict[Any, float]:
    missing = sorted({g.dataset for g in gens} - set(cfg.eval.groovist_threshold))
    if missing:
        raise ValueError(
            f"groovist needs eval.groovist_threshold for {missing}: calibrate it on the "
            "reference stories of that dataset first"
        )
    backend = load_backend(cfg)
    lexicon = load_lexicon(cfg.eval.concreteness_file)
    scores: dict[Any, float] = {}
    for g in gens:
        phrases = story_phrases(g.story, refs[g.key].images, backend, lexicon)
        value = story_score(phrases, cfg.eval.groovist_threshold[g.dataset])
        if not math.isnan(value):
            scores[g.key] = value
    return scores


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
        e.spacy_model, e.clip_model, e.region_model, e.regions_per_image, resolve_device(e.device)
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
    spacy_model: str, clip_model: str, region_model: str, regions_per_image: int, device: str
) -> Backend:
    return _ModelBackend(spacy_model, clip_model, region_model, regions_per_image, device)


class _ModelBackend:
    def __init__(
        self,
        spacy_model: str,
        clip_model: str,
        region_model: str,
        regions_per_image: int,
        device: str,
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
            "groovist: loading %s, %s, %s on %s", spacy_model, clip_model, region_model, device
        )
        self._torch = torch
        self._device = device
        self._k = regions_per_image
        self._nlp = spacy.load(spacy_model)
        self._clip = CLIPModel.from_pretrained(clip_model).to(device).eval()
        self._clip_proc = CLIPProcessor.from_pretrained(clip_model)
        self._owl = Owlv2ForObjectDetection.from_pretrained(region_model).to(device).eval()
        self._owl_proc = Owlv2Processor.from_pretrained(region_model)
        self._regions: dict[str, np.ndarray] = {}

    def noun_chunks(self, text: str) -> list[list[Token]]:
        doc = self._nlp(text)
        return [[(t.lower_, t.lemma_.lower(), t.pos_) for t in chunk] for chunk in doc.noun_chunks]

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
                    pixel_values=inputs["pixel_values"].to(self._device)
                ).pooler_output
                embs = self._clip.visual_projection(pooled).float().cpu().numpy()
            self._regions[image_path] = embs
        return self._regions[image_path]

    def _boxes(self, image: Any) -> list[tuple[int, int, int, int]]:
        if self._k == 0:
            return []
        inputs = self._owl_proc(text=[["an object"]], images=image, return_tensors="pt")
        with self._torch.no_grad():
            out = self._owl(**{k: v.to(self._device) for k, v in inputs.items()})
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
