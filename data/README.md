# data/

Only scripts, documentation and **ID lists** are committed here. Everything else is gitignored:

```
data/raw/<dataset>/        original downloads (annotations + images)     – gitignored
data/images/<dataset>/     resized images (long side ≤ 448 px)           – gitignored
data/processed/            unified JSONL: <dataset>_<split>.jsonl        – gitignored
data/splits/               dev-200 id lists: <dataset>_<split>_dev200.txt – COMMITTED
```

## Datasets

| Dataset | Content | Images / story | Texts | Where to get it (owner: Parsa – fill in) |
|---|---|---|---|---|
| VIST | Flickr photo albums | 5 | DII, DIS, SIS | https://visionandlanguage.net/VIST/ – see `download_vist.sh` |
| VWP | Movie shots | 5–9 | SIS | https://huggingface.co/datasets/tonyhong/vwp (v2.1) – see `download_vwp.py` |
| AESOP | Clip-art panels | 3 | SIS | TODO |

Eval resource: `python data/download_concreteness.py` writes the Brysbaert et al. (2014)
concreteness ratings (official supplementary .xlsx, converted) to
`data/raw/concreteness/brysbaert2014.tsv` for GROOViST.

DII = descriptions of images in isolation · DIS = descriptions of images in sequence ·
SIS = stories of images in sequence (our training target).

## VWP (Visual Writing Prompts)

- Paper: Hong et al., TACL 2023 – https://vwprompt.github.io/
- Files: `vwp_v2.1_{train,val,test}.csv` from Hugging Face; shot images are linked in the
  CSV (`link0`–`link9`, MPI-INF server). `python data/download_vwp.py --split val` gets both
  into `data/raw/vwp/` (images: `data/raw/vwp/images/<split>/<folder>_<shot file>.jpg`; a
  shot can sit in two scene folders, so the folder is part of the name).
- Text: `anonymised_story` (names → `[male0]`, `[female1]` …), rewritten to "a man",
  "another man", "the first man" …; character names the anonymiser missed → "a person".
- Each story has its own shots (writers picked different shots of the same scene), so the
  images of a record are that row's links, in order.
- Dropped: rows whose links do not match `img_id_list` and the text segments (mostly the
  CSV lost the 10th link; 15 in val, 373 in train), duplicates with the same images and the
  same anonymised text (2,294 in train), and the usual length / missing / broken checks. The test CSV is blind (no stories): VWP is val-only for eval.
- dev-200 (val): at most one story per scene.

## Licensing / GDPR

VIST photos are Flickr images of real people. They are used for research only, are never
redistributed, and must **never be committed** to this repository (enforced by pre-commit
and `tests/test_repo_hygiene.py`). Check each dataset's terms before use.

VWP stories are Apache-2.0; the movie shots belong to their copyright holders (see each
`imdb_id`) and are for academic use only. The same rule applies: never commit them.

## Pipeline

```bash
bash data/download_vist.sh                  # manual steps documented inside
python data/download_vwp.py --split val     # CSV + shot images (needs the venv)
p13-prepare --config configs/e0_zeroshot.yaml --split val  --set "data.datasets=[vist,vwp]"
p13-prepare --config configs/e0_zeroshot.yaml --split test --set "data.datasets=[vist]"
```

Run one split at a time: `--split all` runs train first (it fails if the train images are not
downloaded) and VWP refuses its blind test split.
