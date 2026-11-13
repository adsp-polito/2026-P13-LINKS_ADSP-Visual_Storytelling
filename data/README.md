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
| VWP | Movie shots | 5–10 | SIS | TODO |
| AESOP | Clip-art panels | 3 | SIS | TODO |

DII = descriptions of images in isolation · DIS = descriptions of images in sequence ·
SIS = stories of images in sequence (our training target).

## Licensing / GDPR

VIST photos are Flickr images of real people. They are used for research only, are never
redistributed, and must **never be committed** to this repository (enforced by pre-commit
and `tests/test_repo_hygiene.py`). Check each dataset's terms before use.

## Pipeline

```bash
bash data/download_vist.sh          # manual steps documented inside
p13-prepare --config configs/e0_zeroshot.yaml --split all
```
