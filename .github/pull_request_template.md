## What
<!-- One or two sentences. -->

Closes #

## Area
- [ ] data  - [ ] infer  - [ ] train  - [ ] eval  - [ ] common / schema (whole-team review)  - [ ] docs / infra

## How was it tested?
<!-- pytest, a dry-run, a dev-200 run, a notebook… -->

## Checklist
- [ ] `pre-commit run --all-files` and `pytest` pass
- [ ] No images, raw data, outputs, adapter weights or `.env` committed
- [ ] Reads/writes the unified JSONL (`StoryRecord` / `GeneratedStory`)
- [ ] New experiment ⇒ new config file (old configs untouched)
