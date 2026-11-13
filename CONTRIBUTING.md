# Contributing

## Branches
- `main` is protected: always runnable, CI green.
- Work on `<name>/<topic>` branches, e.g. `parsa/vist-loader`, `behrad/e1-lora`,
  `nioosha/groovist`.

## Pull requests
- Open a PR into `main`; **1 approving review** required (CODEOWNERS requests the right person).
- Link the issue (`Closes #12`) and fill in the PR template.
- Changes to `src/p13/common/` (schema, config) or `pyproject.toml` affect everyone –
  tag the whole team.
- CI (ruff + pytest on CPU) must pass.

## Commit messages – [Conventional Commits](https://www.conventionalcommits.org/)
```
<type>(<scope>): <summary>

feat(data): add VWP loader
fix(eval): handle empty stories in distinct-2
exp(train): E2 rank-32 config
docs(readme): GPU switching notes
```
Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`, `exp` (experiment configs /
results). Scopes: `data`, `infer`, `train`, `eval`, `common`, `configs`, `report`.

## Experiments
- One config per experiment in `configs/` named `e<N>_<short_name>.yaml`; open an
  *Experiment* issue describing hypothesis, config and results.
- Never edit a config after its results are reported – copy it to a new id instead.
- Same for prompts: add a new id in `configs/prompts.yaml`, never rewrite an old one.

## Before pushing
```bash
uv run pre-commit run --all-files
uv run pytest -q
```

## Data
Never commit images, raw dataset files, outputs, adapters or `.env`. See README › Data policy.
