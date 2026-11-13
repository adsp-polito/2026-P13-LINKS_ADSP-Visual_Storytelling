#!/usr/bin/env bash
# One-off GitHub project setup: labels, milestones, project board, starter issues.
# Run it yourself, once, after pushing the scaffold:
#
#   brew install gh && gh auth login
#   gh auth refresh -s project          # Projects need the "project" scope
#   bash .github/bootstrap_github.sh
#
# Safe to re-run for labels/milestones (existing ones are updated/skipped);
# issues are NOT de-duplicated – run that part only once.
set -euo pipefail

REPO="${REPO:-adsp-polito/2026-P13-LINKS_ADSP-Visual_Storytelling}"
OWNER="${REPO%%/*}"
PROJECT_TITLE="${PROJECT_TITLE:-P13 Visual Storytelling – Team 10}"

# Fill in GitHub usernames (without @). Leave empty to create unassigned issues.
PARSA="${PARSA:-parsatimi}"
BEHRAD="${BEHRAD:-behradkamyab}"
NIOOSHA="${NIOOSHA:-NiooshaDe}"
ALL="$(printf '%s,' "$PARSA" "$BEHRAD" "$NIOOSHA" | sed 's/,,*/,/g; s/^,//; s/,$//')"

CP2="CP2 – 2 Dec 2026"
CP3="CP3 – 18 Dec 2026"

# ---------------------------------------------------------------- labels
label() { gh label create "$1" --repo "$REPO" --color "$2" --description "$3" --force; }
label data       1D76DB "Dataset loading / preprocessing"
label model      5319E7 "Inference, training, adapters"
label eval       0E8A16 "Metrics, judge, human check, report"
label infra      BFD4F2 "Tooling, CI, environment, GPU"
label docs       C5DEF5 "Documentation"
label experiment FBCA04 "An experiment run (E0, E1, …)"
label blocked    B60205 "Waiting on something"

# ---------------------------------------------------------------- milestones
milestone() {
  if gh api "repos/$REPO/milestones?state=all" --jq '.[].title' | grep -qxF "$1"; then
    echo "milestone exists: $1"
  else
    gh api "repos/$REPO/milestones" -f title="$1" -f due_on="$2" -f description="$3" >/dev/null
    echo "milestone created: $1"
  fi
}
milestone "$CP2" "2026-12-02T23:59:59Z" "Checkpoint 2: data in JSONL, zero-shot results, first LoRA run, metric pipeline"
milestone "$CP3" "2026-12-18T23:59:59Z" "Checkpoint 3: LoRA grid, LLM judge, human check, final run, report"

# ---------------------------------------------------------------- project board
PROJECT_NUM="$(gh project list --owner "$OWNER" --format json \
  --jq ".projects[] | select(.title==\"$PROJECT_TITLE\") | .number" | head -1)"
if [[ -z "$PROJECT_NUM" ]]; then
  PROJECT_NUM="$(gh project create --owner "$OWNER" --title "$PROJECT_TITLE" --format json --jq .number)"
  gh project link "$PROJECT_NUM" --owner "$OWNER" --repo "$REPO" || true
fi
echo "project #$PROJECT_NUM"

# Default Status field is Todo / In Progress / Done – replace with Todo / In progress / Review / Done.
PROJECT_ID="$(gh project view "$PROJECT_NUM" --owner "$OWNER" --format json --jq .id)"
STATUS_ID="$(gh project field-list "$PROJECT_NUM" --owner "$OWNER" --format json \
  --jq '.fields[] | select(.name=="Status") | .id')"
gh api graphql -f query='
mutation($field: ID!) {
  updateProjectV2Field(input: {fieldId: $field, singleSelectOptions: [
    {name: "Todo",        color: GRAY,   description: ""},
    {name: "In progress", color: YELLOW, description: ""},
    {name: "Review",      color: BLUE,   description: ""},
    {name: "Done",        color: GREEN,  description: ""}
  ]}) { projectV2Field { ... on ProjectV2SingleSelectField { name } } }
}' -f field="$STATUS_ID" >/dev/null \
  || echo "WARN: could not update Status options – add a 'Review' column in the board UI."
: "$PROJECT_ID"

# ---------------------------------------------------------------- issues
# issue <title> <labels> <milestone> <assignees> <owner-name> <body>
issue() {
  local args=(--repo "$REPO" --title "$1" --label "$2" --milestone "$3"
              --body "**Owner:** $5"$'\n\n'"$6")
  [[ -n "$4" ]] && args+=(--assignee "$4")
  local url
  url="$(gh issue create "${args[@]}")"
  gh project item-add "$PROJECT_NUM" --owner "$OWNER" --url "$url" >/dev/null
  echo "created $url"
}

# CP2
issue "VIST → unified JSONL" data "$CP2" "$PARSA" Parsa \
  "Implement \`src/p13/data/vist.py\` (group SIS, drop broken images, resize ≤448px, rewrite placeholders, DII/DIS). Write \`data/processed/vist_{train,val,test}.jsonl\` + \`data/splits/vist_*_dev200.txt\`. Tests in \`tests/test_vist.py\`."
issue "VWP + AESOP → unified JSONL" data "$CP2" "$PARSA" Parsa \
  "Implement \`src/p13/data/vwp.py\` and \`aesop.py\` following the VIST loader; dev-200 id lists; tests."
issue "Dataset comparison notebook" data "$CP2" "$PARSA" Parsa \
  "\`notebooks/01_dataset_comparison.ipynb\`: sequence lengths, story lengths, vocabulary, image domains, placeholder frequency across VIST / VWP / AESOP."
issue "Zero-shot P0/P1 on dev-200" "model,experiment" "$CP2" "$BEHRAD" Behrad \
  "E0: implement \`infer/model_loading.py\` + \`infer/generate.py\`; run \`configs/e0_zeroshot.yaml\` (P0, P1) on dev-200 of all datasets."
issue "Zero-shot on full validation" "model,experiment" "$CP2" "$BEHRAD" Behrad \
  "E0 with \`--set data.subset=full\`. Record runtime / VRAM."
issue "First LoRA run E1 on VIST" "model,experiment" "$CP2" "$BEHRAD" Behrad \
  "Implement \`train/sft.py\` (frozen base, masked-prompt loss); run \`e1_lora_2b_6gb.yaml\` or \`e1_lora_4b_24gb.yaml\` depending on GPU; push adapter (private)."
issue "Metric pipeline: GROOViST + RoViST + dHM" eval "$CP2" "$NIOOSHA" Nioosha \
  "Implement headline metrics in \`eval/metrics.py\` + \`evaluate()\`; scores CSV per item."
issue "Hallucination checks: NER + objects" eval "$CP2" "$NIOOSHA" Nioosha \
  "\`hallucination_rate\`: invented named entities (spaCy NER) + objects not present in images."
issue "Confirm GPU access with mentors" "infra,blocked" "$CP2" "$ALL" "all" \
  "Ask LINKS mentors which GPU we get (6 GB laptop vs 24 GB). Decides 2B-QLoRA vs 4B-LoRA config."

# CP3
issue "LoRA grid E2–E6" "model,experiment" "$CP3" "$BEHRAD" Behrad \
  "Vary rank / target modules / lr / image side / data mix. One config per run."
issue "LLM judge + validation" eval "$CP3" "$NIOOSHA" Nioosha \
  "Implement \`eval/blinding.py\` + \`eval/judge.py\` (Qwen3-VL-32B, order-swapped, A/B/tie/both bad); validate against the human check."
issue "40-sequence human check" eval "$CP3" "$ALL" "all" \
  "Export with \`eval/human_check.py\`, 3 raters, blinded; import + Fleiss' κ."
issue "Final run E7" "model,experiment" "$CP3" "$BEHRAD" Behrad \
  "Best configuration from the grid, full train, full validation generation."
issue "Report: zero-shot vs fine-tuned with 95% bootstrap CIs" "eval,docs" "$CP3" "$NIOOSHA" Nioosha \
  "\`eval/report.py\`: paired bootstrap CIs per metric + judge/human win rates; figures in \`report/figures/\`."
issue "Error analysis" "eval,docs" "$CP3" "$ALL" "all" \
  "Qualitative failure categories (hallucinated entities, order errors, repetition) with examples."

echo "Done. Board: $(gh project view "$PROJECT_NUM" --owner "$OWNER" --format json --jq .url)"
