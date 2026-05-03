#!/usr/bin/env bash
# Run DIPN: VinDr → chosen target, 3 seeds (0, 1, 2)
#
# Required:
#   TARGET_RAW   e.g. "INbreast Dataset original"
# Optional:
#   TARGET_KEY   e.g. "inbreast"  (derived automatically if omitted)
#   DEVICE       default cuda:0
#   SEEDS        default "0 1 2"
#   EXP_NAME     default "DIPN_VinDr_to_<TARGET_KEY>"

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

TARGET_RAW="${TARGET_RAW:-}"
if [[ -z "$TARGET_RAW" ]]; then
    echo "Usage: TARGET_RAW='INbreast Dataset original' bash scripts/run.sh" >&2
    exit 1
fi

TARGET_KEY="${TARGET_KEY:-$(echo "$TARGET_RAW" | tr '[:upper:]' '[:lower:]' \
    | sed 's/[^a-z0-9]/_/g; s/__*/_/g; s/^_//; s/_$//')}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS=(${SEEDS:-0 1 2})
EXP_NAME="${EXP_NAME:-DIPN_VinDr_to_${TARGET_KEY}}"
STAGE="${STAGE:-$HERE/data/staging/vindr_to_${TARGET_KEY}}"
LOG_ROOT="${LOG_ROOT:-experiments_logs/$EXP_NAME}"
CONFIG="${CONFIG:-configs/dipn.yaml}"

SRC="$STAGE/vindr_source"
TGT="$STAGE/${TARGET_KEY}_target"
TGT_EVAL="$STAGE/${TARGET_KEY}_target_eval"

if [[ ! -d "$SRC" || ! -d "$TGT" || ! -d "$TGT_EVAL" ]]; then
    echo "ERROR: staging dirs not found under $STAGE" >&2
    echo "  Expected: vindr_source/  ${TARGET_KEY}_target/  ${TARGET_KEY}_target_eval/" >&2
    exit 1
fi

mkdir -p "$LOG_ROOT"

echo "================================================================"
echo "  DIPN  |  VinDr -> $TARGET_RAW"
echo "  device   = $DEVICE"
echo "  seeds    = ${SEEDS[*]}"
echo "  exp_name = $EXP_NAME"
echo "  source   = $SRC"
echo "  target   = $TGT"
echo "  tgt_eval = $TGT_EVAL"
echo "================================================================"

for SEED in "${SEEDS[@]}"; do
    LOG="$LOG_ROOT/run_${SEED}.log"
    echo ""
    echo "  ---- seed=$SEED  →  $LOG  ----"
    PYTHONUNBUFFERED=1 python -u train.py \
        --config         "$CONFIG" \
        --source_dir     "$SRC" \
        --target_dir     "$TGT" \
        --target_eval_dir "$TGT_EVAL" \
        --seed           "$SEED" \
        --save_subdir    "$EXP_NAME/run_${SEED}" \
        --device         "$DEVICE" 2>&1 | tee "$LOG"
    echo "  Done seed=$SEED"
done

echo ""
echo "================================================================"
echo "  All seeds finished."
echo "================================================================"
python - <<'PY'
import json, glob, os, statistics as st
exp = os.environ.get("EXP_NAME","")
files = sorted(glob.glob(f"checkpoints/{exp}/run_*/best_target_metrics.json"))
if not files:
    print("  no best_target_metrics.json files found.")
    raise SystemExit(0)
keys = ["auc","sensitivity","specificity","f1","accuracy"]
agg  = {k: [] for k in keys}
print(f"  {'seed':>6}  {'auc':>7} {'sens':>7} {'spec':>7} {'f1':>7} {'acc':>7}")
for f in files:
    d = json.load(open(f))
    seed = f.split("/")[-2].replace("run_","")
    row  = [d.get(k, float("nan")) for k in keys]
    for k,v in zip(keys,row):
        if v==v: agg[k].append(v)
    print(f"  {seed:>6}  "+" ".join(f"{v:>7.4f}" for v in row))
print(f"  {'mean':>6}  "+" ".join(f"{st.mean(agg[k]) if agg[k] else float('nan'):>7.4f}" for k in keys))
if all(len(agg[k])>=2 for k in keys):
    print(f"  {'std':>6}  "+" ".join(f"{st.pstdev(agg[k]):>7.4f}" for k in keys))
PY
