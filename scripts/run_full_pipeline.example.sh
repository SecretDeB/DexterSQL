#!/bin/bash
# DexterSQL — full A-Z pipeline, from scratch on canonical BIRD dev.
#   value_retrieval + vector DB  : builds once, resumes instantly after.
#   link data (profiles + ABC indices): regenerated from canonical BIRD here.
#   schema linking (ABC 5-fix linker)  : run across all questions.
#   generation (dc|icl|dep_tree) -> revision -> selection -> eval.
#
# Reference/example only — this is a plain sequential run. On a shared cluster,
# submit it through your scheduler (Slurm, PBS, ...) instead of running it
# directly; add your own job directives above `set -uo pipefail`.
#
# Fill in every <PLACEHOLDER> below, then:  bash scripts/run_full_pipeline.example.sh
set -uo pipefail

# Activate your Python environment here, e.g.:
#   conda activate <your_env>
#   source <path_to_openai_api_key_env_file>   # needed for the linker's FAISS embeddings

ROOT=<REPO_ROOT>                                   # e.g. $(pwd)
WORK=$ROOT/workspace
RESULTS=$ROOT/results
BIRD_DB=<PATH_TO_BIRD_DEV_DATABASES>                # .../dev_databases
DEV_JSON=<PATH_TO_BIRD_DEV_JSON>                    # .../dev.json
FEWSHOT=<PATH_TO_FEWSHOT_JSON>                      # optional — omit --few-shot below to run zero-shot
export DEXTERSQL_LINK_DATA_ROOT=$WORK/link_data/dev_databases
export CONFIG_PATH=$ROOT/config/bird.toml
cd "$ROOT"

# If using a self-hosted LLM server, wait for it to come up. Adjust or remove
# this block if you're calling a hosted API instead.
SERVER_INFO=<PATH_TO_SERVER_HANDSHAKE_FILE>         # a file containing "host:port"
echo "[wait] LLM server..."
until [ -f "$SERVER_INFO" ] && curl -sf --max-time 10 "http://$(cat "$SERVER_INFO")/v1/models" >/dev/null 2>&1; do sleep 15; done
echo "[server] $(cat "$SERVER_INFO")"

echo "===== 0. value_retrieval (resume/skip) $(date) ====="
python scripts/run_pipeline.py --config config/bird.toml --stages value_retrieval

echo "===== 1. build link data (profiles + ABC indices, from scratch) $(date) ====="
python -m dextersql.linking.build_link_data \
  --bird-db-root "$BIRD_DB" --out-root "$DEXTERSQL_LINK_DATA_ROOT" --backend vllm_server

echo "===== 2. ABC schema linking, PARALLEL across DB groups, $(date) ====="
LINKS=$RESULTS/schema_links_abc_dev.json
LINKS_A=$RESULTS/schema_links_abc_group_a.json
LINKS_B=$RESULTS/schema_links_abc_group_b.json
GROUP_A=$RESULTS/dev_group_a.json                  # question-id list for group A
GROUP_B=$RESULTS/dev_group_b.json                   # question-id list for group B

# Splitting the databases into two disjoint groups and linking them concurrently
# roughly halves wall time — each DB's link data was already built independently
# in stage 1, so there's no shared state between the two processes.
RESUME_A=""; [ -f "$LINKS_A" ] && RESUME_A="--resume $LINKS_A"
RESUME_B=""; [ -f "$LINKS_B" ] && RESUME_B="--resume $LINKS_B"

python dextersql/linking/run_linking.py --all --questions_per_db 0 \
  --questions_file "$GROUP_A" --backend vllm_server --out "$LINKS_A" $RESUME_A \
  > "$RESULTS/linking_group_a.log" 2>&1 &
PID_A=$!
python dextersql/linking/run_linking.py --all --questions_per_db 0 \
  --questions_file "$GROUP_B" --backend vllm_server --out "$LINKS_B" $RESUME_B \
  > "$RESULTS/linking_group_b.log" 2>&1 &
PID_B=$!
echo "  group A pid=$PID_A   group B pid=$PID_B"
wait $PID_A; EC_A=$?
wait $PID_B; EC_B=$?
echo "  group A exit=$EC_A   group B exit=$EC_B"
if [ $EC_A -ne 0 ] || [ $EC_B -ne 0 ]; then
  echo "[FATAL] a linking group failed -- see linking_group_{a,b}.log"; exit 1
fi

echo "===== 2b. merge group A + B links $(date) ====="
python3 -c "
import json
a = json.load(open('$LINKS_A')); b = json.load(open('$LINKS_B'))
merged = {'results': a.get('results', []) + b.get('results', []),
          'errors':  a.get('errors', [])  + b.get('errors', [])}
print('merged:', len(merged['results']), 'results,', len(merged['errors']), 'errors')
json.dump(merged, open('$LINKS', 'w'))
"

echo "===== 3. merge links -> schema_linking snapshot $(date) ====="
python -m dextersql.linking.build_snapshot \
  --src-items "$WORK/value_retrieval/bird/dev.snapshot.data/items.jsonl" \
  --src-meta  "$WORK/value_retrieval/bird/dev.snapshot" \
  --links "$LINKS" --out-dir "$RESULTS/schema_linking/bird"

echo "===== 4. generation / revision / selection (dep_tree) $(date) ====="
python scripts/run_pipeline.py --config config/bird.toml --few-shot "$FEWSHOT" \
  --stages sql_generation sql_revision sql_selection \
  --trace-dir "$RESULTS/traces" --trace-per-db 5

echo "===== 5. evaluate $(date) ====="
python scripts/evaluate.py \
  --snapshot "$RESULTS/sql_selection/bird/dev.snapshot" \
  --out "$RESULTS/perdb.json" \
  --label "DexterSQL (dep_tree, full A-Z from scratch)"
echo "===== DONE $(date) ====="
