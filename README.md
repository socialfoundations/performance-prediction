# Predictable Performance is Suboptimal

How much can a separate performance predictor improve on a model’s self-confidence, and how does that advantage change after finetuning? This repository implements the main 4B-predictor experiments for [*Predictable performance is suboptimal, and what it means for model routing*](https://arxiv.org/abs/2610.31428), comparing correctness prediction and post-answer cascades across 16 language models and four tasks.

Predictors use the weak model’s answer and confidence; on generation tasks, they also use a sampled reasoning trace. Cascade results measure accuracy across escalation rates. They do not measure end-to-end latency or compute savings.

[Paper (PDF)](https://arxiv.org/pdf/2610.31428) · [Setup](#setup) · [Predictability experiments](#predictability-experiments) · [Cascade experiments](#cascade-experiments) · [Outputs](#outputs) · [Experimental settings](#experimental-settings)

The workflow is: **finetune → score development data → fit and select predictors → evaluate test data → run cascades**. The commands produce the measurements for the predictability and cascade comparisons before and after finetuning. The release includes the main 4B-predictor protocol; predictor-size ablations, dense training trajectories, and paper figure-generation scripts are outside this release.

Experiment settings live in [`configs/`](configs/), computation entry points in [`scripts/`](scripts/), and shared code in [`pproute/`](pproute/).

## Setup

Use **Python 3.10 exactly, NVIDIA GPUs, Bash, and `jq`**. The pinned `lm-eval`, `vllm`, and `torch` versions are verified only on 3.10. Clone the repository with Git rather than downloading an archive: every run records the current commit and refuses a dirty or untracked working tree. Run from the repository root and keep the same shell open; later commands reuse earlier variables.

```bash
python3.10 --version   # must print Python 3.10.x
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c constraints.txt
export HF_HOME="$PWD/data/huggingface"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
set -euo pipefail
```

Authenticate with `hf auth login` for gated models. Dependencies, including `zarth-utils`, are installed above; core package versions are pinned in [`requirements.txt`](requirements.txt) and [`constraints.txt`](constraints.txt). These files do not lock every transitive dependency.

| Stage | Hardware used |
| --- | --- |
| Weak models and predictors | About 80 GB GPU memory; up to 256 GiB host RAM |
| Strong-model finetuning | Two B200 GPUs; 768 GiB host RAM |
| Strong-model generation | Four GPUs by default |

### Teacher traces

The two frozen teacher-trace archives are stored with [Git LFS](https://git-lfs.com/) under `data/teacher_traces/` (about 3 MB combined). Install Git LFS, then fetch and extract them to the paths specified in the configs. The uncompressed files remain ignored by Git.

```bash
git lfs install --local
git lfs pull
for task in gsm8k mmlu_pro; do
  target=$(jq -r '.trace_path' "configs/$task.json")
  mkdir -p "$(dirname "$target")"
  gzip -dc "data/teacher_traces/$task.jsonl.gz" > "$target"
done
for task in gsm8k mmlu_pro; do
  jq -r '"\(.trace_sha256)  \(.trace_path)"' "configs/$task.json"
done | sha256sum --check
```

Runs record the code commit, so commit code changes before running; untracked files outside the ignored `data/`, `results/`, and `.venv/` directories also block a run. Use fresh output directories for a new experiment.

## Predictability experiments

Complete steps 1–7 for each task: `winogrande`, `commonsense_qa`, `gsm8k`, and `mmlu_pro`. Each run includes all 16 models in [`models.json`](configs/models.json). Expand each command block as needed.

Select models and predictors using development data **before producing any test outputs**.

### 1. Choose a task

Set `task` and the output directory, then load the shared settings:

<details>
<summary>Load task and model settings</summary>

```bash
task=commonsense_qa
run="$PWD/results/reproduction"
config="configs/$task.json"
models_config=configs/models.json

task_type=$(jq -r '.task_type' "$config")
max_steps=$(jq -r '.max_steps' "$config")
sft_length=$(jq -r '.sft_max_length' "$config")
sft_rows=$(jq -r '.sft_rows' "$config")
predictor_length=$(jq -r '.predictor_max_length' "$config")
max_new_tokens=$(jq -r '.max_new_tokens' "$config")
mapfile -t checkpoints < <(jq -r '.steps[]' "$config")
mapfile -t models < <(jq -r '.[].name' "$models_config")
trace_args=()
if [[ "$task_type" == consistency ]]; then
  trace_args=(--sft_jsonl "$PWD/$(jq -r '.trace_path' "$config")" --sft_completion_source teacher)
fi

# Set model-specific variables; this function does not launch any computation.
model_settings() {
  model="$1"
  settings=$(jq -c --arg model "$model" '.[] | select(.name == $model)' "$models_config")
  revision=$(jq -r '.revision' <<< "$settings")
  eval_batch=$(jq -r '.eval_batch_size' <<< "$settings")
  chat_args=()
  if [[ $(jq -r '.prompt_format' <<< "$settings") == chat ]]; then
    chat_args=(--dataset_param.apply_chat_template --dataset_param.fewshot_as_multiturn)
  fi
  stop_args=()
  if [[ "$model" == nvidia/Mistral-NeMo-Minitron-8B-Instruct ]]; then
    stop_args=(--extra_until '<extra_id_1>')
  fi
}
```

</details>

### 2. Finetune and select checkpoints

Train seeds 0–2, evaluate their four checkpoints on development data, then select one checkpoint per model. Keep `--dataset_seed 0` for the fixed data split; `--random_seed` controls the run seed.

<details>
<summary>Finetune, validate, and select checkpoints</summary>

```bash
for model in "${models[@]}"; do
  model_settings "$model"
  for seed in 0 1 2; do
    candidate="$run/model_candidates/$task/$model/seed-$seed"
    python -m scripts.finetune_model \
      --exp_name "$candidate/train" --model_name "$model" --revision "$revision" \
      --task_name "$task" --dataset_seed 0 --random_seed "$seed" \
      --dataset_param.max_num_train -1 --dataset_param.max_num_valid 0 --dataset_param.max_num_test 0 \
      --dataset_param.num_fewshot 5 --dataset_param.completion_only "${chat_args[@]}" \
      --expected_sft_rows "$sft_rows" "${trace_args[@]}" \
      --train_param.max_length "$sft_length" --train_param.max_steps "$max_steps" \
      --train_param.save_steps 0.25 --train_param.logging_steps 25 \
      --train_param.per_device_train_batch_size 1 --train_param.gradient_accumulation_steps 8 \
      --train_param.gradient_checkpointing --train_param.learning_rate 5e-5 \
      --peft_config.r 8 --peft_config.lora_alpha 32 --peft_config.lora_dropout 0.1
    for step in "${checkpoints[@]}"; do
      python -m scripts.evaluate_model \
        --exp_name "$candidate/validation/checkpoint-$step" \
        --base_model_name "$model" --revision "$revision" \
        --dir_checkpoint "$candidate/train/checkpoint-$step" --task_name "$task" \
        --validation_only --dataset_param.max_num_train 0 \
        --dataset_param.max_num_valid -1 --dataset_param.max_num_test 0 \
        --dataset_seed 0 --dataset_param.num_fewshot 5 --fewshot_random_seed 1234 "${chat_args[@]}" \
        --max_new_tokens "$max_new_tokens" --eval_max_batch_size "$eval_batch" "${stop_args[@]}"
    done
  done
done
python -m pproute.selection "$task" model --output "$run"
```

</details>

Selection minimizes development answer loss for WinoGrande/CommonsenseQA and maximizes **greedy development accuracy** for GSM8K/MMLU-Pro. Ties prefer the earlier checkpoint, then lower training seed. The selected checkpoints are saved in endpoint records.

### 3. Score both model versions

First run this block with `split=eval`. After predictor selection in step 6, run the same block with `split=test`.

<details>
<summary>Score released and finetuned models</summary>

```bash
split=eval
if [[ "$split" == eval ]]; then
  split_args=(--validation_only --dataset_param.max_num_valid -1 --dataset_param.max_num_test 0)
else
  test -f "$run/predictor_selections/$task/selection.json"
  split_args=(--dataset_param.max_num_valid 0 --dataset_param.max_num_test -1)
fi
for model in "${models[@]}"; do
  model_settings "$model"
  for endpoint in before after; do
    directory="$run/endpoints/$task/$model/$endpoint"
    checkpoint=$(jq -r '.checkpoint' "$directory/endpoint.json")
    checkpoint_args=()
    [[ -z "$checkpoint" ]] || checkpoint_args=(--dir_checkpoint "$checkpoint")
    python -m scripts.evaluate_model \
      --exp_name "$directory/scores" --base_model_name "$model" --revision "$revision" \
      --task_name "$task" "${checkpoint_args[@]}" --dataset_param.max_num_train 0 "${split_args[@]}" \
      --dataset_seed 0 --dataset_param.num_fewshot 5 --fewshot_random_seed 1234 "${chat_args[@]}" \
      --max_new_tokens "$max_new_tokens" --eval_max_batch_size "$eval_batch" "${stop_args[@]}"
  done
done
```

</details>

### 4. Estimate generation confidence

Skip this section for WinoGrande and CommonsenseQA. For GSM8K/MMLU-Pro, merge each selected adapter once:

```bash
for model in "${models[@]}"; do
  directory="$run/endpoints/$task/$model/after"
  python -m scripts.merge_adapter \
    --exp_name "$directory/merged" --endpoint_manifest "$directory/endpoint.json"
done
```

Sample the split scored in step 3: development now, then test in step 7. It takes 128 samples per question at temperature 0.9 and top-p 1.0, with a 2,048-token output limit. Eight shards are merged in question order.

<details>
<summary>Sample and merge confidence estimates</summary>

```bash
split_name=development
[[ "$split" == eval ]] || split_name=test
n=$(jq -r --arg key "${split_name}_rows" '.[$key]' "$config")
fingerprint=$(jq -r --arg key "${split_name}_fingerprint_sha256" '.[$key]' "$config")
context_length=$(jq -r '.conf_max_model_len' "$config")
for model in "${models[@]}"; do
  for endpoint in before after; do
    directory="$run/endpoints/$task/$model/$endpoint"
    confidence="$directory/confidence/$split_name"
    python -m scripts.conf_prepare \
      --exp_name "$confidence/prepared" --scores_dir "$directory/scores" \
      --task_name "$task" --split "$split" --endpoint_manifest "$directory/endpoint.json" \
      --expected_n "$n" --expected_fingerprint_sha256 "$fingerprint"
    merged_args=()
    [[ "$endpoint" == before ]] || merged_args=(--model_path "$directory/merged")
    for shard in {0..7}; do
      shard_name=$(printf '%02d' "$shard")
      python -m scripts.conf_sample \
        --exp_name "$confidence/shards/$shard_name" --prompts "$confidence/prepared/conf_prompts.json" \
        --endpoint_manifest "$directory/endpoint.json" "${merged_args[@]}" \
        --max_model_len "$context_length" --num_shards 8 --shard_index "$shard"
    done
    eval_n=0; test_n=0
    if [[ "$split" == eval ]]; then eval_n="$n"; else test_n="$n"; fi
    python -m scripts.conf_merge \
      --exp_name "$confidence" --shards_dir "$confidence/shards" --num_shards 8 \
      --expected_eval "$eval_n" --expected_test "$test_n"
  done
done
```

</details>

### 5. Fit predictors

The fixed 85/15 development split uses `default_rng(0)`. A Platt calibrator is fitted only on the predictor-fit partition. Train ten Qwen3-Embedding-4B predictors per endpoint; the initial calibrated anchor is also an early-stopping candidate.

Retain all ten predictor seeds and development-only selection for the paper protocol. One-seed GPU comparisons are diagnostic checks and do not replace this selection procedure.

<details>
<summary>Fit calibrators and predictor candidates</summary>

```bash
python -m pproute.selection "$task" split --output "$run"
embedding=Qwen/Qwen3-Embedding-4B
embedding_revision=5cf2132abc99cad020ac570b19d031efec650f2b
micro_batch=8
[[ "$task_type" == mc ]] || micro_batch=1
for model in "${models[@]}"; do
  for endpoint in before after; do
    directory="$run/endpoints/$task/$model/$endpoint"
    calibrator="$run/calibrators/$task/$model/$endpoint"
    source_args=(--scores_dir "$directory/scores")
    [[ "$task_type" == mc ]] || source_args+=(--conf_file "$directory/confidence/development/conf_scores.json")
    shared_args=(--task_name "$task" --task_type "$task_type" "${source_args[@]}"
      --split_manifest "$run/predictor_splits/$task.json" --endpoint_manifest "$directory/endpoint.json")
    python -m scripts.fit_calibrator --exp_name "$calibrator" "${shared_args[@]}"
    for seed in {0..9}; do
      python -m scripts.fit_predictor \
        --exp_name "$run/predictor_candidates/$task/$model/$endpoint/e4b/seed-$seed" \
        "${shared_args[@]}" --calibrator "$calibrator/calibrator.json" \
        --emb_model "$embedding" --emb_revision "$embedding_revision" --backbone_tag e4b \
        --bs 16 --micro_bs "$micro_batch" --max_len "$predictor_length" \
        --lora_r 16 --epochs 3 --lr 1e-4 --random_seed "$seed"
    done
  done
done
```

</details>

### 6. Select predictors

```bash
python -m pproute.selection "$task" predictor --output "$run"
```

This chooses the lowest predictor-selection loss across seeds 0–9, breaking ties by lower seed. Selection refuses existing held-out test artifacts. Only the selected predictor is evaluated on test.

### 7. Evaluate on test

Rerun the scoring block in step 3 with its first line changed to **`split=test`**. For GSM8K/MMLU-Pro, also rerun the sampling block in step 4; the adapters are already merged. Then evaluate the selected predictors:

<details>
<summary>Evaluate selected predictors</summary>

```bash
for model in "${models[@]}"; do
  for endpoint in before after; do
    directory="$run/endpoints/$task/$model/$endpoint"
    selection="$run/predictor_selections/$task/selection.json"
    seed=$(jq -r --arg model "$model" --arg endpoint "$endpoint" \
      '.rows[] | select(.model == $model and .endpoint == $endpoint) | .predictor_seed' "$selection")
    confidence_args=()
    [[ "$task_type" == mc ]] || confidence_args=(--conf_file "$directory/confidence/test/conf_scores.json")
    python -m scripts.evaluate_predictor \
      --exp_name "$run/predictor_tests/$task/$model/$endpoint/e4b" \
      --task_name "$task" --task_type "$task_type" --scores_dir "$directory/scores" "${confidence_args[@]}" \
      --split_manifest "$run/predictor_splits/$task.json" --endpoint_manifest "$directory/endpoint.json" \
      --calibrator "$run/calibrators/$task/$model/$endpoint/calibrator.json" --predictor_selection "$selection" \
      --emb_model "$embedding" --emb_revision "$embedding_revision" --backbone_tag e4b \
      --micro_bs "$micro_batch" --max_len "$predictor_length" --random_seed "$seed" \
      --expected_n "$(jq -r '.test_rows' "$config")" \
      --expected_fingerprint_sha256 "$(jq -r '.test_fingerprint_sha256' "$config")"
  done
done
```

</details>

Repeat steps 1–7 for the remaining tasks before running the cascade experiments.

## Cascade experiments

The strong model is `Qwen/Qwen3.5-122B-A10B`, pinned to `dc4d348443bc740c68e2d77492492c11606384d5`. Generate its answers with the following commands:

```bash
strong="$PWD/results/strong"
strong_model=Qwen/Qwen3.5-122B-A10B
strong_revision=dc4d348443bc740c68e2d77492492c11606384d5
hf download "$strong_model" --revision "$strong_revision"
```

### WinoGrande and CommonsenseQA

Finetune one seed, select by development answer loss including the released model, and score test. The published selections were steps 1,000 and 800; reruns select using their own measurements.
If the released model wins, selection stops because the rerun has not reproduced the finetuned-strong setup.

<details>
<summary>Finetune, select, and evaluate the strong model</summary>

```bash
for task in winogrande commonsense_qa; do
  config="configs/$task.json"
  steps=2000
  [[ "$task" == winogrande ]] || steps=1600
  train="$strong/model_candidates/$task/$strong_model/seed-0/train"
  python -m scripts.finetune_model \
    --exp_name "$train" --model_name "$strong_model" --revision "$strong_revision" \
    --task_name "$task" --dataset_seed 0 --random_seed 0 --model_dtype bfloat16 --device_map balanced \
    --dataset_param.max_num_train -1 --dataset_param.max_num_valid 0 --dataset_param.max_num_test 0 \
    --dataset_param.num_fewshot 5 --dataset_param.completion_only \
    --dataset_param.apply_chat_template --dataset_param.fewshot_as_multiturn \
    --expected_sft_rows "$(jq -r '.sft_rows' "$config")" \
    --train_param.max_length "$(jq -r '.sft_max_length' "$config")" \
    --train_param.max_steps "$steps" --train_param.save_steps 0.25 --train_param.logging_steps 25 \
    --train_param.per_device_train_batch_size 1 --train_param.gradient_accumulation_steps 8 \
    --train_param.gradient_checkpointing --train_param.learning_rate 5e-5 --train_param.loss_type nll \
    --peft_config.r 8 --peft_config.lora_alpha 32 --peft_config.lora_dropout 0.1
  for step in 0 $((steps/4)) $((steps/2)) $((3*steps/4)) "$steps"; do
    checkpoint_args=()
    [[ "$step" == 0 ]] || checkpoint_args=(--dir_checkpoint "$train/checkpoint-$step")
    python -m scripts.evaluate_model \
      --exp_name "$strong/$task/development/step-$step" --base_model_name "$strong_model" \
      --revision "$strong_revision" --model_dtype bfloat16 "${checkpoint_args[@]}" --task_name "$task" \
      --validation_only --dataset_param.max_num_train 0 --dataset_param.max_num_valid -1 --dataset_param.max_num_test 0 \
      --dataset_seed 0 --dataset_param.num_fewshot 5 --fewshot_random_seed 1234 \
      --dataset_param.apply_chat_template --dataset_param.fewshot_as_multiturn --eval_max_batch_size 1
  done
  python -m pproute.selection "$task" strong --output "$strong"
  checkpoint=$(jq -r '.selected.checkpoint' "$strong/$task/selection.json")
  python -m scripts.evaluate_model \
    --exp_name "$strong/$task/test" --base_model_name "$strong_model" --revision "$strong_revision" \
    --model_dtype bfloat16 --dir_checkpoint "$checkpoint" --task_name "$task" \
    --dataset_param.max_num_train 0 --dataset_param.max_num_valid 0 --dataset_param.max_num_test -1 \
    --dataset_seed 0 --dataset_param.num_fewshot 5 --fewshot_random_seed 1234 \
    --dataset_param.apply_chat_template --dataset_param.fewshot_as_multiturn --eval_max_batch_size 1
done
```

</details>

### GSM8K and MMLU-Pro

Use the released thinking model. Its defaults are 128 samples, temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, and 32,768 output tokens.

<details>
<summary>Sample strong-model answers</summary>

```bash
for task in gsm8k mmlu_pro; do
  config="configs/$task.json"
  python -m scripts.cascade_prepare \
    --exp_name "$strong/prompts/$task" --task_name "$task" --split test \
    --scores_dir "$run/endpoints/$task/meta-llama/Llama-3.1-8B/before/scores" \
    --split_manifest "$run/predictor_splits/$task.json" \
    --expected_n "$(jq -r '.test_rows' "$config")" \
    --expected_fingerprint_sha256 "$(jq -r '.test_fingerprint_sha256' "$config")"
  python -m scripts.cascade_sample \
    --exp_name "$strong/$task" --prompts "$strong/prompts/$task/prompts.json" \
    --model_name "$strong_model" --revision "$strong_revision"
done
```

</details>

## Outputs

| Output | Location |
| --- | --- |
| Selected model checkpoints | `results/reproduction/model_selections/<task>/selection.json` |
| Selected predictor seeds | `results/reproduction/predictor_selections/<task>/selection.json` |
| Predictor test measurements | `results/reproduction/predictor_tests/<task>/<model>/<endpoint>/e4b/metrics.json` |
| Strong-model option scores | `results/strong/<task>/test/test_scores.jbl` |
| Strong-model sampled answers | `results/strong/<task>/answers/*.json` |

The commands above generate per-endpoint measurements and strong-model answers. Aggregate comparisons require the following analysis steps; this release does not include a table or figure exporter.

Each predictor test record contains question IDs, correctness labels, raw confidence, predictor probabilities, and answer losses (`test_ids`, `test_z`, `test_fz_raw`, `test_g`, `test_y_loss`). It also reports aggregate answer loss `L_Y` and correctness losses `LZ_f_raw`, `LZ_f_calibrated`, and `LZ_g`.

For the predictability comparison, excess predictability is `LZ_f_raw - LZ_g`, measured in nats. The record's `edge` field instead measures improvement over the calibrated anchor. For cascades, align strong-model correctness with `test_ids`, then use `pproute.cascade.curve` to calculate accuracy and AUC for raw confidence and predictor probabilities. `pproute.scores.mc_examples` reads strong option scores; sampled-answer records carry correctness in `z`. The routing advantage is `auc_predictor - auc_confidence`; multiply by 100 to express the average accuracy advantage across escalation rates in percentage points. Retain negative values in both comparisons.

## Experimental settings

| Task | SFT rows | Development | Predictor fit / selection | Test | SFT steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| WinoGrande | 32,313 | 8,080 | 6,868 / 1,212 | 1,267 | 2,000 |
| CommonsenseQA | 6,394 | 3,342 | 2,840 / 502 | 1,221 | 1,200 |
| GSM8K | 5,088 | 2,241 | 1,904 / 337 | 1,319 | 1,500 |
| MMLU-Pro | 5,233 | 3,000 | 2,550 / 450 | 3,032 | 1,200 |

Weak-model evaluations and strong-model cascade evaluations use five demonstrations, dataset seed 0, and few-shot seed 1234. Chat models use multi-turn demonstrations; base models use raw prompts. WinoGrande uses native context scoring. Gemma 3 finetunes language attention only. The final settings and pinned revisions are in [configs](configs/).

Predictor LoRA uses alpha 32 and dropout 0.05. Only its head receives weight decay (1e-4). Generative predictors use a 16,384-token ceiling; oversized inputs fail rather than truncate. Confidence and answer loss use the same `bucket-v1` sampled distribution.

The reported comparisons retain pairs with positive development accuracy gain (published counts: 16/16/14/13). Each task's reference is the minimum test answer loss across retained endpoints. Test edges remain signed. Cascades route lower confidence first, average random ordering within exact ties, and count wrong strong answers that replace correct weak answers. Correlation comparisons use Pearson correlation and a one-sided positive-correlation p-value.

## Validation

Run `python -m pytest -q` for the offline tests covering splits, scoring, selection, predictor contracts, and cascade metrics.

Release checks include offline tests, command syntax, and an end-to-end Llama-3.1-8B run on all four tasks with one weak-model training seed, fresh weak-model checkpoints and confidence estimates, and reused strong-model outputs. The full 16-model experiment and a fresh GPU installation have not been rerun.

The single-predictor-seed WinoGrande check also did not reproduce a decrease in excess predictability (−0.0053 → 0.0022 nats); its cascade advantage decreased. This diagnostic does not use the paper’s ten-predictor selection budget.

With one weak-model training seed and all ten predictor seeds, the MMLU-Pro H100 check selected predictor seeds 7 before and 8 after finetuning. Excess predictability was 0.0584 → 0.0587, and cascade AUC advantage (×100) was 0.4844 → 0.5060. Both were approximately flat: this single-model run did not reproduce the expected decreases. Hardware-independent and cross-model reproducibility remain unverified.

A reproducibility issue remains open: confidence-sampling seeds depend on endpoint identities. For finetuned endpoints, these identities include the checkpoint path and adapter hashes, so relocating identical checkpoints can change the samples.

## Citation

If you use this code, please cite [our paper](https://arxiv.org/abs/2610.31428):

```bibtex
@misc{zhang2026predictableperformance,
  title={Predictable performance is suboptimal, and what it means for model routing},
  author={Guanhua Zhang and Moritz Hardt},
  year={2026},
  eprint={2610.31428},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2610.31428}
}
```

## License

This code is released under the [MIT License](LICENSE).
