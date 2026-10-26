"""pproute: per-prompt routing / "Predictable Performance is Suboptimal" support modules.

  tasks      - task datasets for finetuning/eval (lm-eval-harness), from finetune_model/evaluate_model
  models     - loading the model under test: which auto class, dtype, pad token (all entry scripts)
  evaluator  - the Evaluator used by evaluate_model; stamps the gold option index into the scores
  scores     - read lm_eval score files -> per-prompt (L, Z, f_Z) + bce; shared everywhere
  predictor  - the LoRA-embedding correctness predictor g (mc + consistency) + training
  consistency - sampled-answer distribution aggregation and artifact validation
  predictor_protocol - immutable endpoint/split/calibrator/selection records
  provenance - artifact hashing, atomic publication, and pinned model revisions
  analysis   - the development-only model and predictor selection rules
"""
