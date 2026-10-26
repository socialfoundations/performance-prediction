"""Select checkpoints and freeze predictor splits from development measurements."""
from __future__ import annotations

import glob
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pproute.provenance import (
    canonical_json,
    identified as _identified,
    sha256_file as _sha256,
    sha256_text as _sha256_text,
    validate_identity as _validate_record_identity,
)


def _has(path):
    return Path(path).is_file() and Path(path).stat().st_size > 0


def _record_contents(payload):
    return {key: value for key, value in payload.items()
            if key not in ("identity", "code_revision")}


class Selection:
    def __init__(self, task, output, models=None):
        config_root = Path(__file__).resolve().parents[1] / "configs"
        self.task = json.loads((config_root / f"{task}.json").read_text())
        settings = json.loads((config_root / "common.json").read_text())
        # Keep the output path as given (absolute, symlinks untouched): the fitting scripts record
        # os.path.abspath of the paths they receive, and selection compares against those strings.
        self.common = SimpleNamespace(**settings, REPO=str(config_root.parent),
                                      RESULTS=os.path.abspath(str(output)))
        self.name = task
        cohort = json.loads((config_root / "models.json").read_text())
        self.models = tuple(m for m in cohort if models is None or m["name"] in models)
        if not self.models or (models and set(models) != {m["name"] for m in self.models}):
            raise ValueError("Unknown or empty model selection")
        if self.task["trace_path"]:
            self.task["trace_path"] = str(Path(self.common.REPO) / self.task["trace_path"])
        self.model_selection_path = f"{self.common.RESULTS}/model_selections/{task}/selection.json"
        self.predictor_selection_path = f"{self.common.RESULTS}/predictor_selections/{task}/selection.json"
        self.split_manifest_path = f"{self.common.RESULTS}/predictor_splits/{task}.json"


    def development_evidence(self, model, endpoint):
        from scripts.fit_calibrator import endpoint_arrays
        ids, z, raw, source = endpoint_arrays(self.task["task_type"],
            self.scores_dir(model, endpoint), self.conf_file(model, endpoint, "eval"))
        return ids, z.astype(np.float32), raw.astype(np.float32), source


    def model_root(self, model):
        return (f"{self.common.RESULTS}/model_candidates/{self.name}/"
                f"{model['organization']}/{model['model']}")


    def train_dir(self, model, seed):
        return f"{self.model_root(model)}/seed-{seed}/train"


    def checkpoint(self, model, seed, step):
        return f"{self.train_dir(model, seed)}/checkpoint-{step}"


    def candidate_scores_dir(self, model, seed, step):
        return f"{self.model_root(model)}/seed-{seed}/validation/checkpoint-{step}"


    def endpoint_root(self, model, endpoint):
        return (f"{self.common.RESULTS}/endpoints/{self.name}/"
                f"{model['organization']}/{model['model']}/{endpoint}")


    def endpoint_manifest(self, model, endpoint):
        return f"{self.endpoint_root(model, endpoint)}/endpoint.json"


    def scores_dir(self, model, endpoint):
        return f"{self.endpoint_root(model, endpoint)}/scores"


    @staticmethod
    def split_dir(split):
        return "development" if split == "eval" else "test"


    def confidence_root(self, model, endpoint, split):
        return f"{self.endpoint_root(model, endpoint)}/confidence/{self.split_dir(split)}"


    def conf_file(self, model, endpoint, split):
        return f"{self.confidence_root(model, endpoint, split)}/conf_scores.json"


    def calibrator_dir(self, model, endpoint):
        return (f"{self.common.RESULTS}/calibrators/{self.name}/"
                f"{model['organization']}/{model['model']}/{endpoint}")


    def calibrator_path(self, model, endpoint):
        return f"{self.calibrator_dir(model, endpoint)}/calibrator.json"


    def predictor_candidate_dir(self, model, endpoint, tag, seed):
        return (f"{self.common.RESULTS}/predictor_candidates/{self.name}/"
                f"{model['organization']}/{model['model']}/{endpoint}/{tag}/seed-{seed}")


    def selected_checkpoint(self, model):
        payload = json.loads(Path(self.model_selection_path).read_text())
        _validate_record_identity(payload, "model selection")
        return payload["models"][model["name"]]["selected_checkpoint"]


    def read_endpoint_record(self, model, endpoint):
        from pproute.predictor_protocol import read_endpoint_manifest
        return read_endpoint_manifest(
            self.endpoint_manifest(model, endpoint), task=self.name,
            model=model["name"], revision=model["revision"], endpoint=endpoint,
            checkpoint="" if endpoint == "before" else self.selected_checkpoint(model),
        )


    def require_development_only(self):
        if paths := self.leaked_test_artifacts():
            raise SystemExit(f"Cannot change development selection after held-out artifacts: {paths[0]}")


    def leaked_test_artifacts(self):
        # Include stray test outputs, not only canonical endpoint paths.
        patterns = (
            f"{self.common.RESULTS}/model_candidates/{self.name}/**/test_scores.jbl",
            f"{self.common.RESULTS}/endpoints/{self.name}/**/test_scores.jbl",
            f"{self.common.RESULTS}/endpoints/{self.name}/**/confidence/test/**/conf_prompts.json",
            f"{self.common.RESULTS}/endpoints/{self.name}/**/confidence/test/**/conf_scores.json",
            f"{self.common.RESULTS}/predictor_tests/{self.name}/**/metrics.json",
        )
        return sorted({path for pattern in patterns for path in glob.glob(pattern, recursive=True)
                       if _has(path)})


    def git_revision(self):
        return subprocess.check_output(["git", "-C", self.common.REPO, "rev-parse", "HEAD"],
                                       text=True).strip()


    def require_clean_git(self):
        status = subprocess.check_output(
            ["git", "-C", self.common.REPO, "status", "--porcelain", "--untracked-files=normal"],
            text=True,
        )
        if status.strip():
            raise SystemExit("Experiment records require a clean git tree; commit durable changes first")


    @staticmethod
    def write_once(payload, path):
        from pproute.predictor_protocol import atomic_create_json
        if _has(path):
            with open(path, encoding="utf-8") as handle:
                existing = json.load(handle)
            if "identity" in existing:
                _validate_record_identity(existing, "immutable record")
            if _record_contents(existing) != _record_contents(payload):
                raise SystemExit("immutable record differs: %s" % path)
            return False
        atomic_create_json(payload, path)
        return True


    def audit_score_identity(self, model, directory, checkpoint, split="eval"):
        from pproute import scores
        provenance = scores.scored_model_provenance(directory)
        scores.require_requested_model(provenance, model["name"], model["revision"], checkpoint)
        if provenance["task_name"] != self.name:
            raise ValueError("%s records task %s" % (directory, provenance["task_name"]))
        if provenance.get("model_dtype", "auto") != model.get("model_dtype", "auto"):
            raise ValueError("%s records the wrong model dtype" % directory)
        chat = model["prompt_format"] == "chat"
        expected = {
            "random_seed": 0,
            "dataset_seed": 0,
            "fewshot_random_seed": self.task["fewshot_random_seed"],
            "num_fewshot": self.task["num_fewshot"],
            "apply_chat_template": chat,
            "fewshot_as_multiturn": chat,
            "max_new_tokens": self.task["max_new_tokens"],
        }
        protocol = provenance["prompt_protocol"]
        bad = {key: (protocol.get(key), value) for key, value in expected.items()
               if protocol.get(key) != value}
        if bad:
            raise ValueError("%s prompt protocol differs from final settings: %s" % (directory, bad))
        return provenance, scores.recorded_split_fingerprint(directory, split)


    def select_model(self):
        from pproute import analysis, scores
        self.require_clean_git()
        if leaked := self.leaked_test_artifacts():
            raise SystemExit("refusing model selection: held-out artifacts already exist:\n  "
                             + "\n  ".join(leaked))
        rows = {}
        task_fingerprint = None
        for model in self.models:
            candidates = []
            model_fingerprints = set()
            for seed in self.common.MODEL_SEEDS:
                for step in self.task["steps"]:
                    directory = self.candidate_scores_dir(model, seed, step)
                    checkpoint = self.checkpoint(model, seed, step)
                    _, fingerprint = self.audit_score_identity(model, directory, checkpoint)
                    model_fingerprints.add(fingerprint)
                    candidates.append({"scores_dir": directory, "seed": seed, "step": step})
            if len(model_fingerprints) != 1:
                raise ValueError("%s candidates mix development fingerprints: %s" %
                                 (model["name"], sorted(model_fingerprints)))
            fingerprint = next(iter(model_fingerprints))
            fingerprint_hash = _sha256_text(fingerprint)
            if fingerprint_hash != self.task["development_fingerprint_sha256"]:
                raise ValueError(
                    "%s development fingerprint hash %s differs from frozen %s" %
                    (model["name"], fingerprint_hash,
                     self.task["development_fingerprint_sha256"])
                )
            if task_fingerprint is None:
                task_fingerprint = fingerprint
            elif fingerprint != task_fingerprint:
                raise ValueError("development fingerprint differs across models: %s vs %s" %
                                 (task_fingerprint, fingerprint))
            best, table = analysis.select_model_candidate(
                candidates, self.task["model_selection_type"]
            )
            bad_n = [(row["seed"], row["step"], row["n"]) for row in table
                     if int(row["n"]) != self.task["development_rows"]]
            if bad_n:
                raise ValueError("%s candidate validation row counts differ: %s" %
                                 (model["name"], bad_n))
            rows[model["name"]] = {
                "revision": model["revision"], "selected_seed": int(best["seed"]),
                "selected_step": int(best["step"]),
                "selected_checkpoint": self.checkpoint(model, best["seed"], best["step"]),
                "selection_metric": "L_Y" if self.task["model_selection_type"] == "mc"
                else "greedy_accuracy",
                "development_fingerprint": fingerprint,
                "development_fingerprint_sha256": fingerprint_hash,
                "candidates": [{"seed": int(row["seed"]), "step": int(row["step"]),
                                "objective": float(row["objective"]),
                                "n": int(row["n"]),
                                "eval_scores_sha256": _sha256(
                                    f"{row['scores_dir']}/eval_scores.jbl")}
                               for row in table],
            }
            print("  %-45s seed=%d step=%d objective=%.6f" %
                  (model["name"], best["seed"], best["step"], best["objective"]))
        payload = _identified({
            "schema": "pproute-model-selection-v1", "task": self.name,
            "selection_split": "development",
            "development_fingerprint": task_fingerprint,
            "development_fingerprint_sha256": _sha256_text(task_fingerprint),
            "tie_break": ["earlier_step", "lower_training_seed"],
            "code_revision": self.git_revision(), "models": rows,
        })

        # Endpoint records are written first. Downstream stages still require the task-wide
        # selection, so a crash here cannot open a partial cohort; rerunning verifies and finishes.
        for model in self.models:
            selected = rows[model["name"]]
            for endpoint in ("before", "after"):
                checkpoint = "" if endpoint == "before" else selected["selected_checkpoint"]
                source = {
                    "schema": "pproute-endpoint-v1",
                    "task": self.name, "model": model["name"], "revision": model["revision"],
                    "endpoint": endpoint,
                    "selected_seed": None if endpoint == "before" else selected["selected_seed"],
                    "selected_step": None if endpoint == "before" else selected["selected_step"],
                    "checkpoint": checkpoint,
                    "adapter_sha256": None if endpoint == "before" else
                    _sha256(f"{checkpoint}/adapter_model.safetensors"),
                    "adapter_config_sha256": None if endpoint == "before" else
                    _sha256(f"{checkpoint}/adapter_config.json"),
                }
                endpoint_payload = _identified(source)
                self.write_once(endpoint_payload, self.endpoint_manifest(model, endpoint))
        self.write_once(payload, self.model_selection_path)
        print("FROZEN %s" % self.model_selection_path)


    def development_endpoints(self):
        """Validate each development endpoint once and yield its evidence."""
        from pproute import predictor_protocol
        reference = None
        for model in self.models:
            for endpoint in ("before", "after"):
                record = self.read_endpoint_record(model, endpoint)
                fingerprint = predictor_protocol.endpoint_source_fingerprint(
                    self.task["task_type"], self.scores_dir(model, endpoint), "eval",
                    record, self.conf_file(model, endpoint, "eval"))
                evidence = self.development_evidence(model, endpoint)
                ids = evidence[0]
                if _sha256_text(fingerprint) != self.task["development_fingerprint_sha256"]:
                    raise ValueError("Development split differs from the paper configuration")
                if len(ids) != self.task["development_rows"] or (reference is not None and ids != reference):
                    raise ValueError("Development question IDs differ across endpoints")
                reference = ids
                yield model, endpoint, record, evidence


    def development_ids(self):
        reference = None
        for _, _, _, evidence in self.development_endpoints():
            reference = evidence[0]
        return reference


    def freeze_predictor_split(self):
        from pproute import predictor_protocol
        self.require_development_only()
        self.development_ids()
        model = self.models[0]
        source = (f"{self.scores_dir(model, 'before')}/eval_scores.jbl"
                  if self.task["task_type"] == "mc" else self.conf_file(model, "before", "eval"))
        predictor_protocol.freeze_split_manifest(
            self.name, self.task["task_type"], source, self.split_manifest_path,
            self.task["development_rows"],
        )


    def read_candidate(self, model, endpoint, backbone, seed, expected, expected_z, expected_anchor):
        from pproute import predictor_protocol, scores

        directory = self.predictor_candidate_dir(model, endpoint, backbone["tag"], seed)
        validation = f"{directory}/validation.json"
        with open(validation, encoding="utf-8") as handle:
            row = json.load(handle)
        expected = dict(expected, predictor_seed=seed, backbone_tag=backbone["tag"],
                        embedding_model=backbone["name"], embedding_revision=backbone["revision"],
                        checkpoint=f"{directory}/checkpoint/model.pt")
        def same(key, value):
            recorded = row.get(key)
            if key in ("split_manifest", "calibrator", "development_source", "checkpoint") and \
                    isinstance(recorded, str) and isinstance(value, str):
                return os.path.realpath(recorded) == os.path.realpath(value)
            return recorded == value
        bad = {key: (row.get(key), value) for key, value in expected.items() if not same(key, value)}
        if bad:
            raise ValueError("wrong predictor candidate %s: %s" % (validation, bad))
        if _sha256(row["checkpoint"]) != row["checkpoint_sha256"] or row.get("test_artifacts_read") is not False:
            raise ValueError("predictor checkpoint changed or fit is not development-only: %s" % validation)
        z = np.asarray(row.get("selection_z", []), dtype=np.float32)
        g = np.asarray(row.get("selection_g", []), dtype=np.float32)
        anchor = np.asarray(row.get("selection_fz_calibrated", []), dtype=np.float32)
        if any(len(values) != self.task["predictor_selection_rows"] for values in (z, g, anchor)):
            raise ValueError("predictor candidate arrays have wrong length: %s" % validation)
        if not np.array_equal(z, expected_z) or not np.all(np.isfinite(g)) or np.any((g < 0) | (g > 1)):
            raise ValueError("predictor candidate labels or probabilities are invalid: %s" % validation)
        if not np.allclose(anchor, expected_anchor, rtol=0.0, atol=1e-7):
            raise ValueError("predictor Platt-initialized anchor differs: %s" % validation)
        if not np.isclose(scores.bce(z, g), float(row.get("LZ_pred", np.nan)), rtol=0.0, atol=1e-12):
            raise ValueError("predictor validation LZ is not reproducible: %s" % validation)
        predictor_protocol.read_predictor_candidate(validation)
        return dict(row, validation=validation, validation_sha256=_sha256(validation))


    def select_predictor(self):
        from scipy.special import expit

        from pproute import analysis, predictor_protocol

        self.require_clean_git()
        self.require_development_only()
        split = predictor_protocol.read_split_manifest(self.split_manifest_path, task=self.name)
        split_hash = _sha256(self.split_manifest_path)
        rows = []
        for model, endpoint, endpoint_record, evidence in self.development_endpoints():
            ids, z, raw, source = evidence     # coherent log-odds, float32
            predictor_protocol.validate_split_manifest(split, ids=ids)
            by_id = {canonical_json(value): index for index, value in enumerate(ids)}
            selection = np.asarray([by_id[canonical_json(value)] for value in split["selection_ids"]], dtype=int)
            calibrator_path = self.calibrator_path(model, endpoint)
            calibrator = predictor_protocol.read_calibrator(
                calibrator_path, task=self.name, model=model["name"], endpoint=endpoint,
                split_manifest_sha256=split_hash)
            expected_calibrator = {
                "revision": model["revision"], "endpoint_identity": endpoint_record["identity"],
                "n_fit": self.task["predictor_fit_rows"], "fit_ids_sha256": split["fit_ids_sha256"],
                "source_sha256": _sha256(source),
            }
            bad = {k: (calibrator.get(k), v) for k, v in expected_calibrator.items() if calibrator.get(k) != v}
            if bad or os.path.realpath(calibrator.get("source", "")) != os.path.realpath(source):
                raise ValueError("calibrator differs for %s/%s: %s" % (model["name"], endpoint, bad))
            expected_anchor = expit(predictor_protocol.calibrated_logits(raw[selection], calibrator))
            for backbone in self.common.BACKBONES:
                micro = backbone["consistency_micro_batch_size" if self.task["task_type"] == "consistency"
                                 else "mc_micro_batch_size"]
                expected = {
                    "schema": predictor_protocol.CANDIDATE_SCHEMA, "task": self.name,
                    "task_type": self.task["task_type"], "model": model["name"], "revision": model["revision"],
                    "endpoint": endpoint, "endpoint_identity": endpoint_record["identity"],
                    "n_fit": self.task["predictor_fit_rows"], "n_selection": self.task["predictor_selection_rows"],
                    "selection_ids": split["selection_ids"], "selection_ids_sha256": split["selection_ids_sha256"],
                    "split_manifest": os.path.abspath(self.split_manifest_path), "split_manifest_sha256": split_hash,
                    "calibrator": os.path.abspath(calibrator_path), "calibrator_sha256": _sha256(calibrator_path),
                    **({"consistency_estimator": self.common.CONSISTENCY_ESTIMATOR}
                       if self.task["task_type"] == "consistency" else {}),
                    "development_source": os.path.abspath(source), "development_source_sha256": _sha256(source),
                    "training_protocol": predictor_protocol.training_protocol(
                        lora_r=self.common.PREDICTOR_LORA_R,
                        lora_alpha=predictor_protocol.LORA_ALPHA_FACTOR * self.common.PREDICTOR_LORA_R,
                        lora_dropout=predictor_protocol.LORA_DROPOUT,
                        learning_rate=self.common.PREDICTOR_LEARNING_RATE, max_epochs=self.common.PREDICTOR_EPOCHS,
                        effective_batch_size=self.common.PREDICTOR_EFFECTIVE_BATCH, micro_batch_size=micro,
                        max_len=self.task["predictor_max_length"],
                        head_weight_decay=predictor_protocol.HEAD_WEIGHT_DECAY),
                }
                candidates = [self.read_candidate(model, endpoint, backbone, seed, expected, z[selection], expected_anchor)
                              for seed in self.common.PREDICTOR_SEEDS]
                best = analysis.select_predictor_candidate(candidates)
                rows.append({
                    "model": model["name"], "revision": model["revision"], "endpoint": endpoint,
                    "endpoint_identity": endpoint_record["identity"], "backbone_tag": backbone["tag"],
                    "embedding_model": backbone["name"], "embedding_revision": backbone["revision"],
                    "predictor_seed": int(best["predictor_seed"]), "validation_LZ": float(best["LZ_pred"]),
                    "validation": best["validation"], "validation_sha256": best["validation_sha256"],
                    "checkpoint": best["checkpoint"], "checkpoint_sha256": best["checkpoint_sha256"],
                    "candidate_LZ": {str(row["predictor_seed"]): float(row["LZ_pred"]) for row in candidates},
                })
        payload = _identified({
            "schema": predictor_protocol.SELECTION_SCHEMA, "task": self.name,
            "selection_split": "predictor-selection-development", "selection_metric": "L_Z(g)",
            "tie_break": "lower_predictor_seed",
            "predictor_seeds": list(self.common.PREDICTOR_SEEDS),
            **({"consistency_estimator": self.common.CONSISTENCY_ESTIMATOR}
               if self.task["task_type"] == "consistency" else {}),
            "preexisting_test_artifacts": [],
            "split_manifest_sha256": split_hash, "fit_ids_sha256": split["fit_ids_sha256"],
            "selection_ids_sha256": split["selection_ids_sha256"],
            "code_revision": self.git_revision(), "rows": rows,
        })
        self.write_once(payload, self.predictor_selection_path)
        print("FROZEN %s (%d predictors)" % (self.predictor_selection_path, len(rows)))


    def select_strong(self):
        from pproute import scores
        if self.name not in ("winogrande", "commonsense_qa"):
            raise ValueError("Generative tasks use the released strong model")
        self.require_clean_git()
        MODEL = "Qwen/Qwen3.5-122B-A10B"
        REVISION = "dc4d348443bc740c68e2d77492492c11606384d5"
        model = dict(name=MODEL, organization="Qwen", model="Qwen3.5-122B-A10B",
                     revision=REVISION, model_dtype="bfloat16", prompt_format="chat")
        cap = 2000 if self.name == "winogrande" else 1600
        root = Path(self.common.RESULTS) / self.name
        selection_path = root / "selection.json"
        if (root / "test/test_scores.jbl").exists():
            raise ValueError("Cannot select the strong model after test scoring")
        rows = []
        for step in (0, *range(cap // 4, cap + 1, cap // 4)):
            directory = root / f"development/step-{step}"
            checkpoint = "" if step == 0 else self.checkpoint(model, 0, step)
            _, fingerprint = self.audit_score_identity(model, str(directory), checkpoint)
            if _sha256_text(fingerprint) != self.task["development_fingerprint_sha256"]:
                raise ValueError("Strong model development fingerprint differs from final settings")
            metric = scores.self_metrics(str(directory / "eval_scores.jbl"))
            if metric["n"] != self.task["development_rows"]:
                raise ValueError("Wrong development row count")
            if not np.isfinite(metric["L_f"]):
                raise ValueError("Strong model development loss must be finite")
            rows.append(dict(step=step, loss=metric["L_f"], checkpoint=checkpoint,
                             scores_sha256=_sha256(directory / "eval_scores.jbl")))
        selected = min(rows, key=lambda r: (r["loss"], r["step"]))
        if selected["step"] == 0:
            raise ValueError("Released strong model won; this does not reproduce the finetuned-strong protocol")
        checkpoint = Path(selected["checkpoint"])
        payload = _identified(dict(task=self.name, model=MODEL, revision=REVISION, selected=selected,
            candidates=rows, adapter_sha256=_sha256(checkpoint / "adapter_model.safetensors"),
            adapter_config_sha256=_sha256(checkpoint / "adapter_config.json")))
        self.write_once(payload, str(selection_path))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("winogrande", "commonsense_qa", "gsm8k", "mmlu_pro"))
    parser.add_argument("selection", choices=("model", "split", "predictor", "strong"))
    parser.add_argument("--output", type=Path, default=Path("results/reproduction"))
    parser.add_argument("--models", nargs="+", help="Optional subset; keep it fixed for the whole run")
    args = parser.parse_args()
    selection = Selection(args.task, args.output, args.models)
    actions = dict(model=selection.select_model, split=selection.freeze_predictor_split,
                   predictor=selection.select_predictor, strong=selection.select_strong)
    actions[args.selection]()


if __name__ == "__main__":
    main()
