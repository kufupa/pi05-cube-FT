import re
import unittest
from pathlib import Path


class StageWiringContractTests(unittest.TestCase):
    def test_phase07_passes_quality_and_storage_flags(self):
        text = Path("scripts/smolvla_vggflow/run_stage.sh").read_text(encoding="utf-8")
        self.assertIn("--max-wm-error-rate", text)
        self.assertIn("--max-policy-error-rate", text)
        self.assertIn("--require-images", text)
        self.assertIn("--episodes-per-shard", text)
        self.assertIn("--max-rss-gb", text)
        self.assertIn("--rss-log-interval-steps", text)
        self.assertIn("--execution-policy", text)
        self.assertIn("--store-cem-plan-seq", text)
        self.assertIn("--store-smolvla-action", text)

    def test_phase08_bridge_passes_wm_heavy_flags(self):
        text = Path("scripts/smolvla_vggflow/run_stage.sh").read_text(encoding="utf-8")
        self.assertIn("--wm-heavy-split-enabled '${SMOLVLA_BRIDGE_WM_HEAVY_SPLIT}'", text)
        self.assertIn("--wm-heavy-val-fraction", text)
        self.assertIn("--wm-heavy-score-margin", text)
        self.assertIn("--fail-on-path-reuse '${SMOLVLA_FAIL_ON_PATH_REUSE}'", text)
        self.assertIn("SMOLVLA_BRIDGE_WM_HEAVY_SPLIT", text)
        self.assertIn("SMOLVLA_BRIDGE_WM_HEAVY_JEPA_FRACTION", text)
        self.assertIn("SMOLVLA_BRIDGE_WM_SCORE_MARGIN", text)
        self.assertIn("phase08 bridge overwrite guard", text)
        self.assertIn("bridge_summary.json", text)
        self.assertIn("train/meta", text)
        self.assertIn("val/meta", text)

    def test_config_defaults_strict_thresholds_and_full_latents(self):
        cfg = Path("scripts/smolvla_vggflow/config.sh").read_text(encoding="utf-8")
        self.assertIn("SMOLVLA_JEPA_EXPORT_MAX_WM_ERROR_RATE", cfg)
        self.assertIn("SMOLVLA_JEPA_EXPORT_MAX_POLICY_ERROR_RATE", cfg)
        self.assertIn("SMOLVLA_JEPA_EXPORT_REQUIRE_IMAGES", cfg)
        self.assertIn("SMOLVLA_JEPA_EXPORT_EXECUTION_POLICY", cfg)
        self.assertIn("SMOLVLA_JEPA_EXPORT_STORE_CEM_PLAN_SEQ", cfg)
        self.assertIn("SMOLVLA_JEPA_EXPORT_STORE_SMOLVLA_ACTION", cfg)
        self.assertIn("SMOLVLA_JEPA_EXPORT_FULL_LATENTS", cfg)
        self.assertIn('SMOLVLA_TRAIN_SAVE_STEPS="${SMOLVLA_TRAIN_SAVE_STEPS:-2000}"', cfg)
        self.assertIn("SMOLVLA_BRIDGE_WM_HEAVY_SPLIT", cfg)
        self.assertIn("SMOLVLA_BRIDGE_WM_HEAVY_SPLIT_ENABLED", cfg)
        self.assertIn('SMOLVLA_BRIDGE_WM_HEAVY_SPLIT="${SMOLVLA_BRIDGE_WM_HEAVY_SPLIT:-${SMOLVLA_BRIDGE_WM_HEAVY_SPLIT_ENABLED:-1}}"', cfg)
        self.assertIn('SMOLVLA_BRIDGE_WM_HEAVY_SPLIT_ENABLED="${SMOLVLA_BRIDGE_WM_HEAVY_SPLIT_ENABLED:-${SMOLVLA_BRIDGE_WM_HEAVY_SPLIT}}"', cfg)
        self.assertIn("SMOLVLA_BRIDGE_WM_HEAVY_JEPA_FRACTION", cfg)
        self.assertIn('SMOLVLA_BRIDGE_WM_HEAVY_JEPA_FRACTION="${SMOLVLA_BRIDGE_WM_HEAVY_JEPA_FRACTION:-${SMOLVLA_BRIDGE_WM_HEAVY_VAL_FRACTION:-0.60}}"', cfg)
        self.assertIn("SMOLVLA_BRIDGE_WM_SCORE_MARGIN", cfg)

    def test_exporter_defines_storage_argparse_and_manifest_fields(self):
        exporter = Path("scripts/smolvla_vggflow/jepa_cem_paired_pushv3_export.py").read_text(encoding="utf-8")
        self.assertIn('"--store-cem-plan-seq"', exporter)
        self.assertIn('"--store-smolvla-action"', exporter)
        self.assertIn('"--full-latents-export"', exporter)
        self.assertIn("store_cem_plan_seq = _as_bool(args.store_cem_plan_seq)", exporter)
        self.assertIn("store_smolvla_action = _as_bool(args.store_smolvla_action)", exporter)
        self.assertIn("full_latents_export = _as_bool(args.full_latents_export)", exporter)
        self.assertIn('"store_cem_plan_seq"', exporter)
        self.assertIn('"store_smolvla_action"', exporter)
        self.assertIn('"full_latents_exported"', exporter)
        self.assertIn('"latent_pred_dim"', exporter)

    def test_stage09_autoselect_handles_init_checkpoint_default(self):
        stage09 = Path("scripts/slurm/stage09_final_eval_and_bundle.slurm").read_text(encoding="utf-8")
        self.assertIn('[[ -z "${SMOLVLA_FINAL_EVAL_CHECKPOINT:-}" || "${SMOLVLA_FINAL_EVAL_CHECKPOINT}" == "${SMOLVLA_INIT_CHECKPOINT}" ]]', stage09)
        self.assertNotIn("006000", stage09)
        self.assertIn("checkpoints/*/pretrained_model", stage09)
        self.assertIn('"${PYTHON_BIN}" - <<\'PY\'', stage09)
        self.assertNotIn(" python - <<'PY'", stage09)

    def test_common_scope_reuse_guard_is_non_blocking(self):
        common = Path("scripts/smolvla_vggflow/common.sh").read_text(encoding="utf-8")
        self.assertIn("non-blocking for staged execution", common)
        for scoped_var in ("_scoped_art", "_scoped_data_root", "_scoped_jepa_export_out"):
            pattern = (
                rf'\[\[ "\$\{{SMOLVLA_FAIL_ON_PATH_REUSE:-0\}}" == "1" \]\] && '
                rf'\[\[ -e "\$\{{{scoped_var}\}}" \]\]; then[\s\S]*?exit 2'
            )
            self.assertIsNone(re.search(pattern, common), msg=f"blocking root guard present for {scoped_var}")


if __name__ == "__main__":
    unittest.main()
