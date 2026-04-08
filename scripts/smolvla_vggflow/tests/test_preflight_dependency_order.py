import unittest
from pathlib import Path


class PreflightDependencyOrderTests(unittest.TestCase):
    def _script_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "preflight_dependency_order.sh"

    def _read_main_body(self, script_text: str) -> str:
        lines = script_text.splitlines()
        main_start = None
        for idx, line in enumerate(lines):
            if line.strip() == "main() {":
                main_start = idx + 1
                break
        self.assertIsNotNone(main_start, "main() function not found")

        body_lines = []
        for line in lines[main_start:]:
            if line == "}":
                break
            body_lines.append(line)
        self.assertTrue(body_lines, "main() body is empty or malformed")
        return "\n".join(body_lines)

    def test_preflight_main_executes_checks_in_dependency_order(self):
        script_path = self._script_path()
        self.assertTrue(script_path.is_file(), "preflight dependency-order script is missing")

        text = script_path.read_text(encoding="utf-8")
        main_body = self._read_main_body(text)

        ordered_calls = [
            "check_01_slurm",
            "check_02_envs",
            "check_03_cuda_render",
            "check_04_jepa",
            "check_05_smolvla",
            "check_06_export",
            "check_07_bridge",
        ]
        positions = []
        for call_name in ordered_calls:
            self.assertIn(call_name, main_body, f"missing main() check call: {call_name}")
            positions.append(main_body.index(call_name))

        self.assertEqual(
            positions,
            sorted(positions),
            "main() must execute dependency checks in deterministic order",
        )

    def test_preflight_marker_constants_exist(self):
        text = self._script_path().read_text(encoding="utf-8")
        for marker in (
            "CHECK_01_SLURM",
            "CHECK_02_ENVS",
            "CHECK_03_CUDA_RENDER",
            "CHECK_04_JEPA",
            "CHECK_05_SMOLVLA",
            "CHECK_06_EXPORT",
            "CHECK_07_BRIDGE",
        ):
            self.assertIn(marker, text, f"missing marker constant: {marker}")


if __name__ == "__main__":
    unittest.main()
