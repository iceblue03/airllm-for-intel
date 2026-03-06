"""Tests for DiagnosticsHistory, PackageManager, and _check_bitsandbytes_intel."""
import importlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_AIRLLM_DIR = os.path.join(os.path.dirname(_TESTS_DIR), "airllm")


def _airllm_file(filename):
    return os.path.join(_AIRLLM_DIR, filename)


def _import_diagnostics():
    """Import diagnostics module fresh."""
    spec = importlib.util.spec_from_file_location(
        "diagnostics_mgr_test",
        _airllm_file("diagnostics.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestDiagnosticsHistory(unittest.TestCase):

    def _make_history(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tmp = f.name
        mod = _import_diagnostics()
        return mod.DiagnosticsHistory(path=tmp), tmp

    def tearDown(self):
        pass

    def test_record_and_load(self):
        mod = _import_diagnostics()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            h = mod.DiagnosticsHistory(path=path)
            h.record("diagnostics_run", {"success": True, "elapsed_sec": 1.23})
            h.record("package_update", {"package": "torch", "from_version": "1.0", "to_version": "2.0", "success": True})
            entries = h.load()
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["event"], "diagnostics_run")
            self.assertEqual(entries[1]["event"], "package_update")
            self.assertIn("timestamp", entries[0])
        finally:
            os.unlink(path)

    def test_load_empty_when_no_file(self):
        mod = _import_diagnostics()
        h = mod.DiagnosticsHistory(path="/tmp/nonexistent_airllm_test_xyz.jsonl")
        entries = h.load()
        self.assertEqual(entries, [])

    def test_print_timeline_empty(self):
        mod = _import_diagnostics()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            h = mod.DiagnosticsHistory(path=path)
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                h.print_timeline()
            self.assertIn("기록된 진단 데이터가 없습니다", buf.getvalue())
        finally:
            os.unlink(path)

    def test_print_timeline_with_entries(self):
        mod = _import_diagnostics()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            h = mod.DiagnosticsHistory(path=path)
            h.record("diagnostics_run", {
                "success": True,
                "elapsed_sec": 0.5,
                "library_versions": {"torch": "2.0.0"},
                "memory_gb": "8.0 GB",
                "errors": [],
            })
            h.record("package_update", {
                "package": "bitsandbytes-intel",
                "from_version": "미설치",
                "to_version": "0.40.0",
                "success": True,
                "error": None,
            })
            h.record("package_downgrade", {
                "package": "torch",
                "from_version": "2.1.0",
                "to_version": "2.0.0",
                "success": True,
                "error": None,
            })
            h.record("package_update_all", {
                "packages": ["torch", "bitsandbytes-intel"],
                "packages_summary": "torch:OK, bitsandbytes-intel:OK",
                "success": True,
            })
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                h.print_timeline()
            output = buf.getvalue()
            self.assertIn("AirLLM 진단 타임라인", output)
            self.assertIn("진단 실행", output)
            self.assertIn("업데이트", output)
            self.assertIn("다운그레이드", output)
            self.assertIn("전체 업데이트", output)
        finally:
            os.unlink(path)

    def test_load_sorts_by_timestamp(self):
        mod = _import_diagnostics()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w", encoding="utf-8") as f:
            path = f.name
            f.write(json.dumps({"timestamp": "2024-01-02T00:00:00Z", "event": "b"}) + "\n")
            f.write(json.dumps({"timestamp": "2024-01-01T00:00:00Z", "event": "a"}) + "\n")
        try:
            h = mod.DiagnosticsHistory(path=path)
            entries = h.load()
            self.assertEqual(entries[0]["event"], "a")
            self.assertEqual(entries[1]["event"], "b")
        finally:
            os.unlink(path)

    def test_load_skips_malformed_lines(self):
        mod = _import_diagnostics()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w", encoding="utf-8") as f:
            path = f.name
            f.write('{"timestamp": "2024-01-01T00:00:00Z", "event": "ok"}\n')
            f.write("not valid json\n")
            f.write('{"timestamp": "2024-01-02T00:00:00Z", "event": "ok2"}\n')
        try:
            h = mod.DiagnosticsHistory(path=path)
            entries = h.load()
            self.assertEqual(len(entries), 2)
        finally:
            os.unlink(path)


class TestPackageManagerUpdate(unittest.TestCase):

    def _make_pm(self, tmp_path):
        mod = _import_diagnostics()
        h = mod.DiagnosticsHistory(path=tmp_path)
        pm = mod.PackageManager(history=h)
        return pm, h, mod

    def test_update_calls_pip_with_upgrade(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            pm, h, mod = self._make_pm(path)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                pm.update("torch")
            call_args = mock_run.call_args[0][0]
            self.assertIn("--upgrade", call_args)
            self.assertIn("torch", call_args)
            self.assertIn("install", call_args)
        finally:
            os.unlink(path)

    def test_downgrade_calls_pip_with_version(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            pm, h, mod = self._make_pm(path)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                pm.downgrade("torch", "2.0.0")
            call_args = mock_run.call_args[0][0]
            self.assertIn("torch==2.0.0", call_args)
            self.assertIn("install", call_args)
        finally:
            os.unlink(path)

    def test_update_all_default_includes_bitsandbytes_intel(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            pm, h, mod = self._make_pm(path)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            with patch("subprocess.run", return_value=mock_result):
                import io
                from contextlib import redirect_stdout
                buf = io.StringIO()
                with redirect_stdout(buf):
                    results = pm.update_all()
            self.assertIn("bitsandbytes-intel", results)
        finally:
            os.unlink(path)

    def test_bitsandbytes_intel_alias(self):
        mod = _import_diagnostics()
        pm = mod.PackageManager.__new__(mod.PackageManager)
        # Test alias resolution
        self.assertEqual(mod.PackageManager._ALIASES.get("bnb-intel"), "bitsandbytes-intel")
        self.assertEqual(mod.PackageManager._ALIASES.get("bitsandbytes-intel"), "bitsandbytes-intel")

    def test_update_records_to_history(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            pm, h, mod = self._make_pm(path)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            with patch("subprocess.run", return_value=mock_result):
                import io
                from contextlib import redirect_stdout
                with redirect_stdout(io.StringIO()):
                    pm.update("psutil")
            entries = h.load()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["event"], "package_update")
            self.assertEqual(entries[0]["package"], "psutil")
        finally:
            os.unlink(path)

    def test_downgrade_records_to_history(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            pm, h, mod = self._make_pm(path)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            with patch("subprocess.run", return_value=mock_result):
                import io
                from contextlib import redirect_stdout
                with redirect_stdout(io.StringIO()):
                    pm.downgrade("psutil", "5.9.0")
            entries = h.load()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["event"], "package_downgrade")
            self.assertEqual(entries[0]["package"], "psutil")
        finally:
            os.unlink(path)

    def test_update_failure_recorded(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            pm, h, mod = self._make_pm(path)
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_result.stderr = "ERROR: package not found"
            with patch("subprocess.run", return_value=mock_result):
                import io
                from contextlib import redirect_stdout
                with redirect_stdout(io.StringIO()):
                    ok = pm.update("nonexistent-package-xyz")
            self.assertFalse(ok)
            entries = h.load()
            self.assertFalse(entries[0]["success"])
        finally:
            os.unlink(path)


class TestBitsandbytesIntelCheck(unittest.TestCase):

    def _import_check(self):
        mod = _import_diagnostics()
        return mod._check_bitsandbytes_intel, mod

    def test_check_detects_installed(self):
        check_fn, mod = self._import_check()
        import importlib.metadata as _imeta
        orig = _imeta.version
        def fake_version(pkg):
            if pkg == "bitsandbytes-intel":
                return "0.40.0"
            return orig(pkg)
        try:
            _imeta.version = fake_version
            name, status, val, detail = check_fn()
            self.assertEqual(name, "bitsandbytes-intel")
            self.assertIn("OK", status)
            self.assertEqual(val, "0.40.0")
            self.assertIsNone(detail)
        finally:
            _imeta.version = orig

    def test_check_warns_when_missing(self):
        check_fn, mod = self._import_check()
        # With no bitsandbytes-intel installed, importlib.metadata.version raises
        import importlib.metadata as _imeta
        orig = _imeta.version
        def fake_version(pkg):
            if pkg == "bitsandbytes-intel":
                raise _imeta.PackageNotFoundError(pkg)
            return orig(pkg)
        try:
            _imeta.version = fake_version
            name, status, val, detail = check_fn()
            self.assertEqual(name, "bitsandbytes-intel")
            self.assertIn("경고", status)
            self.assertIn("미설치", val)
            self.assertIsNotNone(detail)
            self.assertIn("bitsandbytes-intel", detail)
        finally:
            _imeta.version = orig

    def test_check_in_run_diagnostics_results(self):
        """run_diagnostics should include bitsandbytes-intel key."""
        mod = _import_diagnostics()
        import io
        from contextlib import redirect_stdout
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tmp_path = f.name
        try:
            # Patch DiagnosticsHistory to use temp file
            orig_init = mod.DiagnosticsHistory.__init__
            def patched_init(self, path=None):
                orig_init(self, path=tmp_path)
            with patch.object(mod.DiagnosticsHistory, "__init__", patched_init):
                with redirect_stdout(io.StringIO()):
                    result = mod.run_diagnostics(record_history=False)
            self.assertIn("bitsandbytes-intel", result)
        finally:
            os.unlink(tmp_path)


class TestRunDiagnosticsHistory(unittest.TestCase):

    def test_run_diagnostics_records_history(self):
        mod = _import_diagnostics()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            import io
            from contextlib import redirect_stdout

            orig_init = mod.DiagnosticsHistory.__init__
            def patched_init(self, p=None):
                orig_init(self, path=path)

            with patch.object(mod.DiagnosticsHistory, "__init__", patched_init):
                with redirect_stdout(io.StringIO()):
                    mod.run_diagnostics(record_history=True)

            h = mod.DiagnosticsHistory(path=path)
            entries = h.load()
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry["event"], "diagnostics_run")
            self.assertIn("elapsed_sec", entry)
            self.assertIn("library_versions", entry)
            self.assertIn("success", entry)
        finally:
            os.unlink(path)

    def test_run_diagnostics_no_record_when_disabled(self):
        mod = _import_diagnostics()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            import io
            from contextlib import redirect_stdout

            orig_init = mod.DiagnosticsHistory.__init__
            def patched_init(self, p=None):
                orig_init(self, path=path)

            with patch.object(mod.DiagnosticsHistory, "__init__", patched_init):
                with redirect_stdout(io.StringIO()):
                    mod.run_diagnostics(record_history=False)

            h = mod.DiagnosticsHistory(path=path)
            entries = h.load()
            self.assertEqual(len(entries), 0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
