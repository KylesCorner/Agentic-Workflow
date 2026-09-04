from __future__ import annotations

import subprocess
from pathlib import Path


class ExecutionTools:
    def __init__(self, repo: Path) -> None:
        self.repo = repo.resolve()

    def _run(self, command: list[str], timeout: int = 300) -> str:
        try:
            result = subprocess.run(
                command,
                cwd=self.repo,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            return f"ERROR: command not found: {command[0]}"
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {timeout}s: {' '.join(command)}"
        output = (result.stdout + result.stderr).strip()
        return f"EXIT CODE: {result.returncode}\nCOMMAND: {' '.join(command)}\n\n{output}"

    def run_tests(self, kind: str = "auto") -> str:
        """Run the repository's tests using a constrained known command.

        Args:
            kind: One of auto, pytest, ctest, or platformio.
        """
        kind = kind.lower()
        if kind == "auto":
            if (self.repo / "platformio.ini").exists():
                kind = "platformio"
            elif (self.repo / "build" / "CTestTestfile.cmake").exists():
                kind = "ctest"
            elif (self.repo / "pyproject.toml").exists() or (self.repo / "pytest.ini").exists():
                kind = "pytest"
            else:
                return "ERROR: could not auto-detect a supported test runner."

        if kind == "pytest":
            return self._run(["python", "-m", "pytest"], timeout=300)
        if kind == "ctest":
            return self._run(["ctest", "--test-dir", "build", "--output-on-failure"], timeout=300)
        if kind == "platformio":
            return self._run(["pio", "test"], timeout=600)
        return "ERROR: kind must be auto, pytest, ctest, or platformio."

    def run_build(self, kind: str = "auto") -> str:
        """Run a constrained build/check command.

        Args:
            kind: One of auto, cmake, platformio, or python.
        """
        kind = kind.lower()
        if kind == "auto":
            if (self.repo / "platformio.ini").exists():
                kind = "platformio"
            elif (self.repo / "build").is_dir() and (self.repo / "CMakeLists.txt").exists():
                kind = "cmake"
            elif (self.repo / "pyproject.toml").exists():
                kind = "python"
            else:
                return "ERROR: could not auto-detect a supported build/check command."

        if kind == "cmake":
            return self._run(["cmake", "--build", "build"], timeout=600)
        if kind == "platformio":
            return self._run(["pio", "run"], timeout=600)
        if kind == "python":
            return self._run(["python", "-m", "compileall", "-q", "."], timeout=300)
        return "ERROR: kind must be auto, cmake, platformio, or python."
