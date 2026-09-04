from __future__ import annotations

import subprocess
from pathlib import Path


class UnixTools:
    def __init__(self, repo: Path) -> None:
        self.repo = repo.resolve()

    def _run_command(self, command: list[str], timeout: int = 300) -> str:
        """Run a shell command and return formatted output."""
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
        if result.returncode == 0:
            return f"EXIT CODE: {result.returncode}\nCOMMAND: {' '.join(command)}\n\n{output}"
        else:
            return f"EXIT CODE: {result.returncode}\nCOMMAND: {' '.join(command)}\n\n{output}"

    def grep(self, pattern: str, file_pattern: str = "*", recursive: bool = True) -> str:
        """Search for a pattern in files using grep.
        
        Args:
            pattern: The search pattern
            file_pattern: File pattern to search (default: "*")
            recursive: Whether to search recursively (default: True)
        """
        cmd = ["grep", "-n", pattern]
        if recursive:
            cmd.append("-r")
        cmd.append(file_pattern)
        return self._run_command(cmd)

    def find(self, path: str = ".", name: str = "*", type_: str = "f") -> str:
        """Find files matching criteria using find.
        
        Args:
            path: Directory to search in
            name: File name pattern to match
            type_: Type of file to find (f for files, d for directories)
        """
        cmd = ["find", path, "-type", type_]
        if name != "*":
            cmd.extend(["-name", name])
        return self._run_command(cmd)

    def sed(self, pattern: str, replacement: str, file_pattern: str = "*", inplace: bool = False) -> str:
        """Perform text substitution using sed.
        
        Args:
            pattern: Pattern to match
            replacement: Replacement text
            file_pattern: File pattern to process
            inplace: Whether to modify files in-place
        """
        cmd = ["sed", "-e", f"s/{pattern}/{replacement}/g"]
        if inplace:
            cmd.append("-i")
        cmd.append(file_pattern)
        return self._run_command(cmd)