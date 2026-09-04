from __future__ import annotations

import subprocess
from pathlib import Path


class DockerTools:
    def __init__(self, repo: Path) -> None:
        self.repo = repo.resolve()

    def _run_command(self, command: list[str], timeout: int = 300) -> str:
        """Run a docker command and return formatted output."""
        try:
            result = subprocess.run(
                command,
                cwd=self.repo,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            return f"ERROR: docker command not found. Is Docker installed and accessible?"
        except subprocess.TimeoutExpired:
            return f"ERROR: docker command timed out after {timeout}s: {' '.join(command)}"
        
        output = (result.stdout + result.stderr).strip()
        if result.returncode == 0:
            return f"EXIT CODE: {result.returncode}\nCOMMAND: {' '.join(command)}\n\n{output}"
        else:
            return f"EXIT CODE: {result.returncode}\nCOMMAND: {' '.join(command)}\n\n{output}"

    def docker_run(self, image: str, command: str = "", options: str = "") -> str:
        """Run a container with the specified image.
        
        Args:
            image: Docker image to run
            command: Command to execute in the container
            options: Additional docker run options
        """
        cmd = ["docker", "run"]
        if options:
            cmd.extend(options.split())
        cmd.append(image)
        if command:
            cmd.append(command)
        return self._run_command(cmd)

    def docker_ps(self, all_containers: bool = False) -> str:
        """List running containers.
        
        Args:
            all_containers: Whether to show all containers (not just running)
        """
        cmd = ["docker", "ps"]
        if all_containers:
            cmd.append("-a")
        return self._run_command(cmd)

    def docker_images(self) -> str:
        """List available Docker images."""
        cmd = ["docker", "images"]
        return self._run_command(cmd)

    def docker_build(self, path: str = ".", tag: str = "", dockerfile: str = "Dockerfile") -> str:
        """Build a Docker image.
        
        Args:
            path: Path to build context
            tag: Tag for the built image
            dockerfile: Dockerfile name (default: Dockerfile)
        """
        cmd = ["docker", "build", "-f", dockerfile, "-t", tag, path] if tag else ["docker", "build", "-f", dockerfile, path]
        return self._run_command(cmd)