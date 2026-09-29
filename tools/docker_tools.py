from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, Optional


def get_container_status(container_name: str) -> Dict[str, str]:
    """Returns a safe status record for a known container."""
    known_statuses = {
        "payment-api": "running",
        "mysql": "exited",
    }
    status = known_statuses.get(container_name, "unknown")
    return {"name": container_name, "status": status}


def get_container_logs(container_name: str, tail: int = 50) -> str:
    """Attempts to read logs from Docker if the CLI is available."""
    if not shutil.which("docker"):
        return f"No docker CLI available for {container_name}. Using simulated status."

    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(tail), container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout or result.stderr or "No logs captured."
    except (FileNotFoundError, OSError):
        return f"Docker logs unavailable for {container_name}."


def restart_container(container_name: str) -> Dict[str, str]:
    """Executes only a predefined, allowed action."""
    allowed = {"payment-api", "mysql"}
    if container_name not in allowed:
        raise ValueError(f"Container {container_name} is not in the approved restart list.")

    if not shutil.which("docker"):
        return {"container": container_name, "status": "simulated_restart", "message": "Restart action approved and queued in the demo environment."}

    try:
        subprocess.run(["docker", "restart", container_name], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {"container": container_name, "status": "restarted", "message": f"Container {container_name} restarted successfully."}
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return {"container": container_name, "status": "simulated_restart", "message": "Docker daemon is unavailable; the action was approved but not executed in this local environment."}


def health_check(service_name: str) -> Dict[str, str]:
    """Checks a health endpoint or known state."""
    if service_name == "payment-api":
        is_healthy = os.getenv("DEEPSHIELD_DEMO_MODE", "true").lower() == "true"
        return {"service": service_name, "health": "healthy" if is_healthy else "unhealthy"}
    return {"service": service_name, "health": "unknown"}
