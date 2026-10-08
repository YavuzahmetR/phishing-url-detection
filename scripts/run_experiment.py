"""CLI entry point for the existing experiment workflow."""

import sys
from pathlib import Path

# Direct execution puts scripts/ on sys.path; add the repository for imports.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.phishing_guard.experiment import main  # noqa: E402

if __name__ == "__main__":
    main()
