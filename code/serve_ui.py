#!/usr/bin/env python3
"""OmniRAG Web Dashboard Launcher
Serves the FastAPI backend and glassmorphism interface on port 8000.
"""

import sys
import uvicorn
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("Starting OmniRAG Web UI at http://localhost:8000 ...")
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
