#!/usr/bin/env python3
"""OmniRAG Benchmark Suite Runner
Evaluates recall, precision, and faithfulness across multi-hop scenarios.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.eval_suite import OmniRAGEvalSuite


def main():
    suite = OmniRAGEvalSuite()
    report = suite.run_benchmark()
    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
