#!/usr/bin/env python3
"""OmniRAG CLI Pipeline Runner
Executes multi-hop query decomposition, hybrid retrieval, synthesis, and hallucination auditing.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from omnirag.pipeline import OmniRAGEngine


def main():
    parser = argparse.ArgumentParser(description="OmniRAG Multi-Hop Agentic Pipeline Runner")
    parser.add_argument(
        "--query",
        type=str,
        default="How does the coherence time of neutral-atom quantum processors compare to superconducting qubits?",
        help="Query to evaluate",
    )
    args = parser.parse_args()

    rprint(Panel.fit("[bold cyan]⚡ OmniRAG Production Pipeline Execution[/bold cyan]"))

    engine = OmniRAGEngine()
    engine.index()

    rprint(f"\n[bold yellow]Target Query:[/bold yellow] {args.query}\n")
    result = engine.query(args.query)

    # Sub-query plan
    plan_table = Table(title="Decomposed Sub-Queries (DAG)")
    plan_table.add_column("Hop", style="cyan")
    plan_table.add_column("Sub-Query", style="white")
    plan_table.add_column("Target Entity", style="green")
    for hop in result.query_plan.hops:
        plan_table.add_row(str(hop.hop_id), hop.sub_query, hop.target_entity or "N/A")
    rprint(plan_table)

    # Synthesis
    rprint(Panel(result.synthesized_answer, title="[bold green]Synthesized Answer (Citation-Grounded)[/bold green]"))

    # Audit
    audit = result.audit_report
    rprint(f"[bold cyan]Faithfulness Score:[/bold cyan] {audit.faithfulness_score * 100:.1f}%")
    rprint(f"[bold cyan]Hallucination Risk:[/bold cyan] {audit.hallucination_risk}")
    rprint(f"[bold cyan]Verified Claims:[/bold cyan] {len(audit.audited_claims)}")



if __name__ == "__main__":
    main()
