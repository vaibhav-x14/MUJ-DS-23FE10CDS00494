"""Interactive Demo Walkthrough for OmniRAG
Step-by-step demonstration of the multi-hop agentic RAG pipeline.
"""

import sys
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
    rprint(Panel.fit("[bold cyan]OmniRAG Step-by-Step Interactive Walkthrough[/bold cyan]"))

    # Step 1: Initialize Engine and Database
    rprint("\n[bold green]1. Initializing OmniRAG Engine & SQLite Database...[/bold green]")
    engine = OmniRAGEngine()
    engine.index()

    stats = engine.db.get_stats()
    rprint(f"Database Stats: {stats['total_documents']} documents, {stats['total_chunks']} chunks, {stats['total_graph_entities']} entities in SQLite.")

    # Step 2: Query Execution
    query = "How does the coherence time of neutral-atom quantum processors compare to superconducting qubits?"
    rprint(f"\n[bold green]2. Executing Multi-Hop Query:[/bold green] '{query}'")
    result = engine.query(query)

    # Step 3: Print Query Plan
    table = Table(title="Decomposed Query Plan")
    table.add_column("Hop ID", style="cyan")
    table.add_column("Sub-Query", style="white")
    table.add_column("Target Entity", style="magenta")
    for h in result.query_plan.hops:
        table.add_row(str(h.hop_id), h.sub_query, h.target_entity or "N/A")
    rprint(table)

    # Step 4: Synthesized Answer
    rprint("\n[bold green]3. Grounded Answer with Citations:[/bold green]")
    rprint(result.synthesized_answer)

    # Step 5: Hallucination Audit
    rprint("\n[bold green]4. Faithfulness & Hallucination Audit:[/bold green]")
    rprint(f"Faithfulness Score: {result.audit_report.faithfulness_score * 100:.1f}%")
    rprint(f"Hallucination Risk: {result.audit_report.hallucination_risk}")
    rprint(f"Summary: {result.audit_report.overall_summary}")

if __name__ == "__main__":
    main()
