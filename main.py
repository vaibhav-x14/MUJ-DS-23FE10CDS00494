#!/usr/bin/env python3
"""OmniRAG: Multi-Hop Agentic RAG System
CLI Entrypoint and Execution Dispatcher.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from omnirag.pipeline import OmniRAGEngine
from omnirag.core.logger import console, logger
from prompts.prompt_manager import prompt_catalog
from config.settings import settings


def cmd_query(args):
    """Executes a multi-hop query and prints detailed rich inspection."""
    query_text = " ".join(args.query_text) if isinstance(args.query_text, list) else args.query_text
    if not query_text:
        console.print("[bold red]Error: Please provide a query string.[/bold red]")
        sys.exit(1)

    console.print(Panel.fit(f"[bold cyan]OmniRAG Autonomous Reasoning Engine[/bold cyan]\n"
                            f"User Query: [italic white]{query_text}[/italic white]"))

    engine = OmniRAGEngine()
    result = engine.query(query_text)

    # 1. Query Plan Table
    plan_table = Table(title="Step 1: Multi-Hop Query Decomposition Plan")
    plan_table.add_column("Hop ID", style="cyan", no_wrap=True)
    plan_table.add_column("Sub-Query", style="white")
    plan_table.add_column("Target Entity", style="magenta")
    plan_table.add_column("Dependencies", style="yellow")
    plan_table.add_column("Status", style="green")

    for h in result.query_plan.hops:
        deps = ", ".join([f"Hop {d}" for d in h.depends_on]) if h.depends_on else "None (Root)"
        plan_table.add_row(str(h.hop_id), h.sub_query, h.target_entity or "N/A", deps, h.status)
    console.print(plan_table)

    # 2. Retrieved Chunks Table
    chunk_table = Table(title=f"Step 2: Retrieved & Verified Evidence Chunks ({len(result.retrieved_chunks)})")
    chunk_table.add_column("Chunk ID", style="cyan")
    chunk_table.add_column("Document", style="blue")
    chunk_table.add_column("BM25 Score", style="yellow")
    chunk_table.add_column("Dense Score", style="yellow")
    chunk_table.add_column("RRF Score", style="green")
    chunk_table.add_column("Self-RAG Verdict", style="bold green")

    for c in result.retrieved_chunks:
        chunk_table.add_row(
            c.chunk_id,
            c.doc_id,
            f"{c.bm25_score:.3f}",
            f"{c.dense_score:.3f}",
            f"{c.rrf_score:.5f}",
            c.relevance_verdict or "VERIFIED",
        )
    console.print(chunk_table)

    # 3. Knowledge Graph Triples
    if result.knowledge_graph.edges:
        kg_table = Table(title=f"Step 3: Knowledge Graph Relational Context ({len(result.knowledge_graph.edges)} Edges)")
        kg_table.add_column("Subject Entity", style="magenta")
        kg_table.add_column("Predicate (Relation)", style="cyan")
        kg_table.add_column("Object Entity", style="green")
        for e in result.knowledge_graph.edges[:6]:
            kg_table.add_row(e.source, e.predicate, e.target)
        console.print(kg_table)

    # 4. Synthesized Grounded Answer
    console.print("\n[bold green]Step 4: Grounded Multi-Hop Synthesized Answer[/bold green]")
    console.print(Markdown(result.synthesized_answer))

    # 5. Hallucination & Faithfulness Audit
    audit_table = Table(title="Step 5: Hallucination & Faithfulness Self-Audit")
    audit_table.add_column("Metric", style="bold white")
    audit_table.add_column("Audit Finding", style="bold yellow")

    audit_table.add_row("Total Factual Assertions Audited", str(result.audit_report.total_claims))
    audit_table.add_row("Entailed (Verified Grounded) Claims", str(result.audit_report.entailed_claims_count))
    audit_table.add_row("Unverified Claims", str(result.audit_report.unverified_claims_count))
    audit_table.add_row("Faithfulness Score", f"{result.audit_report.faithfulness_score * 100:.1f}%")
    risk_color = "green" if result.audit_report.hallucination_risk == "LOW" else "red"
    audit_table.add_row("Hallucination Risk Level", f"[{risk_color}]{result.audit_report.hallucination_risk}[/{risk_color}]")
    console.print(audit_table)

    # 6. Telemetry & Cost Box
    console.print(
        f"[dim]Execution Latency: {result.execution_time_seconds}s | Prompt Tokens: {result.usage.prompt_tokens} | "
        f"Completion Tokens: {result.usage.completion_tokens} | Model: {result.usage.model}[/dim]\n"
    )


def cmd_benchmark(args):
    """Runs the benchmark suite."""
    from evaluation.eval_suite import OmniRAGEvalSuite
    suite = OmniRAGEvalSuite()
    suite.run_benchmark()


def cmd_prompts(args):
    """Lists all available versioned prompt templates in catalog."""
    templates = prompt_catalog.list_templates()
    table = Table(title="OmniRAG Versioned Prompt Catalog")
    table.add_column("Prompt Name / Key", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Output Format", style="yellow")
    table.add_column("Description", style="white")

    for name, meta in templates.items():
        table.add_row(name, meta["version"], meta["output_format"], meta["description"])
    console.print(table)


def cmd_web(args):
    """Starts the FastAPI Web Dashboard."""
    import uvicorn
    from web.server import app
    port = args.port or settings.web.port
    host = args.host or settings.web.host
    console.print(Panel.fit(f"[bold green]Starting OmniRAG Web UI & API Server[/bold green]\n"
                            f"Access URL: [bold underline cyan]http://localhost:{port}[/bold underline cyan]\n"
                            f"Swagger Docs: [bold underline cyan]http://localhost:{port}/docs[/bold underline cyan]"))
    uvicorn.run("web.server:app", host=host, port=port, reload=False)


def main():
    parser = argparse.ArgumentParser(
        description="OmniRAG: Multi-Hop Agentic RAG System with Self-Reflection & Graph Augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Query command
    p_query = subparsers.add_parser("query", help="Execute an autonomous multi-hop query")
    p_query.add_argument("query_text", nargs="+", help="User question to answer")

    # Benchmark command
    subparsers.add_parser("benchmark", help="Run automated evaluation benchmark suite")

    # Prompts command
    subparsers.add_parser("prompts", help="Inspect prompt templates and schemas")

    # Web command
    p_web = subparsers.add_parser("web", help="Start interactive web interface and API")
    p_web.add_argument("--port", type=int, default=8000, help="Web server port")
    p_web.add_argument("--host", type=str, default="0.0.0.0", help="Web server host")

    args = parser.parse_args()

    if args.command == "query":
        cmd_query(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "prompts":
        cmd_prompts(args)
    elif args.command == "web":
        cmd_web(args)
    else:
        # Default behavior: run benchmark or show quick interactive prompt
        console.print(Panel.fit("[bold cyan]Welcome to OmniRAG[/bold cyan]\n"
                                "Multi-Hop Agentic RAG with Self-Reflection & Graph Augmentation\n\n"
                                "Commands:\n"
                                "  python main.py query \"<question>\"  : Run query with full trace\n"
                                "  python main.py benchmark          : Run evaluation suite\n"
                                "  python main.py prompts            : Inspect prompt catalog\n"
                                "  python main.py web                : Launch Web Dashboard"))


if __name__ == "__main__":
    main()
