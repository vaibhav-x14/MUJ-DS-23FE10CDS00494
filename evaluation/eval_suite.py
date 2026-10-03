import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.table import Table
from rich.panel import Panel
from omnirag.pipeline import OmniRAGEngine
from omnirag.core.logger import console, logger
from config.settings import DATA_DIR, settings



class OmniRAGEvalSuite:
    """Rigorous evaluation harness benchmark suite for multi-hop RAG."""

    def __init__(self, benchmark_file: Optional[Path] = None):
        self.benchmark_path = benchmark_file or (DATA_DIR / "benchmark_qa.json")
        self.engine = OmniRAGEngine()

    def load_benchmarks(self) -> List[Dict[str, Any]]:
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_retrieval_recall(self, retrieved_doc_ids: List[str], gold_doc_ids: List[str]) -> float:
        """Calculates Recall @ K against gold document IDs."""
        if not gold_doc_ids:
            return 1.0
        hits = sum(1 for gid in gold_doc_ids if any(gid in rid for rid in retrieved_doc_ids))
        return hits / len(gold_doc_ids)

    def evaluate_context_precision(self, retrieved_chunks: list, gold_doc_ids: List[str]) -> float:
        """Calculates Precision of retrieved chunks."""
        if not retrieved_chunks:
            return 0.0
        hits = sum(1 for c in retrieved_chunks if any(gid in c.doc_id for gid in gold_doc_ids))
        return hits / len(retrieved_chunks)

    def evaluate_token_f1(self, prediction: str, reference: str) -> float:
        """Computes word-level F1 score between prediction and reference."""
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        common = pred_tokens.intersection(ref_tokens)
        if not common:
            return 0.0
        prec = len(common) / len(pred_tokens)
        rec = len(common) / len(ref_tokens)
        return (2 * prec * rec) / (prec + rec)

    def run_benchmark(self) -> Dict[str, Any]:
        """Runs the complete benchmark suite and prints rich report."""
        benchmarks = self.load_benchmarks()
        self.engine.index()

        console.print(Panel.fit("[bold cyan]OmniRAG MAANG-Tier Benchmark Evaluation Suite[/bold cyan]\n"
                                f"Evaluating {len(benchmarks)} Multi-Hop Cross-Document Benchmark Scenarios..."))

        results = []
        total_recall = 0.0
        total_precision = 0.0
        total_faithfulness = 0.0
        total_f1 = 0.0
        total_latency = 0.0

        table = Table(title="OmniRAG Benchmark Metrics (Per Query)")
        table.add_column("Query ID", style="cyan", no_wrap=True)
        table.add_column("Hops (Plan / Exp)", style="magenta")
        table.add_column("Context Recall", style="green")
        table.add_column("Context Precision", style="green")
        table.add_column("Faithfulness", style="yellow")
        table.add_column("Answer F1", style="blue")
        table.add_column("Latency (s)", style="white")

        for item in benchmarks:
            qid = item["id"]
            query = item["query"]
            gold_docs = item["gold_doc_ids"]
            exp_hops = item["expected_hops"]
            reference = item["ground_truth_reference"]

            t0 = time.perf_counter()
            rag_res = self.engine.query(query)
            latency = time.perf_counter() - t0

            retrieved_doc_ids = [c.doc_id for c in rag_res.retrieved_chunks]
            recall = self.evaluate_retrieval_recall(retrieved_doc_ids, gold_docs)
            precision = self.evaluate_context_precision(rag_res.retrieved_chunks, gold_docs)
            faithfulness = rag_res.audit_report.faithfulness_score
            f1 = self.evaluate_token_f1(rag_res.synthesized_answer, reference)

            actual_hops = len(rag_res.query_plan.hops)

            total_recall += recall
            total_precision += precision
            total_faithfulness += faithfulness
            total_f1 += f1
            total_latency += latency

            results.append({
                "id": qid,
                "recall": recall,
                "precision": precision,
                "faithfulness": faithfulness,
                "answer_f1": f1,
                "latency_s": round(latency, 3),
            })

            table.add_row(
                qid,
                f"{actual_hops} / {exp_hops}",
                f"{recall*100:.1f}%",
                f"{precision*100:.1f}%",
                f"{faithfulness*100:.1f}%",
                f"{f1*100:.1f}%",
                f"{latency:.3f}",
            )

        console.print(table)

        n = len(benchmarks)
        summary = {
            "mean_context_recall": round(total_recall / n, 4),
            "mean_context_precision": round(total_precision / n, 4),
            "mean_faithfulness_score": round(total_faithfulness / n, 4),
            "mean_answer_f1": round(total_f1 / n, 4),
            "mean_latency_seconds": round(total_latency / n, 4),
        }

        summary_table = Table(title="System-Level Aggregate Performance Summary")
        summary_table.add_column("Metric", style="bold white")
        summary_table.add_column("Benchmark Score", style="bold green")
        summary_table.add_column("Target MAANG SLA", style="cyan")

        summary_table.add_row("Mean Context Recall @ K", f"{summary['mean_context_recall']*100:.1f}%", ">= 90.0%")
        summary_table.add_row("Mean Context Precision @ K", f"{summary['mean_context_precision']*100:.1f}%", ">= 75.0%")
        summary_table.add_row("Answer Faithfulness (Self-Audit)", f"{summary['mean_faithfulness_score']*100:.1f}%", ">= 95.0%")
        summary_table.add_row("Answer Grounding Token F1", f"{summary['mean_answer_f1']*100:.1f}%", ">= 50.0%")
        summary_table.add_row("Mean End-to-End Latency", f"{summary['mean_latency_seconds']:.3f} s", "< 2.0 s")

        console.print(summary_table)
        return summary


if __name__ == "__main__":
    suite = OmniRAGEvalSuite()
    suite.run_benchmark()
