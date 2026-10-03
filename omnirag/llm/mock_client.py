import hashlib
import json
import math
import re
import time
from typing import List, Dict, Any, Optional
import numpy as np
from omnirag.llm.base import BaseLLMClient
from omnirag.core.schemas import LLMUsage
from omnirag.core.logger import logger


class MockLLMClient(BaseLLMClient):
    """High-fidelity deterministic offline LLM simulator.
    Ensures OmniRAG can be tested, benchmarked, and demonstrated out-of-the-box
    without requiring external paid API keys.
    """

    def __init__(self, model_name: str = "mock-gemini-simulator"):
        self.model_name = model_name
        logger.info(f"Initialized Mock LLM Engine ({self.model_name})")

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text.split()) * 4 // 3)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, LLMUsage]:
        start = time.perf_counter()

        # Synthesis generation simulation
        if "USER QUERY:" in user_prompt and "VERIFIED EVIDENCE CONTEXTS:" in user_prompt:
            output_text = self._synthesize_grounded_response(user_prompt)
        else:
            output_text = (
                f"### Analysis\nBased on the multi-hop reasoning pipeline, the query has been successfully "
                f"verified against the provided knowledge base."
            )

        prompt_tok = self._estimate_tokens(system_prompt + user_prompt)
        comp_tok = self._estimate_tokens(output_text)
        latency = (time.perf_counter() - start) * 1000 + 45.0  # simulate ~45ms

        usage = LLMUsage(
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            total_tokens=prompt_tok + comp_tok,
            estimated_cost_usd=0.0,
            latency_ms=latency,
            model=self.model_name,
        )
        return output_text, usage

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
    ) -> tuple[Dict[str, Any], LLMUsage]:
        start = time.perf_counter()
        lowered = user_prompt.lower()

        # 1. Query Decomposition simulation
        if "optimal multi-hop query execution plan" in lowered or "decompose" in lowered or "user question:" in lowered:
            data = self._simulate_query_decomposition(user_prompt)

        # 2. Document Grading simulation
        elif "evaluate the relevance of the following retrieved context chunk" in lowered:
            data = self._simulate_document_grading(user_prompt)

        # 3. Knowledge Graph Entity Extraction simulation
        elif "extract all key entities and factual relationships" in lowered:
            data = self._simulate_graph_extraction(user_prompt)

        # 4. Hallucination Auditing simulation
        elif "faithfulness and hallucination audit" in lowered or "audited_claims" in str(schema):
            data = self._simulate_hallucination_audit(user_prompt)

        else:
            data = {"status": "ok", "message": "Parsed successfully"}

        prompt_tok = self._estimate_tokens(system_prompt + user_prompt)
        comp_tok = self._estimate_tokens(json.dumps(data))
        latency = (time.perf_counter() - start) * 1000 + 35.0

        usage = LLMUsage(
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            total_tokens=prompt_tok + comp_tok,
            estimated_cost_usd=0.0,
            latency_ms=latency,
            model=self.model_name,
        )
        return data, usage

    def _simulate_query_decomposition(self, prompt: str) -> Dict[str, Any]:
        """Decomposes user query into logical sub-queries based on entity keywords."""
        query_match = re.search(r"USER QUESTION:\s*\n*(.*?)(?:\n\n|\Z)", prompt, re.DOTALL)
        query = query_match.group(1).strip() if query_match else prompt

        lower_q = query.lower()
        hops = []

        if "battery" in lower_q or "energy" in lower_q or "solid-state" in lower_q or "datacenter" in lower_q or "quantumscape" in lower_q:
            hops = [
                {
                    "hop_id": 1,
                    "sub_query": "QuantumScape solid-state battery energy density Wh/kg lithium-metal LLZO",
                    "target_entity": "QuantumScape Solid-State Battery",
                    "depends_on": [],
                },
                {
                    "hop_id": 2,
                    "sub_query": "conventional lithium-ion battery energy density Wh/kg graphite-silicon",
                    "target_entity": "Conventional Lithium-Ion",
                    "depends_on": [],
                },
                {
                    "hop_id": 3,
                    "sub_query": "hyperscale AI datacenter power demand long duration energy storage LDES Form Energy iron-air VRFB",
                    "target_entity": "Datacenter LDES Integration",
                    "depends_on": [1, 2],
                },
            ]
        elif ("quantum" in lower_q and "quantumscape" not in lower_q) or "coherence" in lower_q or "qubit" in lower_q:
            hops = [
                {
                    "hop_id": 1,
                    "sub_query": "neutral-atom quantum processor coherence time optical tweezers QuEra Atom Computing",
                    "target_entity": "Neutral-Atom Quantum Processor",
                    "depends_on": [],
                },
                {
                    "hop_id": 2,
                    "sub_query": "superconducting qubit coherence time transmon Google Willow IBM Heron gate speeds",
                    "target_entity": "Superconducting Qubit",
                    "depends_on": [],
                },
                {
                    "hop_id": 3,
                    "sub_query": "comparison coherence time gate speed bottlenecks neutral atom vs superconducting",
                    "target_entity": "Architecture Tradeoff",
                    "depends_on": [1, 2],
                },
            ]
        elif "packaging" in lower_q or "tsmc" in lower_q or "intel" in lower_q or "semiconductor" in lower_q:
            hops = [
                {
                    "hop_id": 1,
                    "sub_query": "TSMC N2 process node GAA nanosheet CoWoS SoIC advanced packaging",
                    "target_entity": "TSMC Advanced Packaging",
                    "depends_on": [],
                },
                {
                    "hop_id": 2,
                    "sub_query": "Intel Foundry Intel 18A RibbonFET PowerVia EMIB Foveros High-NA EUV",
                    "target_entity": "Intel Foundry 18A",
                    "depends_on": [],
                },
            ]
        else:
            # Fallback 2-hop decomposition
            hops = [
                {
                    "hop_id": 1,
                    "sub_query": f"Core entity definitions and primary facts for: {query[:60]}",
                    "target_entity": "Primary Subject",
                    "depends_on": [],
                },
                {
                    "hop_id": 2,
                    "sub_query": f"Comparative attributes, metrics, and relationships for: {query[:60]}",
                    "target_entity": "Secondary Relation",
                    "depends_on": [1],
                },
            ]

        return {
            "original_query": query,
            "reasoning": "Decomposed query into independent entity retrieval hops followed by a comparative relational synthesis.",
            "is_multihop": len(hops) > 1,
            "hops": hops,
        }

    def _simulate_document_grading(self, prompt: str) -> Dict[str, Any]:
        """Calculates relevance based on keyword match between sub-query and chunk."""
        sq_match = re.search(r"TARGET SUB-QUERY:\s*\n*(.*?)(?:\n\n|\Z)", prompt, re.DOTALL)
        chunk_match = re.search(r'"""\s*\n*(.*?)\s*"""', prompt, re.DOTALL)
        chunk_id_match = re.search(r"ID:\s*([a-zA-Z0-9_\-:]+)", prompt)

        sub_query = sq_match.group(1).strip() if sq_match else ""
        chunk_text = chunk_match.group(1).strip() if chunk_match else ""
        chunk_id = chunk_id_match.group(1).strip() if chunk_id_match else "chunk_0"

        sq_words = set(re.findall(r"\w+", sub_query.lower()))
        chunk_words = set(re.findall(r"\w+", chunk_text.lower()))

        overlap = len(sq_words.intersection(chunk_words))
        score = min(0.95, max(0.2, (overlap / max(1, len(sq_words))) * 1.5))

        if score >= 0.65:
            verdict = "RELEVANT"
        elif score >= 0.40:
            verdict = "PARTIALLY_RELEVANT"
        else:
            verdict = "IRRELEVANT"

        return {
            "sub_query": sub_query,
            "chunk_id": chunk_id,
            "relevance_score": round(score, 3),
            "verdict": verdict,
            "key_factual_points": [f"Matched keywords: {', '.join(list(sq_words.intersection(chunk_words))[:4])}"],
            "rationale": f"Chunk exhibits {overlap} keyword overlaps with target sub-query.",
        }

    def _simulate_graph_extraction(self, prompt: str) -> Dict[str, Any]:
        """Extracts key technology/corporate entities and relations from text chunk."""
        entities = []
        relationships = []

        patterns = [
            ("QuEra Computing", "ORGANIZATION", ["QuEra"]),
            ("Atom Computing", "ORGANIZATION", []),
            ("IBM Quantum", "ORGANIZATION", ["IBM"]),
            ("Google Quantum AI", "ORGANIZATION", ["Google"]),
            ("TSMC", "ORGANIZATION", ["Taiwan Semiconductor"]),
            ("Intel Foundry", "ORGANIZATION", ["Intel"]),
            ("QuantumScape", "ORGANIZATION", []),
            ("CATL", "ORGANIZATION", []),
            ("Neutral-Atom Quantum Processor", "TECHNOLOGY", ["Neutral-atom"]),
            ("Superconducting Qubit", "TECHNOLOGY", ["Transmon"]),
            ("N2 Process Node", "TECHNOLOGY", ["TSMC N2"]),
            ("Intel 18A", "TECHNOLOGY", ["18A"]),
            ("CoWoS", "TECHNOLOGY", ["Chip-on-Wafer-on-Substrate"]),
            ("EMIB", "TECHNOLOGY", []),
            ("Foveros", "TECHNOLOGY", []),
            ("PowerVia", "TECHNOLOGY", ["Backside Power"]),
            ("Solid-State Battery", "TECHNOLOGY", ["SSB"]),
            ("LLZO", "MATERIAL", ["Lithium Lanthanum Zirconium Oxide"]),
            ("Form Energy", "ORGANIZATION", []),
            ("Vanadium Redox Flow Batteries", "TECHNOLOGY", ["VRFB"]),
        ]

        text_lower = prompt.lower()
        found_entities = []

        for name, etype, aliases in patterns:
            if name.lower() in text_lower or any(a.lower() in text_lower for a in aliases):
                entities.append({"name": name, "type": etype, "aliases": aliases})
                found_entities.append(name)

        # Build relationships between co-occurring entities
        if "neutral-atom" in text_lower and "quera" in text_lower:
            relationships.append({
                "subject": "QuEra Computing",
                "predicate": "DEVELOPED",
                "object": "Neutral-Atom Quantum Processor",
                "evidence_snippet": "Leading industry pioneers QuEra Computing... demonstrated systems scaling beyond 1,180 neutral-atom qubits.",
            })
        if "tsmc" in text_lower and "cowos" in text_lower:
            relationships.append({
                "subject": "TSMC",
                "predicate": "MANUFACTURES",
                "object": "CoWoS",
                "evidence_snippet": "TSMC relies on CoWoS advanced packaging.",
            })
        if "intel" in text_lower and "powervia" in text_lower:
            relationships.append({
                "subject": "Intel Foundry",
                "predicate": "INTEGRATES",
                "object": "PowerVia",
                "evidence_snippet": "Intel 18A introduces RibbonFET and PowerVia backside power delivery.",
            })
        if "quantumscape" in text_lower and "llzo" in text_lower:
            relationships.append({
                "subject": "QuantumScape",
                "predicate": "UTILIZES_ELECTROLYTE",
                "object": "LLZO",
                "evidence_snippet": "Championed by QuantumScape, garnet-type LLZO ceramics provide chemical stability.",
            })

        return {"entities": entities[:6], "relationships": relationships[:4]}

    def _synthesize_grounded_response(self, prompt: str) -> str:
        """Constructs an authoritative cited synthesis based on the retrieved context."""
        contexts_match = re.search(r"VERIFIED EVIDENCE CONTEXTS:\s*\n*(.*?)(?:\n\nKNOWLEDGE|\Z)", prompt, re.DOTALL)
        query_match = re.search(r"USER QUERY:\s*\n*(.*?)(?:\n\nMULTI|\Z)", prompt, re.DOTALL)

        query = query_match.group(1).strip() if query_match else ""
        contexts = contexts_match.group(1).strip() if contexts_match else ""

        # Extract doc citation tags available in the context
        citation_tags = re.findall(r"\[Doc:([a-zA-Z0-9_\-]+):Chunk:(\d+)\]", contexts)
        tag1 = f"[Doc:{citation_tags[0][0]}:Chunk:{citation_tags[0][1]}]" if citation_tags else "[Doc:source:Chunk:1]"
        tag2 = f"[Doc:{citation_tags[1][0]}:Chunk:{citation_tags[1][1]}]" if len(citation_tags) > 1 else tag1

        lower_q = query.lower()

        if "battery" in lower_q or "energy" in lower_q or "datacenter" in lower_q or "quantumscape" in lower_q or "solid-state" in lower_q:
            return (
                f"### Next-Gen Energy Storage & AI Datacenter Power Infrastructure\n\n"
                f"**1. Gravimetric Energy Density Comparison:**\n"
                f"QuantumScape's anodelss lithium-metal solid-state battery cells have certified cell-level gravimetric energy densities of **415 Wh/kg** and volumetric densities exceeding 1,000 Wh/L with 12.2-minute fast-charging {tag1}. In comparison, conventional lithium-ion cells with graphite-silicon anodes (such as Tesla 4680 cells) top out between **260 to 285 Wh/kg** {tag1}.\n\n"
                f"**2. Addressing Hyperscale AI Datacenter Power Demands:**\n"
                f"Modern AI clusters with 100,000 liquid-cooled GPUs require **120 to 180 MW of continuous baseload electric power** {tag2}. To bridge renewable intermittency without fossil fuels, datacenters deploy Long-Duration Energy Storage (LDES):\n"
                f"- **Vanadium Redox Flow Batteries (VRFB):** Independent power and energy capacity with zero capacity fade over 25,000 cycles {tag2}.\n"
                f"- **Form Energy Iron-Air Batteries:** 100-hour continuous discharge cycles operating on reversible rusting with capital costs under $20/kWh {tag2}."
            )
        elif ("quantum" in lower_q and "quantumscape" not in lower_q) or "coherence" in lower_q or "qubit" in lower_q:
            return (
                f"### Executive Summary & Architecture Comparison\n\n"
                f"**1. Coherence Time Benchmarks:**\n"
                f"Neutral-atom quantum architectures demonstrate exceptionally long coherence times, with single-qubit T2* values consistently exceeding 12.4 seconds, and Atom Computing reporting laboratory nuclear-spin coherence times reaching 40 seconds {tag1}. In stark contrast, state-of-the-art superconducting transmon qubits (such as IBM Heron and Google Willow) exhibit coherence times bounded between 150 to 350 microseconds {tag2}. This gives neutral-atom systems a superiority of nearly **five orders of magnitude** in quantum coherence duration.\n\n"
                f"**2. Clock Speed and Latency Bottlenecks:**\n"
                f"While superconducting processors operate with rapid gate times (single-qubit gates <20 ns, two-qubit entangling gates 40–60 ns) {tag2}, neutral-atom systems are physically bottlenecked by optical trap shuttle rearrangement and laser steering routines. Specifically, atom rearrangement requires between **50 to 200 microseconds per cycle** {tag1}, limiting the maximum clock frequency achievable in current neutral-atom hardware.\n\n"
                f"**3. Thermal Integration:**\n"
                f"Superconducting qubits mandate multi-stage dilution cryostats running at 15 millikelvin {tag2}, whereas neutral-atom vacuum chambers operate largely at room temperature, using cryogenic cooling only for single-photon detectors {tag1}."
            )
        elif "packaging" in lower_q or "tsmc" in lower_q or "intel" in lower_q or "semiconductor" in lower_q:
            return (
                f"### Semiconductor Foundry Roadmap & Advanced Packaging\n\n"
                f"**1. Flagship Sub-2nm Nodes:**\n"
                f"- **TSMC N2:** TSMC is ramping its 2nm-class (N2) node utilizing Gate-All-Around (GAA) nanosheets with standard 0.33 NA Extreme Ultraviolet (EUV) lithography scanners from ASML, delivering a 15% speed gain or 30% power reduction over N3E {tag1}.\n"
                f"- **Intel 18A:** Intel has deployed its 1.8nm node (Intel 18A) featuring proprietary RibbonFET nanosheets and **PowerVia backside power delivery**, which reduces IR drop voltage loss by 30% and pushes standard cell utilization above 90% {tag2}. Intel also pioneered the industry's first commercial High-NA (0.55 NA) EUV scanner (Twinscan EXE:5000) at its Oregon foundry {tag2}.\n\n"
                f"**2. Advanced Packaging to Overcome Die Yield Limits:**\n"
                f"Because monolithic dies exceeding 800 mm² suffer low yields, both foundries deploy heterogeneous multi-die packaging:\n"
                f"- **TSMC:** Uses **CoWoS** (Chip-on-Wafer-on-Substrate) and **SoIC** 3D hybrid bonding (>10,000 interconnects/mm²), expanding CoWoS capacity to 75,000 wafers/month to feed AI accelerators like Nvidia Blackwell Ultra {tag1}.\n"
                f"- **Intel:** Deploys **EMIB** (Embedded Multi-die Interconnect Bridge) 2.5D bridges and **Foveros** 3D direct die stacking with sub-9µm bump pitches {tag2}."
            )
        else:
            return (
                f"### Multi-Hop Synthesized Analysis\n\n"
                f"Based on cross-document evidence synthesis across verified sources {tag1} {tag2}:\n"
                f"The target entities exhibit strong relational alignment. All facts presented have been verified against the indexed knowledge base and cross-referenced with extracted entity relationships."
            )

    def _simulate_hallucination_audit(self, prompt: str) -> Dict[str, Any]:
        """Performs simulated faithfulness auditing with claim decomposition."""
        ans_match = re.search(r'GENERATED ANSWER:\s*\n*"""(.*?)"""', prompt, re.DOTALL)
        answer = ans_match.group(1).strip() if ans_match else ""

        # Extract sentences as factual claims
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 25 and not s.strip().startswith("#")]
        audited = []

        for s in sentences[:6]:
            citation_match = re.search(r"\[Doc:[^\]]+\]", s)
            citation = citation_match.group(0) if citation_match else None
            audited.append({
                "claim": s[:140],
                "status": "ENTAILED",
                "citation": citation,
                "source_evidence_snippet": "Directly grounded in retrieved knowledge base chunk.",
                "audit_note": "Factual metrics and entity assertions verified.",
            })

        total = len(audited)
        entailed = total  # Default high-fidelity grounded result

        return {
            "total_claims": total,
            "entailed_claims_count": entailed,
            "unverified_claims_count": 0,
            "contradicted_claims_count": 0,
            "faithfulness_score": 1.0,
            "hallucination_risk": "LOW",
            "audited_claims": audited,
            "overall_summary": "100% of factual assertions were fully grounded and entailed by the verified context chunks.",
        }

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates deterministic 768-dimensional normalized dense vectors.
        Uses bag-of-words and character n-gram hashing to ensure semantic proximity:
        texts with similar words get high cosine similarity (>0.75)!
        """
        dim = 768
        embeddings = []

        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            words = re.findall(r"\w+", text.lower())

            for i, word in enumerate(words):
                # Hash word to feature indices
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
                idx = h % dim
                weight = 1.0 / (1.0 + math.log(1 + i))
                vec[idx] += weight

                # Character 3-grams
                for j in range(len(word) - 2):
                    tri = word[j : j + 3]
                    th = int(hashlib.sha256(tri.encode("utf-8")).hexdigest(), 16)
                    t_idx = th % dim
                    vec[t_idx] += 0.5 * weight

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())

        return embeddings
