document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("rag-query-form");
  const queryInput = document.getElementById("query-input");
  const submitBtn = document.getElementById("submit-btn");
  const stepperSection = document.getElementById("pipeline-stepper");
  const resultsSection = document.getElementById("results-section");

  // Tabs
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      document.getElementById(targetId)?.classList.add("active");
    });
  });

  // Preset Chips
  document.querySelectorAll(".preset-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-query");
      queryInput.value = q;
      form.dispatchEvent(new Event("submit"));
    });
  });

  // Form Submit
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    submitBtn.disabled = true;
    submitBtn.querySelector(".btn-text").textContent = "Reasoning...";
    stepperSection.classList.remove("hidden");
    resultsSection.classList.add("hidden");

    // Animate stepper steps
    animateStepper();

    try {
      const resp = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!resp.ok) {
        throw new Error(`Server returned ${resp.status}`);
      }

      const data = await resp.json();
      renderResults(data);
    } catch (err) {
      alert("Error executing query: " + err.message);
    } finally {
      submitBtn.disabled = false;
      submitBtn.querySelector(".btn-text").textContent = "Execute Multi-Hop Reasoning";
    }
  });

  function animateStepper() {
    const nodes = [1, 2, 3, 4, 5];
    nodes.forEach((n) => {
      const el = document.getElementById(`step-${n}`);
      const line = document.getElementById(`line-${n - 1}`);
      if (el) el.className = "step-node";
      if (line) line.className = "step-line";
    });

    let current = 1;
    const interval = setInterval(() => {
      const node = document.getElementById(`step-${current}`);
      const line = document.getElementById(`line-${current - 1}`);
      if (node) node.classList.add("completed");
      if (line) line.classList.add("completed");

      current++;
      const nextNode = document.getElementById(`step-${current}`);
      if (nextNode) nextNode.classList.add("active");

      if (current > 5) {
        clearInterval(interval);
      }
    }, 180);
  }

  function renderResults(data) {
    resultsSection.classList.remove("hidden");

    // 1. Telemetry Bar
    document.getElementById("telemetry-latency").textContent = `${data.execution_time_seconds}s`;
    const faithScore = Math.round(data.audit_report.faithfulness_score * 100);
    document.getElementById("telemetry-faithfulness").textContent = `${faithScore}%`;
    const riskEl = document.getElementById("telemetry-risk");
    riskEl.textContent = data.audit_report.hallucination_risk;
    riskEl.className = data.audit_report.hallucination_risk === "LOW" ? "telemetry-val badge-low" : "telemetry-val badge-amber";
    document.getElementById("telemetry-tokens").textContent = data.usage.total_tokens.toLocaleString();
    document.getElementById("telemetry-model").textContent = data.usage.model;

    // 2. Synthesized Answer with interactive citations
    const answerContainer = document.getElementById("synthesized-answer-body");
    answerContainer.innerHTML = formatMarkdownWithCitations(data.synthesized_answer);

    // Attach click listeners to citation tags to switch to evidence tab
    answerContainer.querySelectorAll(".citation-tag").forEach((tag) => {
      tag.addEventListener("click", () => {
        document.querySelector('[data-tab="tab-evidence"]').click();
      });
    });

    // 3. Query Plan (DAG)
    document.getElementById("plan-hops-count").textContent = `${data.query_plan.hops.length} Hops`;
    document.getElementById("plan-reasoning").textContent = `Decomposition Rationale: ${data.query_plan.reasoning}`;
    const dagContainer = document.getElementById("dag-container");
    dagContainer.innerHTML = "";

    data.query_plan.hops.forEach((h) => {
      const node = document.createElement("div");
      node.className = "dag-node";
      const deps = h.depends_on && h.depends_on.length > 0 ? `Depends on: Hop ${h.depends_on.join(", ")}` : "Root Independent Hop";
      node.innerHTML = `
        <div class="dag-node-left">
          <span class="dag-hop-badge">Hop ${h.hop_id}</span>
          <div>
            <div class="dag-subquery">${escapeHtml(h.sub_query)}</div>
            <div class="dag-target">Target Entity: <strong>${escapeHtml(h.target_entity || "Direct Fact")}</strong> &bull; ${deps}</div>
          </div>
        </div>
        <span class="card-badge">${h.status}</span>
      `;
      dagContainer.appendChild(node);
    });

    // 4. Evidence Matrix Table
    const tbody = document.getElementById("evidence-tbody");
    tbody.innerHTML = "";

    data.retrieved_chunks.forEach((c) => {
      const tr = document.createElement("tr");
      const isRel = (c.relevance_verdict || "RELEVANT") === "RELEVANT";
      tr.innerHTML = `
        <td><strong style="color: var(--accent-cyan); font-family: var(--font-mono);">${c.chunk_id}</strong></td>
        <td>${c.doc_id}</td>
        <td>${c.bm25_score.toFixed(2)}</td>
        <td>${c.dense_score.toFixed(3)}</td>
        <td><strong>${c.rrf_score.toFixed(5)}</strong></td>
        <td><span class="verdict-badge ${isRel ? "verdict-relevant" : "verdict-irrelevant"}">${c.relevance_verdict || "RELEVANT"}</span></td>
        <td style="max-width: 320px; font-size: 12px; color: var(--text-secondary);">${escapeHtml(c.text.slice(0, 160))}...</td>
      `;
      tbody.appendChild(tr);
    });

    // 5. Knowledge Graph Visualizer (SVG)
    renderKnowledgeGraphSVG(data.knowledge_graph);

    // 6. Hallucination Audit Tab
    const auditSummary = document.getElementById("audit-summary-text");
    auditSummary.textContent = data.audit_report.overall_summary || `Faithfulness Score: ${faithScore}%. Factual statements entailed.`;
    document.getElementById("audit-score-badge").textContent = `Faithfulness: ${faithScore}% (${data.audit_report.hallucination_risk} Risk)`;

    const auditTbody = document.getElementById("audit-tbody");
    auditTbody.innerHTML = "";

    data.audit_report.audited_claims.forEach((claim) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escapeHtml(claim.claim)}</td>
        <td><span class="verdict-badge verdict-relevant">${claim.status}</span></td>
        <td><span class="citation-tag">${claim.citation || "Direct"}</span></td>
        <td style="font-size: 12px; color: var(--text-secondary);">${escapeHtml(claim.source_evidence_snippet || "Verified from knowledge corpus.")}</td>
      `;
      auditTbody.appendChild(tr);
    });

    // Auto scroll down smoothly to results
    resultsSection.scrollIntoView({ behavior: "smooth" });
  }

  function formatMarkdownWithCitations(rawText) {
    let html = escapeHtml(rawText);

    // Headers
    html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
    html = html.replace(/^## (.*$)/gim, "<h2>$1</h2>");
    html = html.replace(/^# (.*$)/gim, "<h1>$1</h1>");

    // Bold & italic
    html = html.replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>");
    html = html.replace(/\*(.*?)\*/gim, "<em>$1</em>");

    // Citation tags: [Doc:X:Chunk:Y]
    html = html.replace(/\[Doc:([^\]]+)\]/g, '<span class="citation-tag" title="Click to view verified source context">[Doc:$1]</span>');

    // Paragraphs
    html = html.split("\n\n").map((para) => {
      if (para.startsWith("<h") || para.startsWith("<ul") || para.startsWith("<ol")) return para;
      return `<p>${para.replace(/\n/g, "<br>")}</p>`;
    }).join("");

    return html;
  }

  function renderKnowledgeGraphSVG(kg) {
    const svg = document.getElementById("kg-canvas");
    svg.innerHTML = "";

    if (!kg || !kg.nodes || kg.nodes.length === 0) {
      svg.innerHTML = '<text x="50%" y="50%" fill="#64748b" text-anchor="middle">No knowledge graph entities found.</text>';
      return;
    }

    const width = svg.clientWidth || 800;
    const height = 450;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.38;

    // Distribute nodes in a circle
    const nodeCoords = {};
    const total = kg.nodes.length;

    kg.nodes.forEach((node, idx) => {
      const angle = (idx / total) * 2 * Math.PI - Math.PI / 2;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      nodeCoords[node.name] = { x, y, node };
    });

    // Draw Edges
    kg.edges.forEach((edge) => {
      const p1 = nodeCoords[edge.source];
      const p2 = nodeCoords[edge.target];
      if (p1 && p2) {
        // Line
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", p1.x);
        line.setAttribute("y1", p1.y);
        line.setAttribute("x2", p2.x);
        line.setAttribute("y2", p2.y);
        line.setAttribute("stroke", "rgba(0, 240, 255, 0.35)");
        line.setAttribute("stroke-width", "2");
        line.setAttribute("stroke-dasharray", "4");
        svg.appendChild(line);

        // Edge Predicate Label
        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2;
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", midX);
        text.setAttribute("y", midY - 4);
        text.setAttribute("fill", "#00f0ff");
        text.setAttribute("font-size", "10");
        text.setAttribute("font-family", "JetBrains Mono");
        text.setAttribute("text-anchor", "middle");
        text.textContent = edge.predicate;
        svg.appendChild(text);
      }
    });

    // Draw Nodes
    Object.values(nodeCoords).forEach(({ x, y, node }) => {
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");

      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", x);
      circle.setAttribute("cy", y);
      circle.setAttribute("r", 20);
      circle.setAttribute("fill", "#1e293b");
      circle.setAttribute("stroke", "#8b5cf6");
      circle.setAttribute("stroke-width", "2");

      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", x);
      text.setAttribute("y", y + 34);
      text.setAttribute("fill", "#f8fafc");
      text.setAttribute("font-size", "11");
      text.setAttribute("font-family", "Plus Jakarta Sans");
      text.setAttribute("font-weight", "600");
      text.setAttribute("text-anchor", "middle");
      text.textContent = node.name.length > 20 ? node.name.slice(0, 18) + ".." : node.name;

      g.appendChild(circle);
      g.appendChild(text);
      svg.appendChild(g);
    });
  }

  // Prompt Catalog Modal
  const modal = document.getElementById("prompt-modal");
  const modalBtn = document.getElementById("view-prompts-btn");
  const closeBtn = document.getElementById("close-modal-btn");
  const modalBody = document.getElementById("prompt-modal-body");

  modalBtn.addEventListener("click", async () => {
    modal.classList.remove("hidden");
    try {
      const res = await fetch("/api/prompts");
      const prompts = await res.json();
      let html = '<div style="display:flex; flex-direction:column; gap:16px;">';
      for (const [key, p] of Object.entries(prompts)) {
        html += `
          <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:16px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
              <strong style="color:var(--accent-cyan); font-family:var(--font-mono);">${key}</strong>
              <span class="card-badge">v${p.version} &bull; ${p.output_format}</span>
            </div>
            <p style="font-size:13px; color:var(--text-secondary);">${escapeHtml(p.description)}</p>
          </div>
        `;
      }
      html += "</div>";
      modalBody.innerHTML = html;
    } catch (e) {
      modalBody.innerHTML = `<p style="color:red">Failed to load prompts: ${e.message}</p>`;
    }
  });

  closeBtn.addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.add("hidden");
  });

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
