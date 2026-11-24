"use strict";

// Estado de la consulta del ranking (se refleja en la URL de la API).
const state = { offset: 0, limit: 50, sort: "priority", order: "desc", search: "", only_kev: false };

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 2) => (x === null || x === undefined ? "—" : Number(x).toFixed(d));

function prioClass(v) {
  if (v === null) return "low";
  if (v >= 7) return "crit";
  if (v >= 5) return "high";
  if (v >= 3) return "med";
  return "low";
}

// --- Ranking ---------------------------------------------------------------
async function loadVulns() {
  const p = new URLSearchParams({
    offset: state.offset, limit: state.limit, sort: state.sort, order: state.order,
  });
  if (state.search) p.set("search", state.search);
  if (state.only_kev) p.set("only_kev", "true");

  const data = await fetch(`/api/vulnerabilities?${p}`).then((r) => r.json());
  const body = $("vuln-body");
  body.innerHTML = "";
  for (const [i, row] of data.rows.entries()) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${data.offset + i + 1}</td>
      <td class="cve">${row.cve_id}</td>
      <td><span class="prio ${prioClass(row.priority)}">${fmt(row.priority, 1)}</span></td>
      <td>${fmt(row.severity, 1)}</td>
      <td>${fmt(row.threat, 1)}</td>
      <td>${fmt(row.cvss, 1)}</td>
      <td>${fmt(row.epss, 3)}</td>
      <td>${row.in_kev ? '<span class="badge kev">KEV</span>' : ""}</td>`;
    tr.addEventListener("click", () => openDetail(row.cve_id));
    body.appendChild(tr);
  }

  const from = data.total === 0 ? 0 : state.offset + 1;
  const to = Math.min(state.offset + state.limit, data.total);
  $("page-info").textContent = `${from.toLocaleString()}–${to.toLocaleString()} de ${data.total.toLocaleString()}`;
  $("prev").disabled = state.offset === 0;
  $("next").disabled = state.offset + state.limit >= data.total;
}

// --- Detalle de un CVE -----------------------------------------------------
async function openDetail(cve) {
  const d = await fetch(`/api/vulnerability/${cve}`).then((r) => r.json());
  $("detail-title").textContent = d.cve_id;
  $("detail-summary").textContent = d.explanation.summary;

  const e = d.explanation;
  $("chain").innerHTML =
    stageHtml("1 · Severidad", e.severity, ["exploitability_subscore", "impact_subscore"],
              { exploitability_subscore: "explotabilidad", impact_subscore: "impacto" }) +
    stageHtml("2 · Amenaza", e.threat, ["epss_percentile", "epss_delta"],
              { epss_percentile: "percentil EPSS", epss_delta: "variación EPSS" }) +
    stageHtml("3 · Prioridad", e.priority_raw, ["severity", "threat", "log_cpe"],
              { severity: "severidad", threat: "amenaza", log_cpe: "superficie" }) +
    (e.kev_boost ? `<div class="stage"><div class="stage-head"><span>Ajuste KEV</span>
        <span class="val">+${e.kev_boost}</span></div>
        <div class="rule">Confirmada como explotada en CISA KEV.</div></div>` : "") +
    `<div class="stage"><div class="stage-head"><span><b>Prioridad final</b></span>
        <span class="val">${fmt(e.priority, 1)}</span></div></div>`;

  const cmp = $("compare-body");
  cmp.innerHTML = "";
  for (const m of d.comparison) {
    const tr = document.createElement("tr");
    if (m.method === "FIS") tr.className = "fis";
    const score = m.method === "EPSS" ? fmt(m.score, 3) : fmt(m.score, 2);
    tr.innerHTML = `<td>${m.method}</td><td>${score}</td><td>#${m.rank.toLocaleString()}</td>`;
    cmp.appendChild(tr);
  }

  $("ranking-view").hidden = true;
  $("detail-view").hidden = false;
  window.scrollTo({ top: 0 });
}

function stageHtml(title, sub, inputs, labels) {
  const mfs = inputs.map((v) => {
    const degs = sub.memberships[v] || {};
    const chips = Object.entries(degs)
      .map(([lbl, mu]) => `<span class="mf ${mu > 0 ? "on" : ""}">${lbl} ${mu.toFixed(2)}</span>`)
      .join("");
    return `<div><span class="hint">${labels[v]}</span><div class="mfbar">${chips}</div></div>`;
  }).join("");
  const rule = sub.fired_rules && sub.fired_rules.length
    ? `<div class="rule">Regla dominante: ${sub.fired_rules[0].antecedents.join(" ∧ ")}
       → ${sub.fired_rules[0].output} (fuerza ${sub.fired_rules[0].strength.toFixed(2)})</div>`
    : "";
  return `<div class="stage">
      <div class="stage-head"><span>${title}: ${sub.dominant_label_es}</span>
        <span class="val">${fmt(sub.value, 1)}</span></div>
      ${mfs}${rule}</div>`;
}

// --- Gráficas y estadísticas ----------------------------------------------
const LAYOUT = {
  paper_bgcolor: "transparent", plot_bgcolor: "transparent",
  font: { color: "#93a4b5", size: 11 }, margin: { t: 40, r: 20, b: 45, l: 50 },
};

async function loadStats() {
  const s = await fetch("/api/stats").then((r) => r.json());
  $("stat-chips").innerHTML = `
    <div class="chip"><div class="num">${s.total.toLocaleString()}</div><div class="lbl">Vulnerabilidades</div></div>
    <div class="chip"><div class="num">${s.n_kev.toLocaleString()}</div><div class="lbl">Explotadas (KEV)</div></div>`;

  // Comparativa de métodos (R@500 y NDCG@500).
  if (s.methods_table.length) {
    const m = s.methods_table;
    Plotly.newPlot("chart-methods", [
      { x: m.map((r) => r.method), y: m.map((r) => r["R@500"]), name: "Recall@500", type: "bar", marker: { color: "#4aa3df" } },
      { x: m.map((r) => r.method), y: m.map((r) => r["NDCG@500"]), name: "NDCG@500", type: "bar", marker: { color: "#5aa777" } },
    ], { ...LAYOUT, title: "Comparativa de métodos (verdad KEV)", barmode: "group", legend: { orientation: "h", y: -0.2 } }, { responsive: true, displayModeBar: false });
  }

  // Distribución de la prioridad.
  const edges = s.distribution.edges, counts = s.distribution.counts;
  const centers = counts.map((_, i) => (edges[i] + edges[i + 1]) / 2);
  Plotly.newPlot("chart-distribution", [
    { x: centers, y: counts, type: "bar", marker: { color: "#e08e3c" } },
  ], { ...LAYOUT, title: "Distribución de la prioridad", xaxis: { title: "Prioridad" }, yaxis: { title: "CVEs" } }, { responsive: true, displayModeBar: false });

  // Robustez ante ruido en EPSS (FIS vs RF).
  const noise = s.robustness.filter((r) => r.escenario === "C_epss_noise");
  if (noise.length) {
    const fis = noise.filter((r) => r.metodo === "FIS").sort((a, b) => a.nivel - b.nivel);
    const rf = noise.filter((r) => r.metodo === "RF").sort((a, b) => a.nivel - b.nivel);
    Plotly.newPlot("chart-robustness", [
      { x: fis.map((r) => r.nivel), y: fis.map((r) => r["R@500"]), name: "FIS", mode: "lines+markers", line: { color: "#4aa3df" } },
      { x: rf.map((r) => r.nivel), y: rf.map((r) => r["R@500"]), name: "Random Forest", mode: "lines+markers", line: { color: "#e5534b" } },
    ], { ...LAYOUT, title: "Robustez ante ruido en EPSS", xaxis: { title: "σ del ruido" }, yaxis: { title: "Recall@500" }, legend: { orientation: "h", y: -0.25 } }, { responsive: true, displayModeBar: false });
  }
}

// --- Eventos ---------------------------------------------------------------
let searchTimer;
$("search").addEventListener("input", (ev) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { state.search = ev.target.value.trim(); state.offset = 0; loadVulns(); }, 300);
});
$("sort").addEventListener("change", (ev) => { state.sort = ev.target.value; state.offset = 0; loadVulns(); });
$("only-kev").addEventListener("change", (ev) => { state.only_kev = ev.target.checked; state.offset = 0; loadVulns(); });
$("prev").addEventListener("click", () => { state.offset = Math.max(0, state.offset - state.limit); loadVulns(); window.scrollTo({ top: 0, behavior: "smooth" }); });
$("next").addEventListener("click", () => { state.offset += state.limit; loadVulns(); window.scrollTo({ top: 0, behavior: "smooth" }); });
$("detail-back").addEventListener("click", () => { $("detail-view").hidden = true; $("ranking-view").hidden = false; window.scrollTo({ top: 0 }); });

loadStats();
loadVulns();
