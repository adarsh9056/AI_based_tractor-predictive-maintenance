const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

const API_STORAGE = "tractor_predictive_maintenance_api_base";

function isLoopbackHost(hostname) {
  return hostname === "localhost" || hostname === "127.0.0.1";
}

function isLoopbackUrl(raw) {
  try {
    const base = typeof window !== "undefined" ? window.location.origin : "http://127.0.0.1:8000";
    const u = new URL(raw, base);
    return isLoopbackHost(u.hostname);
  } catch {
    return false;
  }
}

function shouldIgnoreStoredApiBase(raw) {
  if (typeof window === "undefined" || !window.location?.protocol?.startsWith("http")) return false;
  return !isLoopbackHost(window.location.hostname) && isLoopbackUrl(raw);
}

function defaultApiOrigin() {
  if (typeof window !== "undefined" && window.__TRACTOR_API_BASE__) {
    const b = String(window.__TRACTOR_API_BASE__).trim().replace(/\/$/, "");
    if (b) return b;
  }
  if (typeof window !== "undefined" && window.location?.protocol?.startsWith("http")) {
    return window.location.origin.replace(/\/$/, "");
  }
  return "http://127.0.0.1:8000";
}

function apiBase() {
  const input = $("#api-base");
  const raw = (input?.value || "").trim();
  if (raw) return raw.replace(/\/$/, "");
  return defaultApiOrigin();
}

function saveApiBase() {
  try {
    localStorage.setItem(API_STORAGE, apiBase());
  } catch {
    /* ignore */
  }
}

function loadApiBase() {
  const input = $("#api-base");
  if (!input) return;
  try {
    const v = localStorage.getItem(API_STORAGE);
    if (v) {
      if (shouldIgnoreStoredApiBase(v)) {
        localStorage.removeItem(API_STORAGE);
      } else {
        input.value = v;
        return;
      }
    }
  } catch {
    /* ignore */
  }
  input.value = defaultApiOrigin();
}

function kToC(k) {
  return k - 273.15;
}

function fmtKWithC(k) {
  const kk = Number(k);
  return `${kk.toFixed(1)} K (${kToC(kk).toFixed(1)}°C)`;
}

const PRESETS = {
  healthy: { air: 298, proc: 308, rpm: 1600, torque: 32, wear: 80 },
  heat: { air: 301, proc: 308.5, rpm: 1250, torque: 30, wear: 50 },
  power: { air: 298, proc: 309, rpm: 1150, torque: 18, wear: 40 },
  wear: { air: 298, proc: 307, rpm: 1650, torque: 38, wear: 220 },
};

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  $("#air").value = p.air;
  $("#proc").value = p.proc;
  $("#rpm").value = p.rpm;
  $("#torque").value = p.torque;
  $("#wear").value = p.wear;
  $$("#air, #proc, #rpm, #torque, #wear").forEach((el) => el.dispatchEvent(new Event("input")));
}

function bindRange(id, outId, opts = {}) {
  const r = $(`#${id}`);
  const o = $(`#${outId}`);
  if (!r || !o) return;
  const { kelvin, rpm, torque, wear } = opts;
  const update = () => {
    const val = Number(r.value);
    if (kelvin) o.textContent = fmtKWithC(val);
    else if (rpm) o.textContent = `${Math.round(val)} rpm`;
    else if (torque) o.textContent = `${val.toFixed(1)} Nm`;
    else if (wear) o.textContent = `${Math.round(val)} min`;
    else o.textContent = String(val);
  };
  r.addEventListener("input", update);
  update();
}

function updateDerived() {
  const air = Number($("#air").value);
  const proc = Number($("#proc").value);
  const rpm = Number($("#rpm").value);
  const torque = Number($("#torque").value);
  const td = proc - air;
  const powerKw = (rpm * torque) / 9549;
  $("#temp-diff").textContent = td.toFixed(2);
  $("#temp-diff-hint").textContent = ` (${td.toFixed(2)} K ≈ same Δ in °C)`;
  $("#power-out").textContent = powerKw.toFixed(2);
}

function bandClass(band) {
  if (band === "Healthy") return "band band--healthy";
  if (band === "Critical") return "band band--critical";
  return "band band--warning";
}

async function postJson(path, body) {
  const res = await fetch(`${apiBase()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(text || res.statusText);
  }
  if (!res.ok) throw new Error(data.detail || data.message || res.statusText);
  return data;
}

async function getJson(path) {
  const res = await fetch(`${apiBase()}${path}`, { headers: { Accept: "application/json" } });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

function showResults(data) {
  $("#result-hero").hidden = false;
  $("#gauge-block").hidden = false;
  $("#out-pred").textContent = data.prediction || "—";
  const bandEl = $("#out-band");
  const wrap = $("#out-band-wrap");
  if (data.health_status) {
    wrap.hidden = false;
    bandEl.textContent = data.health_status;
    bandEl.className = bandClass(data.health_status);
  } else {
    wrap.hidden = true;
  }

  const factors = data.explanation_factors;
  const exBlock = $("#explain-block");
  const exList = $("#explain-list");
  if (factors?.length) {
    exBlock.hidden = false;
    exList.innerHTML = "";
    factors.forEach((t) => {
      const li = document.createElement("li");
      li.textContent = t;
      exList.appendChild(li);
    });
  } else {
    exBlock.hidden = true;
  }

  const fp = Number(data.failure_probability ?? 0);
  const pct = Math.min(100, Math.max(0, fp * 100));
  $("#out-fp").textContent = `${(fp * 100).toFixed(1)}%`;
  $("#gauge-fill").style.width = `${pct}%`;
  $("#gauge-bar").setAttribute("aria-valuenow", String(Math.round(pct)));

  const probs = data.class_probabilities;
  const list = $("#prob-list");
  list.innerHTML = "";
  if (probs && typeof probs === "object") {
    $("#prob-block").hidden = false;
    const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    const max = entries[0]?.[1] || 1;
    for (const [name, p] of entries) {
      const li = document.createElement("li");
      const pctBar = Math.round((p / max) * 100);
      li.innerHTML = `
        <span class="prob-list__name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
        <span class="prob-list__val">${(Number(p) * 100).toFixed(1)}%</span>
        <div class="prob-list__bar"><span style="width:${pctBar}%"></span></div>
      `;
      list.appendChild(li);
    }
  } else {
    $("#prob-block").hidden = true;
  }

  const cites = data.manual_citations;
  const citBlock = $("#citations-block");
  const citList = $("#citations-list");
  if (cites?.length) {
    citBlock.hidden = false;
    citList.innerHTML = "";
    cites.forEach((c, i) => {
      const li = document.createElement("li");
      li.innerHTML = `<strong>[${i + 1}]</strong> ${escapeHtml(c.diagnosis || c.condition || "")} — ${escapeHtml((c.action || "").slice(0, 120))}${(c.action || "").length > 120 ? "…" : ""} <span class="unit">(score ${c.match_score ?? "—"})</span>`;
      citList.appendChild(li);
    });
  } else {
    citBlock.hidden = true;
  }

  const advice = data.llm_diagnosis;
  if (advice) {
    $("#advice-block").hidden = false;
    $("#out-advice").textContent = advice;
  } else {
    $("#advice-block").hidden = true;
    $("#out-advice").textContent = "";
  }
}

async function runPredict() {
  const status = $("#status-line");
  status.textContent = "";
  status.classList.remove("is-error");
  const payload = {
    air_temp: Number($("#air").value),
    process_temp: Number($("#proc").value),
    rpm: Number($("#rpm").value),
    torque: Number($("#torque").value),
    tool_wear: Number($("#wear").value),
  };
  const path = $("#use-llm").checked ? "/predict_with_diagnosis" : "/predict";
  try {
    status.textContent = "Running…";
    const data = await postJson(path, payload);
    showResults(data);
    status.textContent = `Logged as #${data.log_id ?? "—"}`;
    await refreshLogs();
  } catch (e) {
    status.textContent = e.message || String(e);
    status.classList.add("is-error");
  }
}

async function refreshLogs() {
  const body = $("#logs-body");
  const meta = $("#logs-meta");
  try {
    const data = await getJson("/logs?limit=25&offset=0");
    const items = data.items || [];
    meta.textContent = data.total != null ? `Total: ${data.total}` : "";
    body.innerHTML = "";
    if (!items.length) {
      body.innerHTML = '<tr class="data-table__empty"><td colspan="7">No rows yet.</td></tr>';
      return;
    }
    for (const row of items) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.id}</td>
        <td>${escapeHtml(row.prediction)}</td>
        <td>${escapeHtml(row.health_status)}</td>
        <td>${(Number(row.failure_probability) * 100).toFixed(1)}%</td>
        <td>${fmtNum(row.rpm, 0)}</td>
        <td>${fmtNum(row.torque, 1)}</td>
        <td>${escapeHtml((row.created_at || "").slice(0, 19))}</td>
      `;
      body.appendChild(tr);
    }
  } catch {
    meta.textContent = "";
    body.innerHTML =
      '<tr class="data-table__empty"><td colspan="7">Could not load logs (is the API running?).</td></tr>';
  }
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmtNum(n, d) {
  const x = Number(n);
  if (Number.isNaN(x)) return "—";
  return d === 0 ? String(Math.round(x)) : x.toFixed(d);
}

function initHeaderScroll() {
  const header = $(".site-header");
  const onScroll = () => {
    if (window.scrollY > 24) header.classList.add("is-scrolled");
    else header.classList.remove("is-scrolled");
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
}

function init() {
  loadApiBase();
  bindRange("air", "air-out", { kelvin: true });
  bindRange("proc", "proc-out", { kelvin: true });
  bindRange("rpm", "rpm-out", { rpm: true });
  bindRange("torque", "torque-out", { torque: true });
  bindRange("wear", "wear-out", { wear: true });
  $$("#air, #proc, #rpm, #torque, #wear").forEach((el) => el.addEventListener("input", updateDerived));
  updateDerived();

  $$("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => applyPreset(btn.getAttribute("data-preset")));
  });

  $("#api-base")?.addEventListener("change", saveApiBase);
  $("#btn-predict")?.addEventListener("click", runPredict);
  $("#btn-refresh-logs")?.addEventListener("click", refreshLogs);

  initHeaderScroll();
  refreshLogs();
}

document.addEventListener("DOMContentLoaded", init);
