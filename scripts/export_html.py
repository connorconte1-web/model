"""Export the trained model into a single self-contained interactive HTML file.

Strategy:
- Re-fit FT and HT Dixon-Coles on all completed matches; export attack/defense
  vectors, home_adv, and rho.  The browser recomputes the score matrix from these
  numbers, so any (home, away) selection runs market math live.
- Precompute corner expectations for every (home, away) pair from the LightGBM
  model (needs rolling features that JS can't easily compute) and embed as a
  lookup table.
- Embed every upcoming fixture together with its closing odds and the model's
  value-bet flags so the page can render a ready-made value-bets table.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from peru_pred.corners import fit_corners, predict_corners
from peru_pred.data import load_default, split_completed
from peru_pred.dixon_coles import fit
from peru_pred.markets import predict_markets
from peru_pred.value import find_value_bets, market_candidates_from_prediction


CORNER_LINES = (7.5, 8.5, 9.5, 10.5, 11.5)
EDGE_THRESHOLD = 0.03
KELLY_CAP = 0.25


def _dc_payload(params) -> dict:
    return {
        "teams": params.teams,
        "attack": params.attack,
        "defense": params.defense,
        "home_adv": params.home_adv,
        "rho": params.rho,
        "league_mean_goals": params.league_mean_goals,
    }


def build_payload() -> dict:
    completed, upcoming = split_completed(load_default())

    print(f"Fitting FT/HT Dixon-Coles on {len(completed)} matches ...")
    ft = fit(completed, xi=0.002, target="ft")
    ht = fit(completed, xi=0.002, target="ht")

    print("Fitting corners model ...")
    cmodel = fit_corners(completed)

    # Build a virtual matchup grid for every (home, away) using the latest known stats.
    teams = sorted(set(completed["home"]).union(completed["away"]))
    pairs = [(h, a) for h in teams for a in teams if h != a]
    virtual = pd.DataFrame(
        {
            "date": [completed["date"].max() + pd.Timedelta(days=1)] * len(pairs),
            "home": [h for h, _ in pairs],
            "away": [a for _, a in pairs],
            "home_corners": np.nan,
            "away_corners": np.nan,
            "home_shots": np.nan,
            "away_shots": np.nan,
            "home_shots_on_target": np.nan,
            "away_shots_on_target": np.nan,
            "pm_home_xg": np.nan,
            "pm_away_xg": np.nan,
        }
    )
    print(f"Predicting corners for {len(pairs)} virtual pairings ...")
    cpred = predict_corners(cmodel, completed, virtual, lines=CORNER_LINES, n_sim=20_000)
    cpred = cpred.reset_index(drop=True)
    corner_table: dict[str, dict] = {}
    for (h, a), idx in zip(pairs, cpred.index):
        row = cpred.loc[idx]
        d = {
            "lam_h": float(row["lam_corners_home"]),
            "lam_a": float(row["lam_corners_away"]),
            "lam_total": float(row["lam_corners_total"]),
        }
        for L in CORNER_LINES:
            d[f"over_{L}"] = float(row[f"p_corners_over_{L}"])
        corner_table[f"{h}||{a}"] = d

    # Build upcoming-fixtures payload with model predictions and value bets.
    upcoming_payload: list[dict] = []
    for _, r in upcoming.iterrows():
        pred = predict_markets(ft, ht, r["home"], r["away"])
        cands = market_candidates_from_prediction(pred, r.to_dict())
        vbets = find_value_bets(cands, threshold=EDGE_THRESHOLD, kelly_cap=KELLY_CAP)
        upcoming_payload.append(
            {
                "date": str(r["date"]),
                "gw": int(r["gw"]) if pd.notna(r["gw"]) else None,
                "home": r["home"],
                "away": r["away"],
                "odds": {
                    "h": _float_or_none(r.get("odds_h")),
                    "d": _float_or_none(r.get("odds_d")),
                    "a": _float_or_none(r.get("odds_a")),
                    "o15": _float_or_none(r.get("odds_o15")),
                    "o25": _float_or_none(r.get("odds_o25")),
                    "o35": _float_or_none(r.get("odds_o35")),
                    "o45": _float_or_none(r.get("odds_o45")),
                    "btts_y": _float_or_none(r.get("odds_btts_y")),
                    "btts_n": _float_or_none(r.get("odds_btts_n")),
                },
                "value_bets": [
                    {
                        "market": vb.market,
                        "selection": vb.selection,
                        "model_prob": vb.model_prob,
                        "odds": vb.odds,
                        "edge": vb.edge,
                        "kelly_stake_pct": vb.kelly_stake,
                    }
                    for vb in vbets
                ],
            }
        )

    return {
        "teams": teams,
        "ft": _dc_payload(ft),
        "ht": _dc_payload(ht),
        "corner_lines": list(CORNER_LINES),
        "corners": corner_table,
        "upcoming": upcoming_payload,
        "config": {
            "edge_threshold": EDGE_THRESHOLD,
            "kelly_cap": KELLY_CAP,
        },
    }


def _float_or_none(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v) or v <= 1.0:
        return None
    return v


HTML_SHELL = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Peru Primera Prediction Model</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {
    --bg: #0f1419;
    --panel: #1a2030;
    --panel-2: #232a3d;
    --border: #2c3550;
    --text: #e5e9f0;
    --muted: #8a93a6;
    --accent: #4cc9f0;
    --good: #4ade80;
    --bad: #f87171;
    --warn: #fbbf24;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.4;
  }
  header {
    padding: 22px 28px 12px;
    border-bottom: 1px solid var(--border);
  }
  header h1 { margin: 0 0 4px 0; font-size: 22px; font-weight: 700; }
  header .sub { color: var(--muted); font-size: 13px; }
  main {
    padding: 22px 28px;
    max-width: 1280px;
    margin: 0 auto;
  }
  .controls {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    align-items: end;
    margin-bottom: 22px;
  }
  .control { display: flex; flex-direction: column; gap: 6px; }
  .control label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
  select, input[type=number] {
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 14px;
    min-width: 200px;
  }
  .lambdas {
    color: var(--muted);
    font-size: 13px;
    margin: 8px 0 16px;
  }
  .lambdas b { color: var(--text); }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 16px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
  }
  .card h3 {
    margin: 0 0 10px;
    font-size: 14px;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: .06em;
  }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { text-align: left; padding: 6px 8px; }
  tbody tr:nth-child(odd) td { background: rgba(255,255,255,0.02); }
  th { color: var(--muted); font-weight: 500; font-size: 12px; text-transform: uppercase; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  td.bar { position: relative; }
  .bar-fill {
    position: absolute; left: 0; top: 0; bottom: 0;
    background: rgba(76, 201, 240, 0.18);
    border-right: 1px solid rgba(76, 201, 240, 0.35);
  }
  .pill {
    display: inline-block; padding: 2px 6px; border-radius: 4px;
    font-size: 11px; font-weight: 600;
  }
  .pill-good { background: rgba(74, 222, 128, 0.18); color: var(--good); }
  .pill-bad  { background: rgba(248, 113, 113, 0.18); color: var(--bad); }
  .pill-warn { background: rgba(251, 191, 36, 0.18); color: var(--warn); }
  .tab-bar {
    display: flex; gap: 4px; margin: 28px 0 12px;
    border-bottom: 1px solid var(--border);
  }
  .tab {
    background: transparent; border: 0; color: var(--muted);
    padding: 8px 14px; cursor: pointer; font-size: 14px;
    border-bottom: 2px solid transparent;
  }
  .tab.active { color: var(--text); border-bottom-color: var(--accent); }
  .tab-pane { display: none; }
  .tab-pane.active { display: block; }
  .odds-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .odds-row label { font-size: 12px; color: var(--muted); }
  .odds-row input { width: 90px; min-width: 80px; }
  .footer-note { color: var(--muted); font-size: 12px; margin-top: 32px; padding-top: 12px; border-top: 1px solid var(--border); }
  .edge-pos { color: var(--good); font-weight: 600; }
  .edge-neg { color: var(--muted); }
  .upcoming-controls { display: flex; gap: 12px; align-items: center; margin-bottom: 10px; }
  .scroll { overflow-x: auto; }
</style>
</head>
<body>
<header>
  <h1>Peru Primera Division – Prediction Model</h1>
  <div class="sub">Dixon-Coles bivariate Poisson (FT + HT) with time decay, LightGBM corners. Trained on 2024-2026 league data.</div>
</header>
<main>
  <div class="controls">
    <div class="control">
      <label>Home team</label>
      <select id="home"></select>
    </div>
    <div class="control">
      <label>Away team</label>
      <select id="away"></select>
    </div>
    <div class="control">
      <label>Display</label>
      <select id="display">
        <option value="percent">Probability (%)</option>
        <option value="decimal">Fair decimal odds</option>
      </select>
    </div>
  </div>

  <div class="tab-bar">
    <button class="tab active" data-tab="predict">Prediction</button>
    <button class="tab" data-tab="value">Value finder</button>
    <button class="tab" data-tab="upcoming">Upcoming fixtures</button>
    <button class="tab" data-tab="ratings">Team ratings</button>
  </div>

  <section id="tab-predict" class="tab-pane active">
    <div class="lambdas" id="lambdas"></div>
    <div class="grid">
      <div class="card">
        <h3>Full-time 1X2</h3>
        <table><tbody id="ft-1x2"></tbody></table>
      </div>
      <div class="card">
        <h3>Over / Under (FT)</h3>
        <table><tbody id="ft-ou"></tbody></table>
      </div>
      <div class="card">
        <h3>BTTS &amp; First to score</h3>
        <table><tbody id="ft-btts-fts"></tbody></table>
      </div>
      <div class="card">
        <h3>Half-time 1X2</h3>
        <table><tbody id="ht-1x2"></tbody></table>
      </div>
      <div class="card">
        <h3>Half-time O/U &amp; BTTS</h3>
        <table><tbody id="ht-ou-btts"></tbody></table>
      </div>
      <div class="card">
        <h3>Top correct scores</h3>
        <table><tbody id="correct-score"></tbody></table>
      </div>
      <div class="card">
        <h3>Corners (LightGBM)</h3>
        <table><tbody id="corners"></tbody></table>
      </div>
    </div>
  </section>

  <section id="tab-value" class="tab-pane">
    <div class="card">
      <h3>Live value finder – enter the bookmaker's decimal odds</h3>
      <div class="lambdas">
        Model prob &times; your odds &minus; 1 = edge. Positive edge ⇒ value bet.
        Kelly stake assumes <span id="kelly-cap-display"></span> Kelly cap.
      </div>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th>Market</th>
              <th>Selection</th>
              <th class="num">Model</th>
              <th>Odds</th>
              <th class="num">Edge</th>
              <th class="num">Kelly %</th>
            </tr>
          </thead>
          <tbody id="value-table"></tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="tab-upcoming" class="tab-pane">
    <div class="card">
      <h3>Pre-computed value bets in remaining 2026 fixtures</h3>
      <div class="upcoming-controls">
        <label>Min edge:</label>
        <input id="upcoming-edge" type="number" step="0.01" min="0" max="1" value="0.03">
      </div>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Match</th>
              <th>Market</th>
              <th>Selection</th>
              <th class="num">Model</th>
              <th class="num">Odds</th>
              <th class="num">Edge</th>
              <th class="num">Kelly %</th>
            </tr>
          </thead>
          <tbody id="upcoming-table"></tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="tab-ratings" class="tab-pane">
    <div class="grid">
      <div class="card">
        <h3>Attack ratings (FT)</h3>
        <table><tbody id="attack-table"></tbody></table>
      </div>
      <div class="card">
        <h3>Defense ratings (FT) – higher = concedes more</h3>
        <table><tbody id="defense-table"></tbody></table>
      </div>
      <div class="card">
        <h3>Model parameters</h3>
        <table><tbody id="param-table"></tbody></table>
      </div>
    </div>
  </section>

  <div class="footer-note">
    Models trained on 3 seasons of league data (~760 completed matches).
    Probabilities are model output, not advice. Bet responsibly.
  </div>
</main>

<script>
// ====================================================================================
// Embedded payload
// ====================================================================================
const DATA = __PAYLOAD__;

// ====================================================================================
// Math: Poisson PMF + Dixon-Coles score matrix
// ====================================================================================
const FACT = [1];
for (let i = 1; i <= 16; i++) FACT.push(FACT[i - 1] * i);
function poissonPmf(k, lam) {
  if (lam <= 0) return k === 0 ? 1 : 0;
  return Math.exp(-lam) * Math.pow(lam, k) / FACT[k];
}

function lambdas(dc, home, away) {
  const ah = dc.attack[home] ?? 0;
  const dh = dc.defense[home] ?? 0;
  const aa = dc.attack[away] ?? 0;
  const da = dc.defense[away] ?? 0;
  const lh = Math.exp(ah + da + dc.home_adv);
  const la = Math.exp(aa + dh);
  return [lh, la];
}

function scoreMatrix(lh, la, rho, maxGoals = 10) {
  const n = maxGoals + 1;
  const ph = new Array(n), pa = new Array(n);
  for (let i = 0; i < n; i++) {
    ph[i] = poissonPmf(i, lh);
    pa[i] = poissonPmf(i, la);
  }
  const m = [];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    m.push(new Array(n));
    for (let j = 0; j < n; j++) m[i][j] = ph[i] * pa[j];
  }
  m[0][0] *= 1 - lh * la * rho;
  m[0][1] *= 1 + lh * rho;
  m[1][0] *= 1 + la * rho;
  m[1][1] *= 1 - rho;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    if (m[i][j] < 0) m[i][j] = 0;
    sum += m[i][j];
  }
  if (sum > 0) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) m[i][j] /= sum;
  return m;
}

function markets(mat) {
  const n = mat.length;
  let h = 0, d = 0, a = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    if (i > j) h += mat[i][j];
    else if (i === j) d += mat[i][j];
    else a += mat[i][j];
  }
  const totals = new Array(2 * n - 1).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) totals[i + j] += mat[i][j];
  let pHomeZero = 0, pAwayZero = 0;
  for (let j = 0; j < n; j++) pHomeZero += mat[0][j];
  for (let i = 0; i < n; i++) pAwayZero += mat[i][0];
  const bttsNo = pHomeZero + pAwayZero - mat[0][0];
  const bttsYes = 1 - bttsNo;
  // Correct score top
  const flat = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) flat.push([`${i}-${j}`, mat[i][j]]);
  flat.sort((x, y) => y[1] - x[1]);
  return {h, d, a, totals, bttsYes, bttsNo, correctScores: flat.slice(0, 10)};
}

function overUnder(totals, line) {
  const cutoff = Math.floor(line) + 1;
  let over = 0;
  for (let k = cutoff; k < totals.length; k++) over += totals[k];
  return [over, 1 - over];
}

function firstToScore(lh, la) {
  const tot = lh + la;
  if (tot <= 0) return [0, 0, 1];
  const noGoal = Math.exp(-tot);
  return [(lh / tot) * (1 - noGoal), (la / tot) * (1 - noGoal), noGoal];
}

function kelly(p, odds, cap) {
  if (!odds || odds <= 1) return [0, 0];
  const b = odds - 1;
  const f = (b * p - (1 - p)) / b;
  if (f <= 0) return [f, 0];
  return [f, Math.max(0, Math.min(1, f * cap))];
}
function edge(p, odds) { return odds && odds > 1 ? p * odds - 1 : NaN; }

// ====================================================================================
// Rendering
// ====================================================================================
function fmt(p, mode) {
  if (mode === "decimal") return p > 0 ? (1 / p).toFixed(2) : "∞";
  return (p * 100).toFixed(1) + "%";
}
function bar(td, pct) {
  const fill = document.createElement("div");
  fill.className = "bar-fill";
  fill.style.width = (pct * 100) + "%";
  td.appendChild(fill);
}
function row(tbody, label, p, mode) {
  const tr = document.createElement("tr");
  const td1 = document.createElement("td"); td1.textContent = label; tr.appendChild(td1);
  const td2 = document.createElement("td"); td2.className = "num bar";
  bar(td2, p);
  const span = document.createElement("span");
  span.textContent = fmt(p, mode);
  span.style.position = "relative";
  td2.appendChild(span);
  tr.appendChild(td2);
  tbody.appendChild(tr);
}

function renderPrediction() {
  const home = document.getElementById("home").value;
  const away = document.getElementById("away").value;
  const mode = document.getElementById("display").value;
  if (!home || !away || home === away) {
    document.getElementById("lambdas").textContent = "Pick two different teams.";
    return;
  }
  const [lhFt, laFt] = lambdas(DATA.ft, home, away);
  const matFt = scoreMatrix(lhFt, laFt, DATA.ft.rho, 10);
  const mFt = markets(matFt);
  const [lhHt, laHt] = lambdas(DATA.ht, home, away);
  const matHt = scoreMatrix(lhHt, laHt, DATA.ht.rho, 6);
  const mHt = markets(matHt);
  const [fhH, fhA, fhN] = firstToScore(lhFt, laFt);

  document.getElementById("lambdas").innerHTML =
    `Expected goals — <b>${home}</b>: ${lhFt.toFixed(2)} &nbsp;·&nbsp; <b>${away}</b>: ${laFt.toFixed(2)} ` +
    `&nbsp;|&nbsp; HT λ: ${lhHt.toFixed(2)} / ${laHt.toFixed(2)}`;

  // FT 1X2
  const ft1x2 = document.getElementById("ft-1x2"); ft1x2.innerHTML = "";
  row(ft1x2, home + " win", mFt.h, mode);
  row(ft1x2, "Draw", mFt.d, mode);
  row(ft1x2, away + " win", mFt.a, mode);

  // FT O/U
  const ftOu = document.getElementById("ft-ou"); ftOu.innerHTML = "";
  for (const L of [1.5, 2.5, 3.5, 4.5]) {
    const [o, u] = overUnder(mFt.totals, L);
    row(ftOu, "Over " + L, o, mode);
    row(ftOu, "Under " + L, u, mode);
  }

  // BTTS + FTS
  const bf = document.getElementById("ft-btts-fts"); bf.innerHTML = "";
  row(bf, "BTTS Yes", mFt.bttsYes, mode);
  row(bf, "BTTS No", mFt.bttsNo, mode);
  row(bf, "First goal: " + home, fhH, mode);
  row(bf, "First goal: " + away, fhA, mode);
  row(bf, "No goal (0-0)", fhN, mode);

  // HT
  const ht1 = document.getElementById("ht-1x2"); ht1.innerHTML = "";
  row(ht1, home + " HT", mHt.h, mode);
  row(ht1, "Draw HT", mHt.d, mode);
  row(ht1, away + " HT", mHt.a, mode);

  const ht2 = document.getElementById("ht-ou-btts"); ht2.innerHTML = "";
  for (const L of [0.5, 1.5]) {
    const [o, u] = overUnder(mHt.totals, L);
    row(ht2, "HT Over " + L, o, mode);
    row(ht2, "HT Under " + L, u, mode);
  }
  row(ht2, "HT BTTS Yes", mHt.bttsYes, mode);
  row(ht2, "HT BTTS No", mHt.bttsNo, mode);

  // Correct scores
  const cs = document.getElementById("correct-score"); cs.innerHTML = "";
  for (const [score, p] of mFt.correctScores) row(cs, score, p, mode);

  // Corners
  const corners = document.getElementById("corners"); corners.innerHTML = "";
  const key = home + "||" + away;
  const c = DATA.corners[key];
  if (c) {
    const tr = (label, val) => {
      const r = document.createElement("tr");
      r.innerHTML = `<td>${label}</td><td class="num">${val}</td>`;
      corners.appendChild(r);
    };
    tr("λ home corners", c.lam_h.toFixed(2));
    tr("λ away corners", c.lam_a.toFixed(2));
    tr("λ total corners", c.lam_total.toFixed(2));
    for (const L of DATA.corner_lines) {
      const p = c["over_" + L];
      const node = document.createElement("tr");
      const td1 = document.createElement("td"); td1.textContent = "Over " + L;
      const td2 = document.createElement("td"); td2.className = "num bar";
      bar(td2, p);
      const span = document.createElement("span"); span.textContent = fmt(p, mode);
      span.style.position = "relative"; td2.appendChild(span);
      node.appendChild(td1); node.appendChild(td2);
      corners.appendChild(node);
    }
  } else {
    corners.innerHTML = `<tr><td>Corners not pre-computed for this pairing.</td></tr>`;
  }

  renderValueFinder({mode, home, away, mFt, mHt, fhH, fhA, fhN});
}

function makeOddsInput(id, def) {
  const el = document.createElement("input");
  el.type = "number"; el.step = "0.01"; el.min = "1.01"; el.id = id;
  if (def != null) el.value = def.toFixed(2);
  el.addEventListener("input", () => renderPrediction());
  return el;
}

function renderValueFinder(state) {
  const tb = document.getElementById("value-table");
  tb.innerHTML = "";
  document.getElementById("kelly-cap-display").textContent = (DATA.config.kelly_cap * 100).toFixed(0) + "%";
  const probs = [
    {market: "1X2", sel: state.home + " win", p: state.mFt.h, id: "v-h"},
    {market: "1X2", sel: "Draw", p: state.mFt.d, id: "v-d"},
    {market: "1X2", sel: state.away + " win", p: state.mFt.a, id: "v-a"},
    {market: "BTTS", sel: "Yes", p: state.mFt.bttsYes, id: "v-byes"},
    {market: "BTTS", sel: "No", p: state.mFt.bttsNo, id: "v-bno"},
  ];
  for (const L of [1.5, 2.5, 3.5, 4.5]) {
    const [o, u] = overUnder(state.mFt.totals, L);
    probs.push({market: "Goals", sel: "Over " + L, p: o, id: "v-o" + (L * 10)});
    probs.push({market: "Goals", sel: "Under " + L, p: u, id: "v-u" + (L * 10)});
  }
  probs.push({market: "FTS", sel: state.home + " first", p: state.fhH, id: "v-fh"});
  probs.push({market: "FTS", sel: state.away + " first", p: state.fhA, id: "v-fa"});
  probs.push({market: "FTS", sel: "No goal", p: state.fhN, id: "v-fn"});

  // Save previously entered odds so they survive re-render
  const prev = window.__lastOdds || {};
  for (const row of probs) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.market}</td>
      <td>${row.sel}</td>
      <td class="num">${(row.p * 100).toFixed(1)}%</td>
    `;
    const oddsTd = document.createElement("td");
    const inp = document.createElement("input");
    inp.type = "number"; inp.step = "0.01"; inp.min = "1.01";
    inp.style.width = "80px";
    inp.value = prev[row.id] ?? "";
    inp.addEventListener("input", () => {
      window.__lastOdds = window.__lastOdds || {};
      window.__lastOdds[row.id] = inp.value;
      recomputeEdge(row, inp, edgeCell, kellyCell);
    });
    oddsTd.appendChild(inp);
    tr.appendChild(oddsTd);
    const edgeCell = document.createElement("td"); edgeCell.className = "num"; tr.appendChild(edgeCell);
    const kellyCell = document.createElement("td"); kellyCell.className = "num"; tr.appendChild(kellyCell);
    tb.appendChild(tr);
    recomputeEdge(row, inp, edgeCell, kellyCell);
  }
}

function recomputeEdge(row, inp, edgeCell, kellyCell) {
  const odds = parseFloat(inp.value);
  if (!odds || odds <= 1 || !isFinite(odds)) {
    edgeCell.textContent = "—";
    kellyCell.textContent = "—";
    return;
  }
  const e = edge(row.p, odds);
  const [, k] = kelly(row.p, odds, DATA.config.kelly_cap);
  edgeCell.textContent = (e * 100).toFixed(1) + "%";
  edgeCell.className = "num " + (e > 0 ? "edge-pos" : "edge-neg");
  kellyCell.textContent = k > 0 ? (k * 100).toFixed(2) + "%" : "—";
}

function renderUpcoming() {
  const minEdge = parseFloat(document.getElementById("upcoming-edge").value) || 0;
  const tb = document.getElementById("upcoming-table");
  tb.innerHTML = "";
  const rows = [];
  for (const fx of DATA.upcoming) {
    for (const vb of fx.value_bets) {
      if (vb.edge >= minEdge) rows.push({fx, vb});
    }
  }
  rows.sort((x, y) => y.vb.edge - x.vb.edge);
  if (rows.length === 0) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="8" style="color: var(--muted)">No value bets above ${(minEdge*100).toFixed(1)}% edge.</td>`;
    tb.appendChild(tr);
    return;
  }
  for (const {fx, vb} of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${fx.date.slice(0, 10)}</td>
      <td>${fx.home} vs ${fx.away}</td>
      <td>${vb.market}</td>
      <td>${vb.selection}</td>
      <td class="num">${(vb.model_prob * 100).toFixed(1)}%</td>
      <td class="num">${vb.odds.toFixed(2)}</td>
      <td class="num edge-pos">${(vb.edge * 100).toFixed(1)}%</td>
      <td class="num">${(vb.kelly_stake_pct * 100).toFixed(2)}%</td>
    `;
    tb.appendChild(tr);
  }
}

function renderRatings() {
  const teams = DATA.teams.slice();
  const att = teams.map(t => [t, DATA.ft.attack[t] ?? 0]).sort((a, b) => b[1] - a[1]);
  const def = teams.map(t => [t, DATA.ft.defense[t] ?? 0]).sort((a, b) => b[1] - a[1]);
  const fill = (id, arr) => {
    const tb = document.getElementById(id); tb.innerHTML = "";
    for (const [t, v] of arr) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${t}</td><td class="num">${v.toFixed(3)}</td>`;
      tb.appendChild(tr);
    }
  };
  fill("attack-table", att);
  fill("defense-table", def);

  const pt = document.getElementById("param-table");
  const rows = [
    ["FT home advantage (log)", DATA.ft.home_adv.toFixed(3)],
    ["FT rho (low-score corr.)", DATA.ft.rho.toFixed(4)],
    ["FT league mean goals", DATA.ft.league_mean_goals.toFixed(2)],
    ["HT home advantage (log)", DATA.ht.home_adv.toFixed(3)],
    ["HT rho", DATA.ht.rho.toFixed(4)],
    ["HT league mean goals", DATA.ht.league_mean_goals.toFixed(2)],
    ["Number of teams", DATA.teams.length],
    ["Upcoming fixtures", DATA.upcoming.length],
  ];
  pt.innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td class="num">${v}</td></tr>`).join("");
}

// ====================================================================================
// Init
// ====================================================================================
function init() {
  const homeSel = document.getElementById("home");
  const awaySel = document.getElementById("away");
  for (const t of DATA.teams) {
    homeSel.appendChild(new Option(t, t));
    awaySel.appendChild(new Option(t, t));
  }
  // Default to the first upcoming fixture if any, else first two teams
  if (DATA.upcoming.length > 0) {
    const fx = DATA.upcoming[0];
    if (DATA.teams.includes(fx.home)) homeSel.value = fx.home;
    if (DATA.teams.includes(fx.away)) awaySel.value = fx.away;
  } else {
    homeSel.value = DATA.teams[0];
    awaySel.value = DATA.teams[1];
  }
  homeSel.addEventListener("change", renderPrediction);
  awaySel.addEventListener("change", renderPrediction);
  document.getElementById("display").addEventListener("change", renderPrediction);
  document.getElementById("upcoming-edge").addEventListener("input", renderUpcoming);

  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    });
  });

  renderPrediction();
  renderUpcoming();
  renderRatings();
}
init();
</script>
</body>
</html>
"""


def main(out_path: str = "peru_pred.html") -> None:
    payload = build_payload()
    json_text = json.dumps(payload, default=str)
    # Embed safely — pre-escape any </script> sequences (defensive)
    json_text = json_text.replace("</", "<\\/")
    page = HTML_SHELL.replace("__PAYLOAD__", json_text)
    Path(out_path).write_text(page, encoding="utf-8")
    print(f"Wrote {out_path} ({len(page) / 1024:.1f} KB)")


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "peru_pred.html"
    main(out)
