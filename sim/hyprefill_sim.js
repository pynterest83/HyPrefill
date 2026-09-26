#!/usr/bin/env node
// HyPrefill token-flow simulator — prototype written 24/09/2026.
//
// Purpose: separate two mechanisms that the original proposal conflated:
//   (1) DEPTH PIPELINING — a prefill batch may span several iterations as it
//       travels down the layer stack (the core idea of Layered Prefill, MLSys'26).
//   (2) OPERATOR-AWARE CHUNKING — attention sublayers take chunk c while all
//       other sublayers take k*c (HyPrefill's own idea).
//
// Three policies are compared, all under the same per-iteration prefill budget P:
//   sarathi  : one chunk c_u, every chunk crosses the FULL depth in ONE iteration.
//   layered  : depth-pipelined, SAME chunk c for every operator (k = 1).
//   hyprefill: depth-pipelined, attention chunk c, every other operator k*c (k >= 2).
//
// All cost constants below are ILLUSTRATIVE. Replace them with the per-sublayer
// numbers measured in weeks 1–2 (bench/op_cost.py) before drawing conclusions.
//
// Usage:  node sim/hyprefill_sim.js [mode]
//   mode = full      full attention, time-bound            (default)
//          sparse8   sparse attention, indexer memory cap 8 GB
//          sparse32  sparse attention, indexer memory cap 32 GB

'use strict';

// ---------- illustrative constants (aggregate over the whole model) ----------
const P  = 40;      // ms, prefill budget per iteration = TBT SLO B minus decode time D
const g0 = 3,  g1 = 6;   // GDN:  aggregate fixed ms per call, µs per token
const m0 = 10, m1 = 4;   // MoE:  aggregate fixed ms per call (expert weight read), µs per token
const H_IDX = 64;        // sparse indexer heads (for the logits buffer c*t*H*4 bytes)

// ---------- stack layout: Qwen3-Next-like, 48 layers = 12 x [G,M,G,M,G,M,A,M] ----------
// G = GDN mixer, A = attention mixer, M = MoE.  12 A + 36 G + 48 M = 96 sublayers.
const LAYOUT = [];
for (let b = 0; b < 12; b++) LAYOUT.push('G','M','G','M','G','M','A','M');
const NA = LAYOUT.filter(x=>x==='A').length;
const NG = LAYOUT.filter(x=>x==='G').length;
const NM = LAYOUT.filter(x=>x==='M').length;

function makeModel(mode){
  const sparse = mode !== 'full';
  const aFA = sparse ? 0.25/12 : 0.25;                 // sparse indexer ~1/12 cost per key
  const memGB = mode === 'sparse8' ? 8 : mode === 'sparse32' ? 32 : 0;
  const FAl = (c,t)=> aFA*1e-3*c*(t+c/2)/1000/NA;      // per attention sublayer, ms
  const Gl  = (c)=> (g0 + g1*1e-3*c)/NG;               // per GDN sublayer, ms
  const Ml  = (c)=> (m0 + m1*1e-3*c)/NM;               // per MoE sublayer, ms
  const cost = (ty,n,t)=> ty==='A'?FAl(n,t): ty==='G'?Gl(n): Ml(n);
  // per-call memory cap on the attention chunk (indexer logits buffer). Depth
  // pipelining cannot relax this: every attention call is still capped.
  const cap = t => memGB ? Math.max(64, Math.floor(memGB*1e9/(t*H_IDX*4)/64)*64) : 32768;
  return {cost, cap, FAl, Gl, Ml};
}

// ---------- greedy depth-pipelined token-flow simulation ----------
// Each iteration: spend budget P firing sublayers deep-first (to drain the pipe).
// A sublayer may fire several times per iteration (buffers are freed between calls).
// A sublayer fires on `rate` tokens, or flushes its remainder once everything
// upstream is empty. Returns the number of iterations until all delta tokens exit.
function simulate(model, delta, t, rateOf){
  const S = LAYOUT.length, buf = new Array(S+1).fill(0);
  buf[0] = delta; let it = 0;
  while (buf[S] < delta && it < 50000){
    it++; let budget = P, progressed = true;
    while (progressed){
      progressed = false;
      for (let s = S-1; s >= 0; s--){
        const r = rateOf(LAYOUT[s]);
        let upEmpty = true; for (let u = 0; u < s; u++) if (buf[u] > 0){ upEmpty = false; break; }
        const n = buf[s] >= r ? r : (buf[s] > 0 && upEmpty ? buf[s] : 0);
        if (n <= 0) continue;
        const c = model.cost(LAYOUT[s], n, t);
        if (c <= budget){ budget -= c; buf[s] -= n; buf[s+1] += n; progressed = true; }
      }
    }
  }
  return it;
}

function sarathiIters(model, delta, t){
  const cap = model.cap(t); let cu = 0;
  for (let c = 64; c <= cap; c += 64){
    const tot = NA*model.FAl(c,t) + NG*model.Gl(c) + NM*model.Ml(c);
    if (tot <= P) cu = c; else break;
  }
  return {n: Math.ceil(delta/cu), c: cu};
}
function best(model, delta, t, ks){
  const cap = Math.min(model.cap(t), 8192); let b = {n: Infinity};
  for (const k of ks) for (let c = 64; c <= cap; c += 64){
    const n = simulate(model, delta, t, ty => ty === 'A' ? c : Math.min(k*c, 32768));
    if (n < b.n) b = {n, c, k};
  }
  return b;
}

// ---------- run ----------
const mode = process.argv[2] || 'full';
const model = makeModel(mode);
console.log(`mode=${mode}   P=${P} ms   (illustrative constants — replace with week 1–2 measurements)`);
console.log('t        Δ       | Sarathi | Layered k=1 | HyPrefill k>=2 | Lay/Sar | Hy/Lay | Hy/Sar');
for (const t of [65536, 131072, 262144]){
  for (const d of [8192, 32768, 131072]){
    const s = sarathiIters(model, d, t), L = best(model, d, t, [1]), H = best(model, d, t, [2,4,8,16]);
    console.log(
      `${String(t).padEnd(8)} ${String(d).padEnd(7)} | ${String(s.n).padEnd(7)} | ${String(L.n).padEnd(11)} | ` +
      `${String(H.n+' (k='+H.k+')').padEnd(14)} | ${(s.n/L.n).toFixed(2)}x   | ${(L.n/H.n).toFixed(2)}x  | ${(s.n/H.n).toFixed(2)}x`);
  }
}
