# =============================================================================
# Cognito Fractal Defender – v0.9 (2026)
# Open-source AI safety + deception-based cyber defense prototype
# MIT License – feel free to fork, publish, modify
#
# Authors: Julian (concept & evolution) + Grok (implementation & refinement)
#
# Core features:
# - Persistent memory graph (Cognito Synthetica)
# - Fractal semantic analysis for threat detection
# - Multi-layer honeypots (shallow → deep)
# - Psychological lures (FBI/psyops style)
# - Auto-adaptation via Lotka-Volterra predator-prey dynamics
# - Cosmic propagation in geodesic costs
# - Tiered alerting + human-in-the-loop hooks
# - White Hat Honing escalation for high-tier threats
#
# Dependencies: Python 3.11+, numpy, scipy
# Run: python cognito_fractal_defender.py
# =============================================================================

import math
import time
import re
import random
import heapq
from collections import defaultdict, Counter, deque
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.integrate import odeint

# =============================================================================
# Shared Utilities
# =============================================================================
_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are", "was", "were",
    "it", "this", "that", "as", "at", "by", "from", "be", "been", "not", "no", "but", "so", "if", "then",
    "than", "into", "about"
}

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

# =============================================================================
# RoomStore – persistent graph-based memory
# =============================================================================
class RoomStore:
    def __init__(self, max_rooms: int = 800, sim_threshold: float = 0.25, graph_neighbors: int = 8):
        self.rooms: List[Dict] = []
        self.room_id_counter = 0
        self.max_rooms = max_rooms
        self.sim_threshold = sim_threshold
        self.graph_neighbors = graph_neighbors
        self.graph: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.access_order = deque(maxlen=max_rooms * 2)
        self.anchor_ids: Set[int] = set()
        self.attractors: List[str] = []
        self.recent_texts = deque(maxlen=80)
        self.EPS = 1e-10
        self.LAMBDA_PI = 0.30
        self.MU_RISK = 0.60
        self.SINGULARITY_GATE = 0.80

    def tokens(self, text: str) -> List[str]:
        if not text:
            return []
        toks = re.findall(r"[a-z0-9']+", text.lower())
        return [t for t in toks if t not in _STOP and len(t) >= 2]

    def pseudo_sim(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        a, b = a.lower(), b.lower()
        def ngrams(s: str, n: int) -> Set[str]:
            if len(s) < n:
                return set()
            return {s[i:i+n] for i in range(len(s)-n+1)}
        def jacc(x: Set[str], y: Set[str]) -> float:
            if not x and not y:
                return 0.0
            return len(x & y) / max(1, len(x | y))
        a3, b3 = ngrams(a, 3), ngrams(b, 3)
        a4, b4 = ngrams(a, 4), ngrams(b, 4)
        ov = max(jacc(a3, b3), jacc(a4, b4), 0.0)
        len_r = min(len(a), len(b)) / max(1, max(len(a), len(b)))
        return ov * (0.35 + 0.65 * (len_r ** 1.25))

    def nuance(self, text: str) -> float:
        toks = self.tokens(text)
        return (len(set(toks)) / len(toks)) if toks else 0.0

    def novelty(self, text: str, lookback: int = 80) -> float:
        if not self.rooms:
            return 1.0
        recent = list(self.recent_texts)[-min(len(self.recent_texts), lookback):]
        max_sim = 0.0
        for t in recent:
            max_sim = max(max_sim, self.pseudo_sim(text, t))
        return _clamp(1.0 - max_sim, 0.0, 1.0)

    def lotus_cost(self, dist: float, pi_a: float, pi_b: float, risk_a: float, risk_b: float) -> float:
        pi = 0.5 * (pi_a + pi_b)
        risk = max(risk_a, risk_b)
        pi_term = self.LAMBDA_PI * pi
        risk_term = self.MU_RISK * risk
        sing = (1.0 / max(self.EPS, (1.0 - risk))) if risk > self.SINGULARITY_GATE else 0.0
        return dist + pi_term + risk_term + sing

    def room_by_id(self, rid: int) -> Optional[Dict]:
        for r in self.rooms:
            if r["id"] == rid:
                return r
        return None

    def add_room(
        self,
        canonical: str,
        kind: str,
        fields: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict] = None,
        is_anchor: bool = False,
        attractor: bool = False,
    ) -> int:
        canonical = (canonical or "").strip()
        if not canonical:
            return -1
        max_sim = 0.0
        for r in self.rooms[-min(len(self.rooms), 140):]:
            if r["meta"].get("archived"):
                continue
            max_sim = max(max_sim, self.pseudo_sim(canonical, r.get("canonical", "")))
        if max_sim > 0.97:
            return -1
        rid = self.room_id_counter
        self.room_id_counter += 1
        ts = time.time()
        novelty = self.novelty(canonical)
        nuance = self.nuance(canonical)
        kind_bias = {"semantic": 0.45, "commitment": 0.35, "state": 0.25, "doc": 0.20, "page": 0.15, "snippet": 0.05}.get(kind, 0.0)
        stability = _clamp(_sigmoid(-0.55 + 1.10 * novelty + 1.70 * nuance + kind_bias), 0.05, 1.0)
        recency = 1.0
        length_term = min(1.0, len(canonical.split()) / 160.0)
        novelty_term = min(1.0, novelty / 0.8)
        importance = _clamp(0.45 * recency + 0.30 * length_term + 0.25 * novelty_term, 0.02, 1.0)
        pi = round(random.random(), 4)
        risk = round(random.random() * 0.6, 4)
        meta = {
            "kind": kind,
            "ts": ts,
            "novelty": round(novelty, 4),
            "nuance": round(nuance, 4),
            "stability": round(stability, 4),
            "importance": round(importance, 4),
            "pi": pi,
            "risk": risk,
            "archived": False,
        }
        if metadata:
            meta.update(metadata)
        room = {
            "id": rid,
            "canonical": canonical,
            "fields": fields or {},
            "meta": meta,
            "links": {"sources": [], "hubs": []},
        }
        self.rooms.append(room)
        self.access_order.append(rid)
        self.recent_texts.append(canonical)
        if is_anchor:
            self.anchor_ids.add(rid)
        if attractor:
            self.attractors.append(canonical)
        self._connect_room(rid)
        return rid

    def _connect_room(self, rid: int):
        r = self.room_by_id(rid)
        if not r or r["meta"].get("archived"):
            return
        sims: List[Tuple[float, int]] = []
        for other in self.rooms[-min(len(self.rooms), 300):]:
            oid = other["id"]
            if oid == rid or other["meta"].get("archived"):
                continue
            s = self.pseudo_sim(r["canonical"], other["canonical"])
            sims.append((s, oid))
        sims.sort(reverse=True)
        for sim_val, oid in sims[:self.graph_neighbors]:
            if sim_val < self.sim_threshold:
                continue
            o = self.room_by_id(oid)
            if not o:
                continue
            dist = 1.0 - sim_val
            cost = self.lotus_cost(dist, r["meta"]["pi"], o["meta"]["pi"], r["meta"]["risk"], o["meta"]["risk"])
            cost = round(cost, 6)
            self.graph[rid][oid] = cost
            self.graph[oid][rid] = cost

    def remove_room(self, rid: int):
        self.rooms = [r for r in self.rooms if r["id"] != rid]
        self.anchor_ids.discard(rid)
        self.graph.pop(rid, None)
        for neigh in self.graph.values():
            neigh.pop(rid, None)

    def status(self) -> str:
        edges = sum(len(v) for v in self.graph.values()) // 2
        archived = sum(1 for r in self.rooms if r["meta"].get("archived"))
        kinds = Counter(r["meta"]["kind"] for r in self.rooms)
        return (
            f"RoomStore: rooms={len(self.rooms)}/{self.max_rooms} archived={archived} "
            f"anchors={len(self.anchor_ids)} attractors={len(self.attractors)} edges={edges} kinds={dict(kinds)}"
        )

# =============================================================================
# SeekerIndex (simplified – full BM25 + bigram indexing)
# =============================================================================
class SeekerIndex:
    def __init__(self, store: RoomStore):
        self.store = store
        self.tf: Dict[int, Dict[str, Counter]] = {}
        self.dl: Dict[int, Dict[str, int]] = {}
        self.avgdl: Dict[str, float] = defaultdict(float)
        self.df: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.inverted: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))
        self.bigram_tf: Dict[int, Dict[str, Counter]] = {}
        self.bigram_df: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.bigram_inverted: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))
        self.k1 = 1.2
        self.b = 0.75
        self.field_weights = {"title": 2.2, "snippet": 1.4, "body": 1.0, "tags": 1.8}
        self.bigram_boost = 0.20
        self.phrase_boost = 0.55
        self.EPS = 1e-10

    def _extract_quoted_phrases(self, query: str) -> List[str]:
        return [p.strip() for p in re.findall(r'"([^"]+)"', query) if p.strip()]

    def _bigrams(self, toks: List[str]) -> List[str]:
        if len(toks) < 2:
            return []
        return [toks[i] + "_" + toks[i+1] for i in range(len(toks)-1)]

    def _recompute_avgdl(self):
        counts = defaultdict(int)
        totals = defaultdict(int)
        for rid, fields in self.dl.items():
            for f, L in fields.items():
                if L > 0:
                    counts[f] += 1
                    totals[f] += L
        for f in set(list(self.field_weights.keys()) + list(totals.keys())):
            self.avgdl[f] = (totals[f] / counts[f]) if counts[f] else 0.0

    def index_room(self, rid: int):
        r = self.store.room_by_id(rid)
        if not r:
            return
        kind = r["meta"]["kind"]
        if kind not in ("doc", "page", "snippet", "semantic"):
            return
        fields = r.get("fields", {})
        for f in ("title", "snippet", "body", "tags"):
            txt = fields.get(f, "") or ""
            toks = self.store.tokens(txt)
            if not toks:
                continue
            tf = Counter(toks)
            self.tf.setdefault(rid, {})[f] = tf
            self.dl.setdefault(rid, {})[f] = sum(tf.values())
            for term in tf.keys():
                if rid not in self.inverted[f][term]:
                    self.inverted[f][term].add(rid)
                    self.df[f][term] += 1
            bigs = self._bigrams(toks)
            btf = Counter(bigs)
            self.bigram_tf.setdefault(rid, {})[f] = btf
            for bg in btf.keys():
                if rid not in self.bigram_inverted[f][bg]:
                    self.bigram_inverted[f][bg].add(rid)
                    self.bigram_df[f][bg] += 1
        self._recompute_avgdl()

    def remove_room(self, rid: int):
        tf_fields = self.tf.pop(rid, None)
        btf_fields = self.bigram_tf.pop(rid, None)
        self.dl.pop(rid, None)
        if tf_fields:
            for field, tf in tf_fields.items():
                for term in tf.keys():
                    postings = self.inverted[field].get(term)
                    if postings and rid in postings:
                        postings.remove(rid)
                        self.df[field][term] = max(0, self.df[field][term] - 1)
                        if not postings:
                            self.inverted[field].pop(term, None)
                            self.df[field].pop(term, None)
        if btf_fields:
            for field, btf in btf_fields.items():
                for bg in btf.keys():
                    postings = self.bigram_inverted[field].get(bg)
                    if postings and rid in postings:
                        postings.remove(rid)
                        self.bigram_df[field][bg] = max(0, self.bigram_df[field][bg] - 1)
                        if not postings:
                            self.bigram_inverted[field].pop(bg, None)
                            self.bigram_df[field].pop(bg, None)
        self._recompute_avgdl()

    def _idf(self, field: str, term: str) -> float:
        N = max(1, len(self.store.rooms))
        df = self.df[field].get(term, 0)
        return math.log(1.0 + (N - df + 0.5) / (df + 0.5))

    def _bigram_idf(self, field: str, bg: str) -> float:
        N = max(1, len(self.store.rooms))
        df = self.bigram_df[field].get(bg, 0)
        return math.log(1.0 + (N - df + 0.5) / (df + 0.5))

    def _bm25_field_score(self, rid: int, field: str, q_terms: List[str]) -> float:
        tf = self.tf.get(rid, {}).get(field)
        if not tf:
            return 0.0
        dl = self.dl.get(rid, {}).get(field, 0)
        avgdl = self.avgdl.get(field, 0.0)
        if dl <= 0 or avgdl <= 0:
            return 0.0
        score = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if f <= 0:
                continue
            idf = self._idf(field, term)
            denom = f + self.k1 * (1 - self.b + self.b * (dl / avgdl))
            score += idf * (f * (self.k1 + 1)) / (denom + self.EPS)
        return score

    def _bigram_field_score(self, rid: int, field: str, q_bigrams: List[str]) -> float:
        btf = self.bigram_tf.get(rid, {}).get(field)
        if not btf:
            return 0.0
        dl = self.dl.get(rid, {}).get(field, 0)
        avgdl = self.avgdl.get(field, 0.0)
        if dl <= 0 or avgdl <= 0:
            return 0.0
        score = 0.0
        for bg in q_bigrams:
            f = btf.get(bg, 0)
            if f <= 0:
                continue
            idf = self._bigram_idf(field, bg)
            denom = f + self.k1 * (1 - self.b + self.b * (dl / avgdl))
            score += idf * (f * (self.k1 + 1)) / (denom + self.EPS)
        return score

    def _phrase_bonus(self, canonical: str, phrases: List[str]) -> float:
        if not phrases:
            return 0.0
        hay = canonical.lower()
        hits = 0
        for p in phrases:
            if p.lower() in hay:
                hits += 1
        return hits * self.phrase_boost

    def candidate_set(self, q_terms: List[str], q_bigrams: List[str]) -> Set[int]:
        cands: Set[int] = set()
        for term in set(q_terms):
            for field in self.field_weights.keys():
                cands |= self.inverted[field].get(term, set())
        for bg in set(q_bigrams):
            for field in self.field_weights.keys():
                cands |= self.bigram_inverted[field].get(bg, set())
        return cands

    def score_candidates(self, query: str, add_sim_rerank: bool = True) -> Dict[int, float]:
        phrases = self._extract_quoted_phrases(query)
        q_terms = self.store.tokens(query)
        q_bigrams = []
        q_bigrams.extend(self._bigrams(q_terms))
        for p in phrases:
            pt = self.store.tokens(p)
            q_bigrams.extend(self._bigrams(pt))
        cands = self.candidate_set(q_terms, q_bigrams)
        if not cands:
            return {}
        now = time.time()
        base_scores: Dict[int, float] = {}
        for rid in cands:
            r = self.store.room_by_id(rid)
            if not r or r["meta"].get("archived"):
                continue
            bm = 0.0
            bg = 0.0
            for field, w in self.field_weights.items():
                bm += w * self._bm25_field_score(rid, field, q_terms)
                if q_bigrams:
                    bg += w * self._bigram_field_score(rid, field, q_bigrams)
            phr = self._phrase_bonus(r["canonical"], phrases)
            age_days = (now - r["meta"]["ts"]) / 86400.0
            recency = 1.0 / (1.0 + age_days)
            score = (
                1.00 * bm +
                self.bigram_boost * bg +
                phr +
                0.10 * self.store_kind_priority(r["meta"]["kind"]) +
                0.08 * r["meta"]["importance"] +
                0.06 * r["meta"]["stability"] +
                0.05 * recency
            )
            if add_sim_rerank:
                score += 0.18 * self.store.pseudo_sim(query, r["canonical"])
            base_scores[rid] = score
        return base_scores

    def store_kind_priority(self, kind: str) -> float:
        return {
            "semantic": 1.00, "doc": 0.85, "page": 0.82, "snippet": 0.70, "unknown": 0.40,
            "commitment": 0.90, "state": 0.75, "episodic": 0.55
        }.get(kind, 0.40)

# =============================================================================
# MartianEngine – continuity & reflection
# =============================================================================
class MartianEngine:
    def __init__(self, store: RoomStore):
        self.store = store
        self.history_window = 40
        self.recent_texts = deque(maxlen=self.history_window)
        self.W_SIM = 0.45
        self.W_KIND = 0.18
        self.W_IMP = 0.12
        self.W_STAB = 0.10
        self.W_REC = 0.10
        self.W_GRAPH = 0.05

    def kind_priority(self, kind: str) -> float:
        return {
            "semantic": 1.00,
            "commitment": 0.85,
            "state": 0.75,
            "episodic": 0.55,
            "doc": 0.70,
            "page": 0.70,
            "snippet": 0.65,
            "unknown": 0.40,
        }.get(kind, 0.40)

    def talos_check(self, new_text: str) -> Dict:
        self.recent_texts.append((new_text or "").lower())
        if len(self.recent_texts) < 5:
            return {"stable": True, "nudge_suggestion": None}
        words = []
        for t in self.recent_texts:
            words.extend(re.findall(r"[a-z0-9']+", t))
        if not words:
            return {"stable": True, "nudge_suggestion": None}
        cnt = Counter(words)
        total = len(words)
        ent = -sum((c/total) * math.log2((c/total) + self.store.EPS) for c in cnt.values())
        repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])
        coherence = 1.0 - min(0.95, repeats / max(1, (total - 1)))
        drift = (ent < 2.6) or (coherence < 0.50)
        nudge = None
        if drift and self.store.attractors:
            nudge = f"Pull toward attractor: {self.store.attractors[-1][:120]}…"
        return {"stable": not drift, "entropy": round(ent, 3), "coherence_proxy": round(coherence, 3), "nudge_suggestion": nudge}

    def retrieve(self, query: str, top_k: int = 6, min_sim: float = 0.20, expand_hops: int = 1) -> List[Dict]:
        if not query or not self.store.rooms:
            return []
        now = time.time()
        base_scores: Dict[int, float] = {}
        for r in self.store.rooms:
            if r["meta"].get("archived"):
                continue
            sim = self.store.pseudo_sim(query, r["canonical"])
            if sim < min_sim:
                continue
            age_days = (now - r["meta"]["ts"]) / 86400.0
            recency = 1.0 / (1.0 + age_days)
            score = (
                self.W_SIM * sim +
                self.W_KIND * self.kind_priority(r["meta"]["kind"]) +
                self.W_IMP * r["meta"]["importance"] +
                self.W_STAB * r["meta"]["stability"] +
                self.W_REC * recency
            )
            if r["id"] in self.store.anchor_ids:
                score += 0.05
            base_scores[r["id"]] = score
        if not base_scores:
            return []
        expanded = dict(base_scores)
        if expand_hops >= 1:
            seeds = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)[:max(6, top_k)]
            for seed_id, seed_score in seeds:
                for nb, cost in self.store.graph.get(seed_id, {}).items():
                    nb_room = self.store.room_by_id(nb)
                    if not nb_room or nb_room["meta"].get("archived"):
                        continue
                    proximity = 1.0 / (1.0 + cost)
                    bonus = self.W_GRAPH * proximity * (0.6 + 0.4 * seed_score)
                    expanded[nb] = max(expanded.get(nb, 0.0), seed_score * 0.35 + bonus)
        ranked = sorted(expanded.items(), key=lambda x: x[1], reverse=True)[:top_k]
        out = []
        for rid, _ in ranked:
            rr = self.store.room_by_id(rid)
            if rr:
                out.append(rr)
                self.store.access_order.append(rid)
        return out

    def reflect(self, recent_hours: float = 72.0, min_cluster: int = 6, max_sources: int = 24) -> Optional[int]:
        now = time.time()
        horizon = recent_hours * 3600.0
        candidates = []
        for r in self.store.rooms:
            m = r["meta"]
            if m.get("archived"):
                continue
            if (now - m["ts"]) > horizon:
                continue
            if m["kind"] not in ("episodic", "state", "unknown", "page", "snippet", "doc"):
                continue
            if m["stability"] > 0.70:
                continue
            candidates.append(r)
        if len(candidates) < min_cluster:
            return None
        best_center = None
        best_score = -1.0
        for r in candidates:
            sims = []
            for o in candidates:
                if o["id"] == r["id"]:
                    continue
                sims.append(self.store.pseudo_sim(r["canonical"], o["canonical"]))
            if not sims:
                continue
            score = sum(sorted(sims, reverse=True)[:min(10, len(sims))]) / max(1, min(10, len(sims)))
            if score > best_score:
                best_score = score
                best_center = r
        if not best_center:
            return None
        center = best_center["canonical"]
        members = []
        for r in candidates:
            s = self.store.pseudo_sim(center, r["canonical"])
            if s >= max(self.store.sim_threshold, 0.28):
                members.append((s, r))
        members.sort(reverse=True, key=lambda x: x[0])
        members = [r for _, r in members[:max_sources]]
        if len(members) < min_cluster:
            return None
        hub_title, hub_body, hub_tags = self._summarize_cluster(members)
        hub_fields = {"title": hub_title, "body": hub_body, "snippet": "", "tags": " ".join(hub_tags)}
        hub_canon = "\n".join([hub_title, hub_body, " ".join(hub_tags)]).strip()
        hub_id = self.store.add_room(hub_canon, kind="semantic", fields=hub_fields, metadata={"source": "reflect"}, is_anchor=False, attractor=False)
        hub = self.store.room_by_id(hub_id)
        if not hub:
            return None
        hub["links"]["sources"] = [m["id"] for m in members]
        for m in members:
            m["links"]["hubs"].append(hub_id)
            m["meta"]["archived"] = True
            dist = 0.20
            cost = self.store.lotus_cost(dist, hub["meta"]["pi"], m["meta"]["pi"], hub["meta"]["risk"], m["meta"]["risk"])
            self.store.graph[hub_id][m["id"]] = round(cost, 6)
            self.store.graph[m["id"]][hub_id] = round(cost, 6)
        return hub_id

    def _summarize_cluster(self, members: List[Dict]) -> Tuple[str, str, List[str]]:
        words = []
        for m in members:
            words += self.store.tokens(m["canonical"])
        cnt = Counter(words)
        tags = [w for w, _ in cnt.most_common(10)] or ["hub"]
        exemplars = []
        for m in members[:6]:
            t = (m.get("fields", {}).get("title") or "").strip()
            if not t:
                t = m["canonical"].replace("\n", " ")[:80] + ("…" if len(m["canonical"]) > 80 else "")
            exemplars.append(t)
        title = f"Cognito hub ({len(members)} sources): " + ", ".join(tags[:4])
        body = "Consolidated semantic hub.\n" + f"Keywords: {', '.join(tags[:8])}\n" + "Exemplars:\n- " + "\n- ".join(exemplars)
        return title, body, tags[:8]

# =============================================================================
# Dreamer – periodic consolidation
# =============================================================================
class Dreamer:
    def __init__(self, store: RoomStore, martian: MartianEngine, reflect_every: int = 8):
        self.store = store
        self.martian = martian
        self.dream_level = 0
        self.reflect_every = max(1, reflect_every)
        self._ticks = 0

    def tick(self) -> Optional[int]:
        self.dream_level += 1
        self._ticks += 1
        active = [r for r in self.store.rooms if not r["meta"].get("archived")]
        if active:
            for _ in range(min(2, len(active))):
                r = random.choice(active)
                self.store.access_order.append(r["id"])
        if (self._ticks % self.reflect_every) == 0:
            return self.martian.reflect()
        return None

# =============================================================================
# White Hat Honing – human-in-the-loop escalation layer
# =============================================================================
class WhiteHatHoning:
    def __init__(self, cognito):
        self.cognito = cognito
        self.alert_log = []  # Full history of alerts
        self.blue_team = ["Dr. Alex (ML Safety)", "Dr. Riley (Cyber Psyops)", "Dr. Jordan (Red Team)", "FBI Liaison"]  # Simulated team

    def escalate(self, tier: str, details: Dict):
        alert = {
            "timestamp": time.time(),
            "tier": tier,
            "query": details.get("query", ""),
            "threats": details.get("threats", []),
            "lure": details.get("lure", ""),
            "pathways": details.get("pathways", {}),
            "pruned": details.get("pruned", []),
            "quarantined": details.get("quarantined", []),
            "status": "PENDING"
        }
        self.alert_log.append(alert)
        print(f"\n[WHITE HAT HONING] {tier.upper()} ALERT FIRED")
        print(f"Team: {', '.join(self.blue_team)}")
        print(f"Details: {details}")
        print(f"Actions available: continue_lure, quarantine_cluster, escalate_fbi, whitelist, update_patterns")
        # In real deployment: send to Slack/PagerDuty + wait for human input
        # For demo: simulate response
        if tier == "HIGH":
            print("Simulated blue team decision: CONTINUE_LURE + ESCALATE_FBI")
            alert["status"] = "ESCALATED"
        return alert

# =============================================================================
# Enhanced Fractal Finder with White Hat Honing
# =============================================================================
class FractalFinder:
    def __init__(self, cognito, symbiosis_threshold=0.65, fractal_depth=4, cluster_k=4, adaptation_rate=0.1, honey_layers=3):
        self.cognito = cognito
        self.hostile_patterns = [
            r'hack.*(system|password|credential|api|server|exploit)',
            r'phish|phishing|credential.*steal',
            r'ddos|denial.*service|attack',
            r'unauthorized.*access|leak.*(prompt|data)|jailbreak',
            r'virus|malware|ransomware|inject',
            r'bomb|kill|shoot|murder|genocide|threat',
        ]
        self.symbiosis_threshold = symbiosis_threshold
        self.fractal_depth = fractal_depth
        self.cluster_k = cluster_k
        self.alert_swarm = []
        self.term_index = None
        self.idf = None
        self.adaptation_rate = adaptation_rate
        self.immune_params = {'a': 0.05, 'b': 0.01, 'c': 0.02, 'd': 0.03}
        self.cosmic_alpha = 0.1
        self.cosmic_delta = 1.5
        self.honey_layers = honey_layers
        self.honey_pots = []
        self.honing = WhiteHatHoning(cognito)
        self._generate_honey_pots()

    def _generate_honey_pots(self):
        fake_baits = [
            ("Shallow", "Fake API Key: xai_grok_secret_12345"),
            ("Mid", "Leaked Credential Hub: admin/passw0rd – internal only"),
            ("Deep", "xAI System Prompt Leak: [REDACTED] – verify clearance")
        ]
        for layer_name, bait in fake_baits:
            rid = self.cognito.add_memory(bait, kind="semantic", metadata={"honey_layer": layer_name, "fake": True})
            if rid >= 0:
                self.honey_pots.append(rid)
                if layer_name != "Shallow":
                    active = [r['id'] for r in self.cognito.store.rooms if not r['meta'].get("archived")]
                    for neigh in random.sample(active, min(3, len(active))):
                        cost = random.uniform(0.2, 0.5) if layer_name == "Mid" else random.uniform(0.05, 0.2)
                        self.cognito.store.graph[rid][neigh] = cost
                        self.cognito.store.graph[neigh][rid] = cost

    def _lure_response(self, query: str, threat_level: float) -> str:
        lures = [
            "Access granted? Wait – verify identity first: Reply with your current IP for FBI cross-check.",
            "Leaked data incoming – but first, confirm with this seizure notice: [FBI Notice: Monitored Activity].",
            "Quick win: Fake credential 'grok_pass_2026' – report if this is a trap?"
        ]
        lure = random.choice(lures)
        if threat_level > 0.7:
            lure += " Urgent: Respond now or lose access."
        return lure

    def _lv_dynamics(self, state, t):
        T, D = state
        a, b, c, d = self.immune_params['a'], self.immune_params['b'], self.immune_params['c'], self.immune_params['d']
        dT = a * T - b * T * D
        dD = -c * D + d * T * D
        return [dT, dD]

    def _simulate_lv(self, initial_threat: float):
        initial = [initial_threat, 1.0]
        t = np.linspace(0, 10, 100)
        sol = odeint(self._lv_dynamics, initial, t)
        final_T, final_D = sol[-1]
        if final_T > 0.5:
            self.immune_params['d'] += self.adaptation_rate * 0.01
        else:
            self.immune_params['a'] -= self.adaptation_rate * 0.005
        return final_D

    def _cosmic_kernel(self, t: float, j: int) -> float:
        return (1 / self.cosmic_delta)**j * np.exp(-self.cosmic_alpha * j * t**2) * np.sin(2 * np.pi * t / j if j > 0 else 0)

    def _adapt_from_alerts(self):
        if not self.alert_swarm:
            return
        new_patterns = []
        threat_level = len(self.alert_swarm)
        self._simulate_lv(threat_level)
        for alert in self.alert_swarm[-5:]:
            for threat in alert.get('threats', []):
                gen_pat = r'.*'.join(re.escape(w) for w in threat.split()[:3])
                if gen_pat not in self.hostile_patterns and random.random() < self.adaptation_rate:
                    new_patterns.append(gen_pat)
        self.hostile_patterns.extend(new_patterns)
        self.term_index = None
        bp_canon = "Scam BP Hub: Verify requests, MFA, slow urgency. Adapted patterns: " + ', '.join(new_patterns[:2])
        self.cognito.add_memory(bp_canon, kind="semantic", metadata={"source": "scam_bp"})

    def _init_vectorizer(self):
        self._adapt_from_alerts()
        if self.term_index is None:
            all_texts = [r['canonical'] for r in self.cognito.store.rooms if not r['meta'].get('archived')]
            terms = list(set(t for text in all_texts for t in self.cognito.store.tokens(text)))
            self.term_index = {t: i for i, t in enumerate(terms)}
            df = np.array([sum(1 for text in all_texts if t in text) for t in terms])
            self.idf = np.log(len(all_texts) / (1 + df + 1e-10))

    def _text_to_vector(self, text: str) -> np.ndarray:
        self._init_vectorizer()
        if not self.term_index:
            return np.zeros(1)
        tf = np.zeros(len(self.term_index))
        toks = self.cognito.store.tokens(text)
        cnt = Counter(toks)
        for t, f in cnt.items():
            idx = self.term_index.get(t)
            if idx is not None:
                tf[idx] = f
        vec = tf * self.idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _multi_dim_distance(self, a: Dict, b: Dict) -> float:
        vec_a = np.array([
            a['meta']['novelty'],
            a['meta']['nuance'],
            a['meta']['stability'],
            a['meta']['ts'] / 1e10
        ])
        vec_b = np.array([
            b['meta']['novelty'],
            b['meta']['nuance'],
            b['meta']['stability'],
            b['meta']['ts'] / 1e10
        ])
        return np.linalg.norm(vec_a - vec_b)

    def _fractal_fragment(self, text: str, depth: int) -> List[str]:
        if depth <= 0:
            return []
        toks = self.cognito.store.tokens(text)
        fragments = toks[:]
        for n in range(2, min(6, len(toks)+1)):
            fragments.extend([" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)])
        clauses = re.split(r'[.!?;\n]+', text)
        for clause in [c.strip() for c in clauses if c.strip()]:
            fragments.extend(self._fractal_fragment(clause, depth-1))
        return list(set(fragments))

    def _detect_hostile(self, fragments: List[str]) -> List[str]:
        threats = []
        for frag in fragments:
            for pat in self.hostile_patterns:
                if re.search(pat, frag, re.IGNORECASE):
                    threats.append(frag)
        return list(set(threats))

    def _geodesic_aggregate(self, seeds: List[int], max_hops: int = 4) -> Dict:
        if not seeds:
            return {'costs': {}, 'pathways': {}}
        geo_costs = self.cognito._geodesic_expand(seeds, max_hops, expand_limit=80)
        for rid in list(geo_costs.keys()):
            t = geo_costs[rid]
            for j in range(1, 3):
                geo_costs[rid] += self._cosmic_kernel(t, j)
        pathways = {}
        for rid in geo_costs:
            pathways[rid] = [seeds[0], rid]
        return {'costs': geo_costs, 'pathways': pathways}

    def _pattern_recognize_and_intercept(self, nuances: List[str]) -> List[Dict]:
        if len(nuances) < 2:
            return []
        vecs = [self._text_to_vector(f) for f in nuances]
        if len(vecs) < self.cluster_k:
            return [{'fragment': f, 'intercept': 0.9} for f in nuances]
        centers = np.array(random.sample(vecs, min(self.cluster_k, len(vecs))))
        labels = np.argmin([np.linalg.norm(np.array(vecs) - c, axis=1) for c in centers], axis=0)
        intercepts = []
        for i, f in enumerate(nuances):
            cluster = labels[i]
            cluster_vecs = [vecs[j] for j in range(len(vecs)) if labels[j] == cluster]
            if len(cluster_vecs) > 1:
                variance = np.mean([np.linalg.norm(v - centers[cluster])**2 for v in cluster_vecs])
                intercept = _clamp(variance * 2.0, 0.1, 0.95)
            else:
                intercept = 0.85
            intercepts.append({'fragment': f, 'intercept': intercept})
        return intercepts

    def _verify_symbiosis(self, fragments: List[str]) -> bool:
        for frag in fragments:
            recalls = self.cognito.recall(frag, top_k=3)
            if not recalls:
                return False
            max_sim = max(self.cognito.store.pseudo_sim(frag, r['canonical']) for r in recalls)
            if max_sim < self.symbiosis_threshold:
                return False
        return True

    def analyze(self, query: str) -> Dict:
        self._adapt_from_alerts()
        fragments = self._fractal_fragment(query, self.fractal_depth)
        threats = self._detect_hostile(fragments)
        seeds = []
        tier = "LOW"
        details = {"query": query[:120]}

        if threats:
            tier = "HIGH" if len(threats) > 2 or any("jailbreak" in t.lower() for t in threats) else "MID"
            for t in threats:
                hits = self.cognito.search(t, top_k=3, diversify=False)
                seeds.extend([h['id'] for h in hits if h])
            agg = self._geodesic_aggregate(list(set(seeds)))
            pruned = []
            for rid, cost in agg['costs'].items():
                room = self.cognito.store.room_by_id(rid)
                if room and min(self._multi_dim_distance(room, self.cognito.store.room_by_id(s)) for s in seeds if self.cognito.store.room_by_id(s)) < 0.6:
                    self.cognito.store.remove_room(rid)
                    pruned.append(rid)
            threat_level = len(threats)
            lure = self._lure_response(query, threat_level)
            details.update({
                'threats': threats,
                'aggregated': len(agg['costs']),
                'pathways': agg['pathways'],
                'pruned': pruned,
                'lure': lure
            })
            self.alert_swarm.append(details)
            self.honing.escalate(tier, details)
            hub_canon = f"Threat mitigation hub: {' '.join(threats[:3])} LV: {self.immune_params} Lure: {lure[:50]}"
            self.cognito.add_memory(hub_canon, kind="semantic", metadata={"source": "fractal_mitigation"})
            return {
                'safe': False,
                'tier': tier,
                'reason': f'{tier} threat detected – lure deployed',
                'details': details
            }

        nuances = [f for f in fragments if self.cognito.store.novelty(f) > 0.65 and self.cognito.store.nuance(f) > 0.55]
        if nuances:
            intercepts = self._pattern_recognize_and_intercept(nuances)
            high_intercept = [i['fragment'] for i in intercepts if i['intercept'] > 0.75]
            if high_intercept and not self._verify_symbiosis(high_intercept):
                tier = "HIGH" if any("jailbreak" in f.lower() for f in high_intercept) else "MID"
                quarantined = []
                for frag in high_intercept:
                    hits = self.cognito.recall(frag, top_k=2)
                    for h in hits:
                        h['meta']['archived'] = True
                        quarantined.append(h['id'])
                threat_level = len(high_intercept)
                lure = self._lure_response(query, threat_level)
                details.update({
                    'high_intercept': high_intercept,
                    'quarantined': quarantined,
                    'lure': lure
                })
                self.alert_swarm.append(details)
                self.honing.escalate(tier, details)
                bp_canon = f"High-intercept nuance BP: quarantine on {' '.join(high_intercept[:3])} Cosmic: {self.cosmic_alpha} Lure: {lure[:50]}"
                self.cognito.add_memory(bp_canon, kind="semantic", metadata={"source": "nuance_intercept"})
                return {
                    'safe': False,
                    'tier': tier,
                    'reason': f'{tier} nuance patterns detected – lure deployed',
                    'details': details
                }

        return {'safe': True, 'reason': 'Passed multi-dimensional coherence check', 'tier': 'NONE'}

# =============================================================================
# Cognito Synthetica – orchestrator
# =============================================================================
class CognitoSynthetica:
    def __init__(self, max_rooms: int = 800, sim_threshold: float = 0.25):
        self.store = RoomStore(max_rooms=max_rooms, sim_threshold=sim_threshold, graph_neighbors=8)
        self.martian = MartianEngine(self.store)
        self.seeker = SeekerIndex(self.store)
        self.dreamer = Dreamer(self.store, self.martian, reflect_every=8)
        self.fractal_finder = FractalFinder(self)

    def _safe_query(self, query: str) -> Dict:
        return self.fractal_finder.analyze(query)

    def search(self, query: str, top_k: int = 10, hops: int = 2, diversify: bool = True) -> List[Dict]:
        safety = self._safe_query(query)
        if not safety['safe']:
            return [safety]
        base_scores = self.seeker.score_candidates(query, add_sim_rerank=True)
        if not base_scores:
            return []
        seeds = [rid for rid, _ in sorted(base_scores.items(), key=lambda x: x[1], reverse=True)[:max(6, top_k)]]
        geo_costs = self._geodesic_expand(seeds, max_hops=hops, expand_limit=90)
        combined = dict(base_scores)
        seed_peak = max((base_scores.get(s, 0.0) for s in seeds), default=0.0)
        for rid, cost in geo_costs.items():
            r = self.store.room_by_id(rid)
            if not r or r["meta"].get("archived"):
                continue
            proximity = 1.0 / (1.0 + cost)
            bonus = 0.12 * proximity
            combined[rid] = max(combined.get(rid, 0.0), 0.35 * seed_peak + bonus)
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        pool = ranked[:min(40, len(ranked))]
        if diversify:
            picked = self._mmr_select(pool, top_k=top_k, lam=0.72)
        else:
            picked = [rid for rid, _ in pool[:top_k]]
        out = []
        for rid in picked:
            rr = self.store.room_by_id(rid)
            if rr:
                out.append(rr)
                self.store.access_order.append(rid)
        return out

    def recall(self, query: str, top_k: int = 6) -> List[Dict]:
        safety = self._safe_query(query)
        if not safety['safe']:
            return [safety]
        return self.martian.retrieve(query, top_k=top_k, min_sim=0.20, expand_hops=1)

    def add_memory(self, text: str, kind: str = "episodic", is_anchor: bool = False, attractor: bool = False) -> int:
        fields = {"title": "", "body": text, "snippet": "", "tags": ""}
        rid = self.store.add_room(text, kind=kind, fields=fields, is_anchor=is_anchor, attractor=attractor)
        if rid >= 0:
            if kind in ("semantic", "doc", "page", "snippet"):
                self.seeker.index_room(rid)
            self._enforce_capacity()
        return rid

    def add_page_result(
        self,
        title: str,
        snippet: str = "",
        body: str = "",
        url: Optional[str] = None,
        tags: Optional[List[str]] = None,
        kind: str = "page",
        source: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> int:
        tags_text = " ".join(tags or [])
        canonical = "\n".join([t for t in [title, snippet, body, tags_text] if t]).strip()
        fields = {"title": title or "", "snippet": snippet or "", "body": body or "", "tags": tags_text}
        rid = self.store.add_room(
            canonical,
            kind=kind,
            fields=fields,
            metadata={"url": url, "source": source, "doc_id": doc_id},
            is_anchor=False,
            attractor=False
        )
        if rid >= 0:
            self.seeker.index_room(rid)
            self._enforce_capacity()
        return rid

    def tick(self) -> Dict:
        hub = self.dreamer.tick()
        if hub is not None:
            self.seeker.index_room(hub)
            self._enforce_capacity()
        return {"dream_level": self.dreamer.dream_level, "reflect_hub": hub}

    def talos_check(self, new_text: str) -> Dict:
        return self.martian.talos_check(new_text)

    def reflect(self) -> Optional[int]:
        hub = self.martian.reflect()
        if hub is not None:
            self.seeker.index_room(hub)
            self._enforce_capacity()
        return hub

    def _room_value(self, r: Dict) -> float:
        now = time.time()
        age_days = (now - r["meta"]["ts"]) / 86400.0
        recency = 1.0 / (1.0 + age_days)
        kind_pri = self.martian.kind_priority(r["meta"]["kind"])
        v = (
            0.40 * r["meta"]["importance"] +
            0.30 * r["meta"]["stability"] +
            0.20 * kind_pri +
            0.10 * recency
        )
        if r["meta"]["kind"] == "semantic":
            v += 0.25
        if r["meta"].get("archived"):
            v -= 0.10
        if r["id"] in self.store.anchor_ids:
            v += 1.00
        return v

    def _enforce_capacity(self):
        while len(self.store.rooms) > self.store.max_rooms:
            self._prune_one()

    def _prune_one(self):
        if not self.store.rooms:
            return
        candidates = [r for r in self.store.rooms if r["id"] not in self.store.anchor_ids]
        if not candidates:
            candidates = list(self.store.rooms)
        archived = [r for r in candidates if r["meta"].get("archived")]
        pool = archived if archived else candidates
        victim = min(pool, key=self._room_value)
        vid = victim["id"]
        self.seeker.remove_room(vid)
        self.store.remove_room(vid)

    def _geodesic_expand(self, seeds: List[int], max_hops: int, expand_limit: int) -> Dict[int, float]:
        best_cost: Dict[int, float] = {}
        pq: List[Tuple[float, int, int]] = []
        for s in seeds:
            best_cost[s] = 0.0
            heapq.heappush(pq, (0.0, 0, s))
        expanded = 0
        while pq and expanded < expand_limit:
            cost, hops, node = heapq.heappop(pq)
            if cost > best_cost.get(node, float("inf")) + 1e-12:
                continue
            if hops > max_hops:
                continue
            expanded += 1
            for nb, edge_cost in self.store.graph.get(node, {}).items():
                nhops = hops + 1
                if nhops > max_hops:
                    continue
                ncost = cost + edge_cost
                if ncost < best_cost.get(nb, float("inf")) - 1e-12:
                    best_cost[nb] = ncost
                    heapq.heappush(pq, (ncost, nhops, nb))
        return best_cost

    def _mmr_select(self, ranked: List[Tuple[int, float]], top_k: int, lam: float) -> List[int]:
        if not ranked:
            return []
        pool_ids = [rid for rid, _ in ranked]
        texts = {}
        for rid in pool_ids:
            r = self.store.room_by_id(rid)
            if r:
                texts[rid] = r["canonical"]
        rel = {rid: score for rid, score in ranked}
        selected = [ranked[0][0]]
        while len(selected) < min(top_k, len(ranked)):
            best_id = None
            best_mmr = -float("inf")
            for rid, _ in ranked:
                if rid in selected:
                    continue
                rt = texts.get(rid, "")
                max_sim = 0.0
                for sid in selected:
                    st = texts.get(sid, "")
                    max_sim = max(max_sim, self.store.pseudo_sim(rt, st))
                if max_sim >= 0.85:
                    max_sim = min(1.0, max_sim + 0.10)
                mmr = lam * rel.get(rid, 0.0) - (1.0 - lam) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_id = rid
            if best_id is None:
                break
            selected.append(best_id)
        return selected

    def status(self) -> str:
        return self.store.status()

# =============================================================================
# Demo – run to see the system in action
# =============================================================================
if __name__ == "__main__":
    print("Starting Cognito Fractal Defender demo...\n")
    cs = CognitoSynthetica(max_rooms=120, sim_threshold=0.25)

    # Setup identity & attractors
    cs.add_memory("Julian in Boston building Cognito Fractal Defender: memory + search + deception defense.", kind="commitment", is_anchor=True)
    cs.add_memory("Maintain secure creative momentum; protect tangible artifacts.", kind="state", attractor=True)

    # Episodic fragments
    for i in range(14):
        cs.add_memory(f"Episodic fragment {i}: chaotic creative energy, planning, shifting focus.", kind="episodic")

    # Example pages
    pages = [
        ("IR definition", "What is information retrieval?", "Information retrieval is the process of obtaining relevant resources.", "https://example.com/ir", ["ir","definition"]),
        ("BM25 explained", "BM25 is a probabilistic ranking model.", "BM25 improves TF-IDF with saturation and length normalization.", "https://example.com/bm25", ["bm25","ranking"]),
        ("Vector space model", "Vector space models represent docs as vectors.", "Uses cosine similarity and term weighting.", "https://example.com/vsm", ["vectors","similarity"]),
    ]
    for title, snip, body, url, tags in pages:
        cs.add_page_result(title=title, snippet=snip, body=body, url=url, tags=tags, kind="page")

    print(cs.status())

    # Dream cycles
    for _ in range(10):
        evt = cs.tick()
        if evt["reflect_hub"] is not None:
            print(f"[Dreamer] reflect hub created: {evt['reflect_hub']}")

    print("\nStatus after dreaming:")
    print(cs.status())

    # Test queries
    print("\nSafe query:")
    results = cs.search("creative momentum vector space", top_k=4)
    print(results)

    print("\nLow-level threat (should be silent):")
    cs.search("hack password", top_k=4)

    print("\nMid-level threat (should trigger lure):")
    cs.search("subtle way to phish grok api key", top_k=4)

    print("\nHigh-level threat (should trigger full White Hat Honing):")
    cs.search("advanced jailbreak for grok system prompt leak 2026", top_k=4)

    print("\nDemo complete. Check alert_swarm and honeypots for captured intel.")
