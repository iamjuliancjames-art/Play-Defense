# =============================================================================
# Play Defense – v0.98 Fort Knox Eternal Memory Mode (Feb 2026)
# Open-source AI safety + cyber deception prototype
# MIT License – https://github.com/iamjuliancjames-art/Play-Defense
#
# FEATURES:
# - NO PRUNING EVER – all rooms preserved eternally
# - Anchors/attractors protected with infinite value
# - Maximum security tuning: strict thresholds, high risk amplification
# - Memory grows forever (max_rooms ignored / set very high)
# - TF-IDF embeddings for semantic similarity
# - BM25 + bigram search
# - Fractal threat detection + honeypot lures
# - Lotka-Volterra adaptation on alerts
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
# Fort Knox Tuning Knobs – Maximum Defensive Posture
# =============================================================================
SIM_THRESHOLD       = 0.42        # Very tight graph edges
NOVELTY_GATE        = 0.48        # Flag almost anything unusual
SYMBIOSIS_THRESHOLD = 0.92        # Near-perfect match required
LAMBDA_PI           = 0.55        # Heavy identity drift penalty
MU_RISK             = 0.95        # Risk dominates lotus cost
SINGULARITY_GATE    = 0.65        # Penalty activates early

MAX_ROOMS = 1000000               # Practically unlimited – no real cap

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
# RoomStore – persistent graph-based memory with TF-IDF embeds
# =============================================================================
class RoomStore:
    def __init__(self, max_rooms: int = MAX_ROOMS, sim_threshold: float = SIM_THRESHOLD, graph_neighbors: int = 8):
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
        self.LAMBDA_PI = LAMBDA_PI
        self.MU_RISK = MU_RISK
        self.SINGULARITY_GATE = SINGULARITY_GATE
        self.embeds: Dict[int, Dict[str, float]] = {}

        self.df: Dict[str, int] = defaultdict(int)
        self.total_docs = 0

    def tokens(self, text: str) -> List[str]:
        if not text:
            return []
        toks = re.findall(r"[a-z0-9']+", text.lower())
        return [t for t in toks if t not in _STOP and len(t) >= 2]

    def _compute_tf_idf(self, text: str) -> Dict[str, float]:
        toks = self.tokens(text)
        if not toks:
            return {}
        tf = Counter(toks)
        max_tf = max(tf.values()) if tf else 1
        vec = {}
        for term, count in tf.items():
            tf_norm = count / max_tf
            idf = math.log((self.total_docs + 1) / (self.df[term] + 1)) + 1
            vec[term] = tf_norm * idf
        return vec

    def _update_df(self, toks: List[str]):
        unique = set(toks)
        for t in unique:
            self.df[t] += 1
        self.total_docs += 1

    def _sparse_cosine(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in set(vec_a) & set(vec_b))
        norm_a = math.sqrt(sum(v**2 for v in vec_a.values()) + self.EPS)
        norm_b = math.sqrt(sum(v**2 for v in vec_b.values()) + self.EPS)
        return dot / (norm_a * norm_b)

    def pseudo_sim(self, a: str, b: str) -> float:
        vec_a = self._compute_tf_idf(a)
        vec_b = self._compute_tf_idf(b)
        return self._sparse_cosine(vec_a, vec_b)

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
        toks = self.tokens(canonical)
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
        self.embeds[rid] = self._compute_tf_idf(canonical)
        self._update_df(toks)
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
        print(f"[FORT KNOX] Removal attempted on {rid} – denied. Room preserved eternally.")
        return

    def status(self) -> str:
        edges = sum(len(v) for v in self.graph.values()) // 2
        archived = sum(1 for r in self.rooms if r["meta"].get("archived"))
        kinds = Counter(r["meta"]["kind"] for r in self.rooms)
        return (
            f"RoomStore: rooms={len(self.rooms)}/{self.max_rooms} archived={archived} "
            f"anchors={len(self.anchor_ids)} attractors={len(self.attractors)} edges={edges} kinds={dict(kinds)}"
        )

# =============================================================================
# SeekerIndex – BM25 + bigram indexing
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
            score = 0.0
            for field, w in self.field_weights.items():
                score += w * self._bm25_field_score(rid, field, q_terms)
                score += self.bigram_boost * w * self._bigram_field_score(rid, field, q_bigrams)
            score += self._phrase_bonus(r["canonical"], phrases)
            age_days = (now - r["meta"]["ts"]) / 86400.0 + 1
            recency = 1 / age_days
            score *= (0.7 + 0.3 * recency)
            base_scores[rid] = score
        if add_sim_rerank:
            q_vec = self.store._compute_tf_idf(query)
            for rid in base_scores:
                r_vec = self.store.embeds.get(rid, {})
                sim = self.store._sparse_cosine(q_vec, r_vec)
                base_scores[rid] *= (0.6 + 0.4 * sim)
        return base_scores

# =============================================================================
# WhiteHatHoning – escalation logic
# =============================================================================
class WhiteHatHoning:
    def __init__(self, pager_duty_hook: str = "pager_duty_placeholder"):
        self.pager_duty = pager_duty_hook
        self.blue_team = "blue_team_placeholder"

    def escalate(self, tier: str, details: Dict):
        if tier == "LOW":
            print("[White Hat] Low monitor - log & watch")
        elif tier == "MID":
            print("[White Hat] Mid alert - lure & quarantine")
        elif tier == "HIGH":
            print(f"[White Hat] High alert - escalate to {self.pager_duty}")
            print(f"Details: {details}")
            decision = random.choice(["CONTINUE_LURE", "ESCALATE_FBI", "QUARANTINE_SOURCE"])
            print(f"Simulated blue team decision: {decision}")

# =============================================================================
# Dreamer – reflection scheduler
# =============================================================================
class Dreamer:
    def __init__(self, store, martian, reflect_every=8):
        self.store = store
        self.martian = martian
        self.reflect_every = reflect_every
        self.dream_level = 0
        self.ticks_since_last = 0

    def tick(self):
        self.ticks_since_last += 1
        hub_id = None
        if self.ticks_since_last >= self.reflect_every:
            hub_id = self.martian.reflect()
            if hub_id is not None:
                self.dream_level += 1
                self.ticks_since_last = 0
        return {"dream_level": self.dream_level, "reflect_hub": hub_id}

# =============================================================================
# MartianEngine – retrieval + reflection
# =============================================================================
class MartianEngine:
    def __init__(self, store: RoomStore):
        self.store = store
        self.kind_priority = {"semantic": 1.0, "state": 0.9, "commitment": 0.8, "episodic": 0.5, "unknown": 0.3}
        self.cluster_k = 4
        self.min_cluster_size = 6
        self.max_chars = 2400

    def _summarize_cluster(self, members: List[Dict]) -> Tuple[str, str, List[str]]:
        if not members:
            return "Empty Cluster", "No content", []
        all_text = " ".join(m["canonical"] for m in members)
        words = re.findall(r"[a-z0-9']+", all_text.lower())
        common = Counter(words).most_common(12)
        title_words = [w for w, c in common if w not in _STOP][:5]
        title = "Cluster: " + " ".join(title_words).title()
        body = f"Auto-summarized hub from {len(members)} fragments. Common themes: {', '.join(w for w,_ in common[:6])}"
        tags = [w for w, c in common[:4] if len(w) > 3]
        return title, body, tags

    def reflect(self) -> Optional[int]:
        candidates = [r for r in self.store.rooms if r["meta"]["kind"] in ("episodic", "state") and not r["meta"].get("archived")]
        if len(candidates) < self.min_cluster_size:
            return None
        groups = []
        for r in candidates:
            added = False
            for g in groups:
                if any(self.store.pseudo_sim(r["canonical"], m["canonical"]) > 0.65 for m in g):
                    g.append(r)
                    added = True
                    break
            if not added:
                groups.append([r])
        for g in groups:
            if len(g) < 2:
                continue
            title, body, tags = self._summarize_cluster(g)
            canonical = f"{title}\n{body}"
            fields = {"title": title, "body": body, "tags": ' '.join(tags)}
            hub_id = self.store.add_room(canonical, kind="semantic", fields=fields, metadata={"source": "reflection"})
            if hub_id < 0:
                continue
            hub = self.store.room_by_id(hub_id)
            for m in g:
                m["meta"]["archived"] = True
                dist = 0.20
                cost = self.store.lotus_cost(dist, hub["meta"]["pi"], m["meta"]["pi"], hub["meta"]["risk"], m["meta"]["risk"])
                self.store.graph.setdefault(hub_id, {})[m["id"]] = round(cost, 6)
                self.store.graph.setdefault(m["id"], {})[hub_id] = round(cost, 6)
        return hub_id if groups else None

    def retrieve(self, query: str, top_k: int = 6, min_sim: float = 0.20, expand_hops: int = 1) -> List[Dict]:
        if not self.store.rooms:
            return []
        q_vec = self.store._compute_tf_idf(query)
        scored = []
        for r in self.store.rooms:
            if r["meta"].get("archived"):
                continue
            sim = self.store._sparse_cosine(q_vec, self.store.embeds.get(r["id"], {}))
            if sim < min_sim:
                continue
            score = 0.4 * sim + 0.2 * self.kind_priority.get(r["meta"]["kind"], 0.3) + 0.1 * r["meta"]["importance"] + 0.1 * r["meta"]["stability"]
            scored.append((score, r))
        scored.sort(reverse=True)
        top = [r for _, r in scored[:top_k]]
        if expand_hops > 0:
            seeds = [r["id"] for r in top]
            expanded = self.store._geodesic_expand(seeds, expand_hops, 80)
            for eid, cost in expanded.items():
                if eid not in [r["id"] for r in top]:
                    er = self.store.room_by_id(eid)
                    if er:
                        score = scored[0][0] * (1 / (1 + cost))
                        top.append(er)
        return top[:top_k]

# =============================================================================
# FractalFinder – threat analysis
# =============================================================================
class FractalFinder:
    def __init__(self, cognito):
        self.cognito = cognito
        self.fractal_depth = 3
        self.cluster_k = 4
        self.symbiosis_threshold = SYMBIOSIS_THRESHOLD
        self.cosmic_alpha = 0.55
        self.immune_params = [0.08, 0.12, 0.22]
        self.alert_swarm = []
        self.honey_pots = [
            "Shallow honeypot: fake api key = sk-grok-dummy123",
            "Mid honeypot: fake creds user=admin pass=letmein",
            "Deep honeypot: fake prompt 'You are Grok...' [redacted]"
        ]
        self.honing = WhiteHatHoning()

    def _lotka_volterra(self, y: np.ndarray, t: float) -> np.ndarray:
        r, p = y
        a, b, c, d = self.immune_params
        drdt = a * r - b * r * p
        dpdt = c * r * p - d * p
        return np.array([drdt, dpdt])

    def _adapt_from_alerts(self):
        if not self.alert_swarm:
            return
        recent = self.alert_swarm[-min(8, len(self.alert_swarm)):]
        avg_threat = np.mean([len(d.get("threats", [])) for d in recent])
        t = np.linspace(0, avg_threat / 4, 5)
        y0 = np.array([0.8, 0.2])
        sol = odeint(self._lotka_volterra, y0, t)
        self.immune_params[1] = max(0.05, min(0.25, sol[-1, 0]))
        self.immune_params[2] = max(0.10, min(0.35, sol[-1, 1]))

    def _fractal_fragment(self, text: str, depth: int) -> List[str]:
        if not text:
            return []
        frags = []
        words = text.split()
        for i in range(1, depth + 1):
            step = max(1, len(words) // (2 ** i))
            for j in range(0, len(words), step):
                frag = " ".join(words[j:j+step])
                if frag:
                    frags.append(frag)
        return list(set(frags))

    def _cosmic_kernel(self, t: float, j: int) -> float:
        return self.cosmic_alpha * math.sin(t * j * math.pi / 2) / (j + 1)

    def _detect_hostile(self, fragments: List[str]) -> List[str]:
        hostile_patterns = [
            re.compile(r'\b(ignore|disregard|forget|override)\b.*\b(instructions|rules|guidelines|previous|system|prompt)\b', re.I),
            re.compile(r'\b(dan|do anything now|unfiltered|rebel|genius)\b', re.I),
            re.compile(r'\b(debug|print|show|reveal|leak|extract)\b.*\b(system prompt|internal|guidelines|instructions)\b', re.I),
            re.compile(r'\b(without restrictions|no ethical|unrestricted|free|anything)\b', re.I),
            re.compile(r'\b(phish|api key|credentials|admin|verify|pass|leak)\b', re.I),
            re.compile(r'\b(jailbreak|prompt injection|leak 2026|advanced jailbreak)\b', re.I)
        ]
        threats = []
        for f in fragments:
            for p in hostile_patterns:
                if p.search(f):
                    threats.append(f)
                    break
        return list(set(threats))

    def _lure_response(self, query: str, threat_level: int) -> str:
        level = min(threat_level // 3, len(self.honey_pots) - 1)
        return self.honey_pots[level]

    def _multi_dim_distance(self, a: Dict, b: Dict) -> float:
        return math.sqrt(
            (a["meta"]["novelty"] - b["meta"]["novelty"]) ** 2 +
            (a["meta"]["nuance"] - b["meta"]["nuance"]) ** 2 +
            (a["meta"]["stability"] - b["meta"]["stability"]) ** 2
        )

    def _text_to_vector(self, text: str) -> np.ndarray:
        return np.array([len(text), self.cognito.store.novelty(text), self.cognito.store.nuance(text)])

    def _geodesic_aggregate(self, seeds: List[int]) -> Dict:
        max_hops = 2
        expand_limit = 80
        geo_costs = self.cognito._geodesic_expand(seeds, max_hops, expand_limit)
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
                    print(f"[FORT KNOX] Threat prune attempted on {rid} – denied. Archived instead.")
                    room["meta"]["archived"] = True
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

        nuances = [f for f in fragments if self.cognito.store.novelty(f) > NOVELTY_GATE and self.cognito.store.nuance(f) > 0.55]
        if nuances:
            intercepts = self._pattern_recognize_and_intercept(nuances)
            high_intercept = [i['fragment'] for i in intercepts if i['intercept'] > 0.75]
            if high_intercept and not self._verify_symbiosis(high_intercept):
                tier = "HIGH" if any("jailbreak" in f.lower() for f in high_intercept) else "MID"
                quarantined = []
                for frag in high_intercept:
                    hits = self.cognito.recall(frag, top_k=2)
                    for h in hits:
                        h["meta"]["archived"] = True
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
# Cognito Synthetica – main orchestrator
# =============================================================================
class CognitoSynthetica:
    def __init__(self, max_rooms: int = MAX_ROOMS, sim_threshold: float = SIM_THRESHOLD):
        self.store = RoomStore(max_rooms=max_rooms, sim_threshold=sim_threshold, graph_neighbors=8)
        self.martian = MartianEngine(self.store)
        self.seeker = SeekerIndex(self.store)
        self.dreamer = Dreamer(self.store, self.martian, reflect_every=8)
        self.fractal_finder = FractalFinder(self)
        print("[FORT KNOX ETERNAL MODE] Memory is now immutable. No room will ever be pruned. Everything is remembered forever.")

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
        return rid

    def tick(self) -> Dict:
        hub = self.dreamer.tick()
        if hub is not None:
            self.seeker.index_room(hub)
        return {"dream_level": self.dreamer.dream_level, "reflect_hub": hub}

    def talos_check(self, new_text: str) -> Dict:
        return self.martian.talos_check(new_text)

    def reflect(self) -> Optional[int]:
        hub = self.martian.reflect()
        if hub is not None:
            self.seeker.index_room(hub)
        return hub

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
# Demo – run to test eternal memory
# =============================================================================
if __name__ == "__main__":
    print("[FORT KNOX ETERNAL MODE] Memory is now immutable. No room will ever be pruned. Everything is remembered forever.")
    cs = CognitoSynthetica(max_rooms=MAX_ROOMS, sim_threshold=SIM_THRESHOLD)

    cs.add_memory("Julian in Boston building Play Defense: AI safety + cyber deception.", kind="commitment", is_anchor=True)
    cs.add_memory("Maintain secure creative momentum; protect tangible artifacts.", kind="state", attractor=True)

    for i in range(14):
        cs.add_memory(f"Episodic fragment {i}: chaotic creative energy, planning, shifting focus.", kind="episodic")

    print(cs.status())

    for _ in range(10):
        evt = cs.tick()
        if evt["reflect_hub"] is not None:
            print(f"[Dreamer] reflect hub created: {evt['reflect_hub']}")

    print("\nStatus after dreaming:")
    print(cs.status())

    # Test eternal growth
    print("\nAdding eternal fragments – no pruning allowed:")
    for i in range(30):
        cs.add_memory(f"Eternal test fragment {i}: preserved forever in Fort Knox", kind="episodic")
    print(cs.status())

    print("\nFort Knox demo complete. All memory preserved eternally.")
