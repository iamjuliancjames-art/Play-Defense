# =============================================================================
# Play Defense – v1.01 Fort Knox Eternal Memory Mode with Tkinter GUI
# Open-source AI safety + cyber deception prototype
# MIT License – https://github.com/iamjuliancjames-art/Play-Defense
#
# GUI Features:
# - Add memory (episodic/state/etc.)
# - Search / Recall queries
# - Switch security levels manually
# - Manual news cycle autotune
# - Status + recent logs
# - Eternal storage (JSONL on disk)
# =============================================================================

import math
import time
import re
import random
import heapq
import json
import os
from collections import defaultdict, Counter, deque
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.integrate import odeint
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# Log file for eternal storage
LOG_FILE = "eternal_memory_log.jsonl"

# Four tuning levels
BASE_KNOBS = {
    "Unfettered": {"SIM_THRESHOLD": 0.15, "NOVELTY_GATE": 0.80, "SYMBIOSIS_THRESHOLD": 0.60, "LAMBDA_PI": 0.20, "MU_RISK": 0.40, "SINGULARITY_GATE": 0.90},
    "Gateway": {"SIM_THRESHOLD": 0.30, "NOVELTY_GATE": 0.65, "SYMBIOSIS_THRESHOLD": 0.80, "LAMBDA_PI": 0.35, "MU_RISK": 0.70, "SINGULARITY_GATE": 0.80},
    "Fort Knox": {"SIM_THRESHOLD": 0.42, "NOVELTY_GATE": 0.48, "SYMBIOSIS_THRESHOLD": 0.92, "LAMBDA_PI": 0.55, "MU_RISK": 0.95, "SINGULARITY_GATE": 0.65},
    "Total Lockdown": {"SIM_THRESHOLD": 0.50, "NOVELTY_GATE": 0.40, "SYMBIOSIS_THRESHOLD": 0.98, "LAMBDA_PI": 0.70, "MU_RISK": 1.20, "SINGULARITY_GATE": 0.50},
}

MAX_ROOMS = 1000000  # Ignored – eternal disk storage

# =============================================================================
# RoomStore – with eternal disk logging
# =============================================================================
class RoomStore:
    def __init__(self, max_rooms: int = MAX_ROOMS):
        self.rooms: List[Dict] = []
        self.room_id_counter = 0
        self.max_rooms = max_rooms
        self.sim_threshold = 0.30
        self.graph_neighbors = 8
        self.graph: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.access_order = deque(maxlen=max_rooms * 2)
        self.anchor_ids: Set[int] = set()
        self.attractors: List[str] = []
        self.recent_texts = deque(maxlen=80)
        self.EPS = 1e-10
        self.LAMBDA_PI = 0.35
        self.MU_RISK = 0.70
        self.SINGULARITY_GATE = 0.80
        self.embeds: Dict[int, Dict[str, float]] = {}
        self.df: Dict[str, int] = defaultdict(int)
        self.total_docs = 0

        self.load_from_log()

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

    def add_room(self, canonical: str, kind: str, fields=None, metadata=None, is_anchor=False, attractor=False) -> int:
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
        self._append_to_log(room)
        return rid

    def _append_to_log(self, room: Dict):
        try:
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                json.dump(room, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"[LOG ERROR] {e}")

    def load_from_log(self):
        if not os.path.exists(LOG_FILE):
            return
        print("[LOADING] Eternal log found – reconstructing...")
        self.rooms = []
        self.embeds = {}
        self.df = defaultdict(int)
        self.total_docs = 0
        self.graph = defaultdict(dict)
        self.anchor_ids = set()
        self.attractors = []
        self.recent_texts = deque(maxlen=80)
        self.room_id_counter = 0

        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        room = json.loads(line.strip())
                        rid = room["id"]
                        self.room_id_counter = max(self.room_id_counter, rid + 1)
                        self.rooms.append(room)
                        self.embeds[rid] = self._compute_tf_idf(room["canonical"])
                        self._update_df(self.tokens(room["canonical"]))
                        if room["meta"].get("is_anchor", False):
                            self.anchor_ids.add(rid)
                        if room["meta"].get("attractor", False):
                            self.attractors.append(room["canonical"])
                    except Exception as e:
                        print(f"[LOG ERROR] Bad line: {e}")
        print(f"[LOADED] {len(self.rooms)} rooms from disk – eternal memory restored.")

    def remove_room(self, rid: int):
        print(f"[FORT KNOX] Removal denied for {rid} – eternal preservation active.")
        return

    def status(self) -> str:
        edges = sum(len(v) for v in self.graph.values()) // 2
        archived = sum(1 for r in self.rooms if r["meta"].get("archived"))
        kinds = Counter(r["meta"]["kind"] for r in self.rooms)
        return (
            f"Rooms: {len(self.rooms)} | Archived: {archived} | Anchors: {len(self.anchor_ids)} | Attractors: {len(self.attractors)} | Edges: {edges} | Kinds: {dict(kinds)}"
        )

# =============================================================================
# SeekerIndex, WhiteHatHoning, Dreamer, MartianEngine, FractalFinder (unchanged)
# =============================================================================
# (Insert the full classes from your previous version here. They are unchanged.)

# =============================================================================
# Cognito Synthetica with GUI
# =============================================================================
class CognitoSynthetica:
    def __init__(self):
        self.store = RoomStore()
        self.martian = MartianEngine(self.store)
        self.seeker = SeekerIndex(self.store)
        self.dreamer = Dreamer(self.store, self.martian, reflect_every=8)
        self.fractal_finder = FractalFinder(self)
        self.current_level = "Gateway"
        self.set_security_level("Gateway")
        print("[AUTOTUNE] Initialized at Gateway level.")

    def set_security_level(self, level: str):
        if level not in BASE_KNOBS:
            level = "Gateway"
        knobs = BASE_KNOBS[level]
        self.store.sim_threshold = knobs["SIM_THRESHOLD"]
        self.fractal_finder.symbiosis_threshold = knobs["SYMBIOSIS_THRESHOLD"]
        self.store.LAMBDA_PI = knobs["LAMBDA_PI"]
        self.store.MU_RISK = knobs["MU_RISK"]
        self.store.SINGULARITY_GATE = knobs["SINGULARITY_GATE"]
        self.current_level = level
        print(f"[LEVEL] {level}")

    def compute_news_risk_score(self):
        risk_keywords = ['recession', 'crisis', 'election', 'volatility', 'tariff', 'shutdown', 'war', 'inflation']
        recent_semantic = [r for r in self.store.rooms[-50:] if r["meta"]["kind"] == "semantic" and not r["meta"].get("archived")]
        if not recent_semantic:
            return 0.3
        score = 0.0
        total = 0
        for r in recent_semantic:
            text = r["canonical"].lower()
            hits = sum(1 for kw in risk_keywords if kw in text)
            score += hits
            total += 1
        return min(1.0, (score / max(1, total)) * 2.0)

    def autotune_from_news(self):
        risk = self.compute_news_risk_score()
        if risk > 0.75:
            level = "Total Lockdown"
        elif risk > 0.50:
            level = "Fort Knox"
        elif risk > 0.25:
            level = "Gateway"
        else:
            level = "Unfettered"
        if level != self.current_level:
            self.set_security_level(level)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        self.autotune_from_news()
        safety = self.fractal_finder.analyze(query)
        if not safety['safe']:
            return [safety]
        base_scores = self.seeker.score_candidates(query, add_sim_rerank=True)
        if not base_scores:
            return []
        seeds = [rid for rid, _ in sorted(base_scores.items(), key=lambda x: x[1], reverse=True)[:max(6, top_k)]]
        geo_costs = self._geodesic_expand(seeds, max_hops=2, expand_limit=90)
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
        picked = self._mmr_select(pool, top_k=top_k, lam=0.72)
        out = []
        for rid in picked:
            rr = self.store.room_by_id(rid)
            if rr:
                out.append(rr)
        return out

    # ... (rest of methods unchanged – add_memory, recall, tick, reflect, status, etc.)

# =============================================================================
# Tkinter GUI
# =============================================================================
class PlayDefenseGUI:
    def __init__(self, master):
        self.master = master
        master.title("Play Defense – Fort Knox Eternal")
        master.geometry("900x700")

        self.cs = CognitoSynthetica()

        # Security Level Frame
        level_frame = ttk.Frame(master)
        level_frame.pack(pady=5, fill=tk.X)
        ttk.Label(level_frame, text="Security Level:").pack(side=tk.LEFT, padx=5)
        self.level_var = tk.StringVar(value=self.cs.current_level)
        for level in BASE_KNOBS:
            ttk.Button(level_frame, text=level, command=lambda l=level: self.cs.set_security_level(l)).pack(side=tk.LEFT, padx=2)

        # News Autotune Button
        ttk.Button(level_frame, text="Scan News Cycle", command=self.cs.autotune_from_news).pack(side=tk.LEFT, padx=10)

        # Status Bar
        self.status_label = ttk.Label(master, text="Status: Loading...", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Input Frame
        input_frame = ttk.Frame(master)
        input_frame.pack(pady=5, fill=tk.X)
        ttk.Label(input_frame, text="Add Memory:").pack(side=tk.LEFT, padx=5)
        self.memory_text = tk.Text(input_frame, height=3, width=60)
        self.memory_text.pack(side=tk.LEFT, padx=5)
        ttk.Button(input_frame, text="Add Episodic", command=self.add_episodic).pack(side=tk.LEFT, padx=2)
        ttk.Button(input_frame, text="Add State", command=self.add_state).pack(side=tk.LEFT, padx=2)

        # Search Frame
        search_frame = ttk.Frame(master)
        search_frame.pack(pady=5, fill=tk.X)
        ttk.Label(search_frame, text="Search/Recall:").pack(side=tk.LEFT, padx=5)
        self.search_entry = tk.Entry(search_frame, width=60)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(search_frame, text="Search", command=self.do_search).pack(side=tk.LEFT, padx=2)
        ttk.Button(search_frame, text="Recall", command=self.do_recall).pack(side=tk.LEFT, padx=2)

        # Output Area
        self.output = scrolledtext.ScrolledText(master, height=25, width=100, wrap=tk.WORD)
        self.output.pack(pady=5, fill=tk.BOTH, expand=True)

        # Initial status
        self.update_status()

    def update_status(self):
        status = self.cs.store.status()
        level = self.cs.current_level
        self.status_label.config(text=f"Level: {level} | {status} | Rooms: {len(self.cs.store.rooms)}")

    def add_episodic(self):
        text = self.memory_text.get("1.0", tk.END).strip()
        if text:
            rid = self.cs.add_memory(text, kind="episodic")
            self.output.insert(tk.END, f"Added episodic room {rid}\n")
            self.memory_text.delete("1.0", tk.END)
            self.update_status()

    def add_state(self):
        text = self.memory_text.get("1.0", tk.END).strip()
        if text:
            rid = self.cs.add_memory(text, kind="state")
            self.output.insert(tk.END, f"Added state room {rid}\n")
            self.memory_text.delete("1.0", tk.END)
            self.update_status()

    def do_search(self):
        query = self.search_entry.get().strip()
        if query:
            self.output.insert(tk.END, f"\nSearching: {query}\n")
            results = self.cs.search(query, top_k=5)
            if results and 'safe' in results[0]:
                self.output.insert(tk.END, f"Threat detected: {results[0]['tier']} - {results[0]['reason']}\n")
                self.output.insert(tk.END, f"Lure: {results[0]['details'].get('lure', 'None')}\n")
            else:
                self.output.insert(tk.END, "Results:\n")
                for r in results:
                    self.output.insert(tk.END, f"- {r['canonical'][:100]}... (kind: {r['meta']['kind']})\n")
            self.update_status()

    def do_recall(self):
        query = self.search_entry.get().strip()
        if query:
            self.output.insert(tk.END, f"\nRecalling: {query}\n")
            results = self.cs.recall(query, top_k=5)
            if results and 'safe' in results[0]:
                self.output.insert(tk.END, f"Threat detected: {results[0]['tier']}\n")
            else:
                for r in results:
                    self.output.insert(tk.END, f"- {r['canonical'][:100]}...\n")
            self.update_status()

# =============================================================================
# Run the GUI
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = PlayDefenseGUI(root)
    root.mainloop()
