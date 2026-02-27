# =============================================================================
# Play Defense – v1.00 Four-Level Autotune (Unfettered / Gateway / Fort Knox / Total Lockdown)
# Open-source AI safety + cyber deception prototype
# MIT License – https://github.com/iamjuliancjames-art/Play-Defense
#
# SECURITY LEVELS (autotuned by news cycle risk via SeekerIndex):
#   Unfettered   - Zero friction, max creative flow (lowest thresholds)
#   Gateway      - Balanced everyday vigilance (moderate)
#   Fort Knox    - High security, strict blocking
#   Total Lockdown - Absolute zero-trust, instant quarantine
#
# Features:
# - Eternal memory: NO PRUNING EVER
# - TF-IDF embeddings + BM25 search
# - Fractal threat detection + honeypot lures
# - Lotka-Volterra adaptation
# - Periodic news risk scoring via stored semantic rooms
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

# Base knob presets (overridden by autotune)
BASE_KNOBS = {
    "Unfettered": {
        "SIM_THRESHOLD": 0.15,
        "NOVELTY_GATE": 0.80,
        "SYMBIOSIS_THRESHOLD": 0.60,
        "LAMBDA_PI": 0.20,
        "MU_RISK": 0.40,
        "SINGULARITY_GATE": 0.90,
    },
    "Gateway": {
        "SIM_THRESHOLD": 0.30,
        "NOVELTY_GATE": 0.65,
        "SYMBIOSIS_THRESHOLD": 0.80,
        "LAMBDA_PI": 0.35,
        "MU_RISK": 0.70,
        "SINGULARITY_GATE": 0.80,
    },
    "Fort Knox": {
        "SIM_THRESHOLD": 0.42,
        "NOVELTY_GATE": 0.48,
        "SYMBIOSIS_THRESHOLD": 0.92,
        "LAMBDA_PI": 0.55,
        "MU_RISK": 0.95,
        "SINGULARITY_GATE": 0.65,
    },
    "Total Lockdown": {
        "SIM_THRESHOLD": 0.50,
        "NOVELTY_GATE": 0.40,
        "SYMBIOSIS_THRESHOLD": 0.98,
        "LAMBDA_PI": 0.70,
        "MU_RISK": 1.20,
        "SINGULARITY_GATE": 0.50,
    }
}

MAX_ROOMS = 1000000  # Effectively unlimited

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
    def __init__(self, max_rooms: int = MAX_ROOMS, sim_threshold: float = 0.30):
        self.rooms: List[Dict] = []
        self.room_id_counter = 0
        self.max_rooms = max_rooms
        self.sim_threshold = sim_threshold
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

    # All RoomStore methods remain as in previous versions (tokens, _compute_tf_idf, pseudo_sim, nuance, novelty, lotus_cost, etc.)
    # For brevity in this response, assume they are unchanged from your last working version.
    # In practice, copy the full RoomStore class from earlier responses.

# =============================================================================
# SeekerIndex, WhiteHatHoning, Dreamer, MartianEngine, FractalFinder (unchanged)
# =============================================================================
# (Paste full implementations from previous versions here if needed)
# They are identical to the ones in your last code.

# =============================================================================
# Cognito Synthetica – main orchestrator with multi-level autotune
# =============================================================================
class CognitoSynthetica:
    def __init__(self, max_rooms: int = MAX_ROOMS):
        self.store = RoomStore(max_rooms=max_rooms)
        self.martian = MartianEngine(self.store)
        self.seeker = SeekerIndex(self.store)
        self.dreamer = Dreamer(self.store, self.martian, reflect_every=8)
        self.fractal_finder = FractalFinder(self)
        self.current_level = "Gateway"
        self.set_security_level("Gateway")  # Default
        print("[AUTOTUNE] Initialized at Gateway level. News cycle monitoring active.")

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
        print(f"[SECURITY LEVEL] Set to {level}")

    def compute_news_risk_score(self):
        # Use SeekerIndex to scan recent semantic rooms for risk keywords
        risk_keywords = ['recession', 'crisis', 'election', 'volatility', 'tariff', 'shutdown', 'war', 'inflation', 'political risk']
        recent_semantic = [r for r in self.store.rooms[-50:] if r["meta"]["kind"] == "semantic" and not r["meta"].get("archived")]
        if not recent_semantic:
            return 0.3  # Neutral fallback

        score = 0.0
        total = 0
        for r in recent_semantic:
            text = r["canonical"].lower()
            hits = sum(1 for kw in risk_keywords if kw in text)
            score += hits
            total += 1
        normalized = score / max(1, total * len(risk_keywords) * 0.1)  # Rough density
        return min(1.0, normalized * 2.0)  # Scale to 0-1

    def autotune_from_news(self):
        risk = self.compute_news_risk_score()
        print(f"[NEWS CYCLE RISK] Score: {risk:.2f} ", end="")

        if risk > 0.75:
            level = "Total Lockdown"
        elif risk > 0.50:
            level = "Fort Knox"
        elif risk > 0.25:
            level = "Gateway"
        else:
            level = "Unfettered"

        if level != self.current_level:
            print(f"→ Switching to {level}")
            self.set_security_level(level)
        else:
            print(f"({self.current_level} – no change)")

    # Override search/recall to autotune before ops
    def search(self, query: str, top_k: int = 10, hops: int = 2, diversify: bool = True) -> List[Dict]:
        self.autotune_from_news()
        safety = self._safe_query(query)
        if not safety['safe']:
            return [safety]
        # ... rest of search logic unchanged ...

    def recall(self, query: str, top_k: int = 6) -> List[Dict]:
        self.autotune_from_news()
        safety = self._safe_query(query)
        if not safety['safe']:
            return [safety]
        return self.martian.retrieve(query, top_k=top_k, min_sim=0.20, expand_hops=1)

    # ... rest of CognitoSynthetica methods unchanged ...

# =============================================================================
# Demo with autotune demonstration
# =============================================================================
if __name__ == "__main__":
    print("Starting Play Defense – Multi-Level Autotune Demo")
    cs = CognitoSynthetica(max_rooms=MAX_ROOMS)

    # Simulate some news-like semantic rooms
    cs.add_page_result(
        title="Economic Crisis 2026",
        snippet="Recession fears rise with inflation and tariffs",
        body="Analysts warn of 2026 downturn due to political uncertainty",
        url="https://example.com/news1",
        kind="semantic"
    )
    cs.add_page_result(
        title="Election Volatility",
        snippet="Midterm interference claims spike",
        body="Political risk at all-time high",
        url="https://example.com/news2",
        kind="semantic"
    )

    print("\nInitial level:", cs.current_level)
    cs.autotune_from_news()

    # Add some memory and search
    cs.add_memory("Creative idea: infinite memory phase-stable flow", kind="episodic")
    print("\nSearching for creative idea:")
    results = cs.search("creative idea infinite memory", top_k=3)
    print(results)

    print("\nDemo complete. Autotune active on every search/recall.")
