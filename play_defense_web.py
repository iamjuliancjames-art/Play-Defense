# =============================================================================
# Play Defense – v1.03 Local Web App (Flask + Browser UI) – Flash Fixed
# Runs locally on http://127.0.0.1:5000, auto-opens in Chrome
# MIT License – https://github.com/iamjuliancjames-art/Play-Defense
# =============================================================================

from flask import Flask, request, render_template_string, jsonify, flash, session
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
import webbrowser
import threading

app = Flask(__name__)
app.secret_key = 'super-secret-fort-knox-key-change-me'  # Required for flash/session

# Log file
LOG_FILE = "eternal_memory_log.jsonl"

# Four tuning levels
BASE_KNOBS = {
    "Unfettered": {"SIM_THRESHOLD": 0.15, "NOVELTY_GATE": 0.80, "SYMBIOSIS_THRESHOLD": 0.60, "LAMBDA_PI": 0.20, "MU_RISK": 0.40, "SINGULARITY_GATE": 0.90},
    "Gateway": {"SIM_THRESHOLD": 0.30, "NOVELTY_GATE": 0.65, "SYMBIOSIS_THRESHOLD": 0.80, "LAMBDA_PI": 0.35, "MU_RISK": 0.70, "SINGULARITY_GATE": 0.80},
    "Fort Knox": {"SIM_THRESHOLD": 0.42, "NOVELTY_GATE": 0.48, "SYMBIOSIS_THRESHOLD": 0.92, "LAMBDA_PI": 0.55, "MU_RISK": 0.95, "SINGULARITY_GATE": 0.65},
    "Total Lockdown": {"SIM_THRESHOLD": 0.50, "NOVELTY_GATE": 0.40, "SYMBIOSIS_THRESHOLD": 0.98, "LAMBDA_PI": 0.70, "MU_RISK": 1.20, "SINGULARITY_GATE": 0.50},
}

MAX_ROOMS = 1000000

# =============================================================================
# Shared Utilities (unchanged)
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
# RoomStore – eternal, disk-backed
# =============================================================================
class RoomStore:
    def __init__(self, max_rooms: int = MAX_ROOMS):
        self.rooms: List[Dict] = []
        self.room_id_counter = 0
        self.max_rooms = max_rooms
        self.sim_threshold = 0.30
        self.graph_neighbors = 8
        self.graph: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.access_order = deque()
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
        print(f"[FORT KNOX] Removal denied for {rid} – eternal preservation active.")
        return

    def status(self) -> str:
        edges = sum(len(v) for v in self.graph.values()) // 2
        archived = sum(1 for r in self.rooms if r["meta"].get("archived"))
        kinds = Counter(r["meta"]["kind"] for r in self.rooms)
        return (
            f"Rooms: {len(self.rooms)} | Archived: {archived} | Anchors: {len(self.anchor_ids)} | Attractors: {len(self.attractors)} | Edges: {edges} | Kinds: {dict(kinds)}"
        )

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

# =============================================================================
# Simplified classes for demo
# =============================================================================
class SeekerIndex:
    def __init__(self, store):
        self.store = store

    def score_candidates(self, query, add_sim_rerank=True):
        return {r["id"]: random.random() for r in self.store.rooms[:5]}

class WhiteHatHoning:
    def escalate(self, tier, details):
        print(f"[White Hat] {tier} alert")

class Dreamer:
    def __init__(self, store, martian, reflect_every=8):
        self.store = store
        self.martian = martian

    def tick(self):
        return {"reflect_hub": None}

class MartianEngine:
    def __init__(self, store):
        self.store = store

    def reflect(self):
        return None

    def retrieve(self, query, top_k=6, min_sim=0.20, expand_hops=1):
        return self.store.rooms[:top_k]

class FractalFinder:
    def __init__(self, cognito):
        self.cognito = cognito
        self.alert_swarm = []

    def analyze(self, query):
        return {'safe': True, 'reason': 'Passed', 'tier': 'NONE'}

# =============================================================================
# Cognito Synthetica – main orchestrator
# =============================================================================
class CognitoSynthetica:
    def __init__(self, max_rooms: int = MAX_ROOMS):
        self.store = RoomStore(max_rooms=max_rooms)
        self.martian = MartianEngine(self.store)
        self.seeker = SeekerIndex(self.store)
        self.dreamer = Dreamer(self.store, self.martian, reflect_every=8)
        self.fractal_finder = FractalFinder(self)
        self.current_level = "Gateway"
        self.set_security_level("Gateway")

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

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        safety = self.fractal_finder.analyze(query)
        if not safety['safe']:
            return [safety]
        return self.martian.retrieve(query, top_k=top_k)

    def recall(self, query: str, top_k: int = 5) -> List[Dict]:
        safety = self.fractal_finder.analyze(query)
        if not safety['safe']:
            return [safety]
        return self.martian.retrieve(query, top_k=top_k)

    def add_memory(self, text: str, kind: str = "episodic", is_anchor: bool = False, attractor: bool = False) -> int:
        fields = {"title": "", "body": text, "snippet": "", "tags": ""}
        rid = self.store.add_room(text, kind=kind, fields=fields, is_anchor=is_anchor, attractor=attractor)
        if rid >= 0:
            self.seeker.index_room(rid)
        return rid

    def status(self) -> str:
        return self.store.status()

# =============================================================================
# Flask Web App
# =============================================================================
app = Flask(__name__)
app.secret_key = 'super-secret-fort-knox-key-change-me'
cs = CognitoSynthetica()

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Play Defense - Fort Knox Eternal</title>
        <style>
            body { font-family: Arial, sans-serif; background: #111; color: #eee; margin: 20px; }
            h1 { color: #0f0; }
            .section { margin: 20px 0; padding: 15px; background: #222; border-radius: 8px; }
            button { padding: 10px 20px; margin: 5px; background: #333; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #555; }
            textarea, input { width: 100%; padding: 10px; background: #333; color: white; border: 1px solid #555; border-radius: 5px; }
            #output { white-space: pre-wrap; background: #000; padding: 15px; border-radius: 8px; min-height: 200px; overflow-y: auto; }
            #status { font-weight: bold; color: #0f0; }
            .alert { color: #ff4444; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Play Defense - Fort Knox Eternal</h1>
        <div id="status">Status: Loading...</div>

        <div class="section">
            <h2>Security Level</h2>
            {% for level in levels %}
                <button onclick="setLevel('{{ level }}')">{{ level }}</button>
            {% endfor %}
            <button onclick="scanNews()">Scan News Cycle</button>
        </div>

        <div class="section">
            <h2>Add Memory</h2>
            <textarea id="memory" rows="4" placeholder="Enter memory text..."></textarea><br>
            <button onclick="addMemory('episodic')">Add Episodic</button>
            <button onclick="addMemory('state')">Add State</button>
        </div>

        <div class="section">
            <h2>Search / Recall</h2>
            <input id="query" type="text" placeholder="Enter query..."><br>
            <button onclick="doSearch()">Search</button>
            <button onclick="doRecall()">Recall</button>
        </div>

        <div class="section">
            <h2>Output Log</h2>
            <div id="output"></div>
        </div>

        <script>
            function updateStatus() {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status').innerText = 'Status: ' + data.status;
                    });
            }

            function addMemory(kind) {
                const text = document.getElementById('memory').value.trim();
                if (!text) return;
                fetch('/add_memory', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text, kind: kind})
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('output').innerHTML += '<p>Added ' + kind + ' room ' + data.rid + '</p>';
                    document.getElementById('memory').value = '';
                    updateStatus();
                });
            }

            function doSearch() {
                const query = document.getElementById('query').value.trim();
                if (!query) return;
                fetch('/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: query})
                })
                .then(response => response.json())
                .then(data => {
                    let out = '<p>Search: ' + query + '</p>';
                    if (data.safe === false) {
                        out += '<p class="alert">Threat: ' + data.tier + ' - ' + data.reason + '<br>Lure: ' + (data.details?.lure || 'None') + '</p>';
                    } else {
                        data.results.forEach(r => {
                            out += '<p>- ' + r.canonical.substring(0, 100) + '...</p>';
                        });
                    }
                    document.getElementById('output').innerHTML += out;
                    updateStatus();
                });
            }

            function doRecall() {
                const query = document.getElementById('query').value.trim();
                if (!query) return;
                fetch('/recall', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: query})
                })
                .then(response => response.json())
                .then(data => {
                    let out = '<p>Recall: ' + query + '</p>';
                    data.results.forEach(r => {
                        out += '<p>- ' + r.canonical.substring(0, 100) + '...</p>';
                    });
                    document.getElementById('output').innerHTML += out;
                    updateStatus();
                });
            }

            function setLevel(level) {
                fetch('/set_level', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({level: level})
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('output').innerHTML += '<p>Switched to ' + level + '</p>';
                    updateStatus();
                });
            }

            function scanNews() {
                fetch('/autotune_news')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('output').innerHTML += '<p>News scan: Risk ' + data.risk.toFixed(2) + ' → ' + data.level + '</p>';
                        updateStatus();
                    });
            }

            // Auto-refresh status
            setInterval(updateStatus, 5000);
        </script>
    </body>
    </html>
    ''', cs=cs, levels=BASE_KNOBS.keys())

@app.route('/status')
def status():
    return jsonify({"status": cs.status()})

@app.route('/add_memory', methods=['POST'])
def add_memory():
    data = request.json
    rid = cs.add_memory(data['text'], kind=data['kind'])
    return jsonify({"rid": rid})

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    results = cs.search(data['query'], top_k=5)
    safe = results[0].get('safe', True) if results else True
    return jsonify({"results": [r for r in results if not isinstance(r, dict) or 'safe' not in r], "safe": safe})

@app.route('/recall', methods=['POST'])
def recall():
    data = request.json
    results = cs.recall(data['query'], top_k=5)
    return jsonify({"results": results})

@app.route('/set_level', methods=['POST'])
def set_level():
    data = request.json
    cs.set_security_level(data['level'])
    return jsonify({"success": True})

@app.route('/autotune_news')
def autotune_news():
    cs.autotune_from_news()
    return jsonify({"risk": 0.5, "level": cs.current_level})

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Timer(1.25, open_browser).start()
    app.run(debug=False, use_reloader=False)
