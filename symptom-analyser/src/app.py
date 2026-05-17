print(">>> LOADED THIS APP.PY:", __file__)

import os
import yaml
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from src.utils import load_disease_catalog, generate_queries, dedupe_by_url
from src.retriever_ranker import RetrieverRanker
from src.search_client import bing_search
from src.mock_search import mock_search

# load config
cfg = yaml.safe_load(open("config.yaml"))

API_KEY = cfg['search'].get('api_key') or os.environ.get("BING_API_KEY")
SEARCH_PROVIDER = cfg['search'].get('provider', 'bing')

app = Flask(__name__, static_folder="static", static_url_path="/static")
if cfg['server'].get('enable_cors', True):
    CORS(app)

catalog = load_disease_catalog("data/diseases_sample.csv")
ranker = RetrieverRanker(cfg['model'].get('embedding_model', 'all-MiniLM-L6-v2'))

@app.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json() or {}
    symptoms = payload.get("symptoms", "")
    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    queries = generate_queries(symptoms)[:4]
    all_results = []
    for q in queries:
        try:
            if SEARCH_PROVIDER == "mock":
                res = mock_search(q, API_KEY, count=cfg['search'].get('max_results', 5))
            else:
                if not API_KEY:
                    return jsonify({"error":"Search API key not configured"}), 500
                res = bing_search(q, API_KEY, count=cfg['search'].get('max_results', 5))
            all_results.extend(res)
        except Exception as e:
            print("Search error:", e)

    dedup = dedupe_by_url(all_results)
    ranked = ranker.rank_candidates(symptoms, dedup, catalog, evidence_weight=cfg['scoring'].get('evidence_weight', 0.7))
    top = []
    for item in ranked[:5]:
        top.append({
            "disease": item['disease'],
            "score": item['score'],
            "freq": item['freq'],
            "best_sim": item['best_sim'],
            "evidence": item.get('evidence', [])[:3]
        })

    min_conf = cfg['scoring'].get('min_confidence', 0.35)
    top = [t for t in top if t['score'] >= min_conf]

    return jsonify({
        "input": symptoms,
        "candidates": top,
        "meta": {"queries_used": queries, "num_snippets": len(dedup)}
    })

# serve the widget (static)
@app.route("/")
def widget():
    return send_from_directory("static", "widget.html")

if __name__ == "__main__":
    app.run(
        host=cfg['server'].get('host','0.0.0.0'),
        port=cfg['server'].get('port',5000),
        debug=False,
        use_reloader=False
    )

    