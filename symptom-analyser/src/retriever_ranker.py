# src/retriever_ranker.py
from sentence_transformers import SentenceTransformer, util
import numpy as np

class RetrieverRanker:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        if not texts:
            return []
        return self.model.encode(texts, convert_to_tensor=True)

    def rank_candidates(self, symptom_text, evidence_items, disease_catalog, evidence_weight=0.7):
        sym_emb = self.embed([symptom_text])[0]
        snippets = [ (f"{it.get('title','')} . {it.get('snippet','')}", it) for it in evidence_items ]
        texts = [s for s,_ in snippets]
        if not texts:
            return []
        snip_embs = self.embed(texts)
        sim_scores = util.pytorch_cos_sim(sym_emb, snip_embs)[0].cpu().numpy()

        disease_scores = {}
        for idx, (text, it) in enumerate(snippets):
            text_l = text.lower()
            for disease, aliases in disease_catalog.items():
                for a in aliases:
                    if a and a in text_l:
                        info = disease_scores.setdefault(disease, {"freq":0, "best_sim":0.0, "evidence":[]})
                        info['freq'] += 1
                        info['best_sim'] = max(info['best_sim'], float(sim_scores[idx]))
                        info['evidence'].append(it)

        results = []
        for disease, info in disease_scores.items():
            freq_norm = info['freq'] / max(1, len(evidence_items))
            sim = info['best_sim']
            score = evidence_weight * sim + (1 - evidence_weight) * freq_norm
            results.append({
                "disease": disease,
                "score": float(score),
                "freq": info['freq'],
                "best_sim": float(sim),
                "evidence": info['evidence']
            })
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
