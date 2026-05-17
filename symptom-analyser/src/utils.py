# src/utils.py
import re
import pandas as pd

def load_disease_catalog(csv_path="data/diseases_sample.csv"):
    df = pd.read_csv(csv_path)
    catalog = {}
    for _, row in df.iterrows():
        disease = str(row['disease']).strip()
        aliases = str(row.get('aliases','')).split('|') if not pd.isna(row.get('aliases')) else []
        catalog[disease] = [disease.lower()] + [a.lower().strip() for a in aliases if a]
    return catalog

def generate_queries(symptoms):
    s = symptoms.strip()
    return [f"{s} causes", f"{s} differential diagnosis", f"{s} symptoms cause", f"{s} possible causes"]

def dedupe_by_url(items):
    seen = set()
    out = []
    for it in items:
        u = it.get('url') or it.get('link') or it.get('uri')
        if not u:
            out.append(it)
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(it)
    return out
