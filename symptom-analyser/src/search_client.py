# src/search_client.py
import os
import requests

BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"

def bing_search(query, api_key, count=5):
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": count, "textDecorations": False, "textFormat": "Raw"}
    resp = requests.get(BING_ENDPOINT, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    results = []
    if "webPages" in data:
        for v in data["webPages"].get("value", []):
            results.append({
                "title": v.get("name"),
                "url": v.get("url"),
                "snippet": v.get("snippet")
            })
    return results
