import os
import io
import json
from pathlib import Path

import httpx


ROOT = Path(r"D:/Resume")
OUT = Path(__file__).parent / "batch_results.json"
API = os.getenv("API_URL", "http://localhost:5000/api/analyze")

JOB_DESC = (
    "Software Engineer role requiring Python, REST APIs, Docker, cloud deployment, testing, and scalable backend architecture."
)


def analyze_file(path: Path):
    print(f"Processing: {path.name}")
    with path.open('rb') as f:
        files = {"resume": (path.name, f, "application/pdf")}
        data = {"jobDescription": JOB_DESC}
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(API, files=files, data=data)
            if resp.status_code == 200:
                print(f"  OK: {resp.status_code}")
                return resp.json()
            else:
                print(f"  Failed: {resp.status_code} {resp.text[:200]}")
                return {"error": resp.text, "status_code": resp.status_code}
        except Exception as exc:
            print(f"  Exception: {exc}")
            return {"error": str(exc)}


def main():
    pdfs = sorted([p for p in ROOT.glob('*.pdf')])
    if not pdfs:
        print("No PDF resumes found in", ROOT)
        return

    results = {}
    for p in pdfs:
        res = analyze_file(p)
        results[p.name] = res

    with OUT.open('w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"Saved results to {OUT}")


if __name__ == '__main__':
    main()
