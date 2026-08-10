from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def load_samples(path: str | Path, limit: int) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            jd_text = str(row.get("jd_text") or "").strip()
            resume_text = str(row.get("resume_text") or "").strip()
            if jd_text and resume_text:
                samples.append(
                    {
                        "id": str(row.get("id") or f"sample_{len(samples)}"),
                        "jd_text": jd_text,
                        "resume_text": resume_text,
                    }
                )
            if len(samples) >= limit:
                break
    if not samples:
        raise ValueError(f"No usable match samples found in {path}")
    return samples


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run_sample(
    *,
    base_url: str,
    sample: dict[str, str],
    max_new_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = post_json(
            f"{base_url.rstrip('/')}/api/match",
            {
                "jd_text": sample["jd_text"],
                "resume_text": sample["resume_text"],
                "max_new_tokens": max_new_tokens,
            },
            timeout,
        )
        return {
            "id": sample["id"],
            "ok": bool(payload.get("ok")),
            "wall_seconds": time.perf_counter() - started,
            "server_seconds": float(payload.get("latency_seconds") or 0.0),
            "backend": str((payload.get("execution") or {}).get("backend") or "unknown"),
            "parse_mode": str((payload.get("execution") or {}).get("parse_mode") or "unknown"),
        }
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "id": sample["id"],
            "ok": False,
            "wall_seconds": time.perf_counter() - started,
            "server_seconds": 0.0,
            "backend": "unknown",
            "parse_mode": "unknown",
            "error_type": type(exc).__name__,
        }


def summarize(results: list[dict[str, Any]], total_wall_seconds: float) -> dict[str, Any]:
    successful = [result for result in results if result.get("ok")]
    wall_values = [float(result["wall_seconds"]) for result in successful]
    server_values = [float(result["server_seconds"]) for result in successful]
    return {
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "total_wall_seconds": round(total_wall_seconds, 3),
        "throughput_requests_per_second": (
            round(len(successful) / total_wall_seconds, 4) if total_wall_seconds else 0.0
        ),
        "wall_latency_seconds": {
            "mean": round(mean(wall_values), 3) if wall_values else 0.0,
            "p50": round(percentile(wall_values, 0.5), 3),
            "p95": round(percentile(wall_values, 0.95), 3),
            "max": round(max(wall_values), 3) if wall_values else 0.0,
        },
        "server_latency_seconds": {
            "mean": round(mean(server_values), 3) if server_values else 0.0,
            "p50": round(percentile(server_values, 0.5), 3),
            "p95": round(percentile(server_values, 0.95), 3),
        },
        "backend_distribution": Counter(result["backend"] for result in results).most_common(),
        "parse_mode_distribution": Counter(
            result["parse_mode"] for result in results
        ).most_common(),
        "error_type_distribution": Counter(
            str(result.get("error_type") or "") for result in results if not result.get("ok")
        ).most_common(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the JobMatchTune match API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--input", default="data/eval/match_manual_eval_seed.jsonl")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.samples <= 0 or args.concurrency <= 0:
        raise ValueError("--samples and --concurrency must be positive")

    samples = load_samples(args.input, args.samples)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(samples))) as executor:
        results = list(
            executor.map(
                lambda sample: run_sample(
                    base_url=args.base_url,
                    sample=sample,
                    max_new_tokens=args.max_new_tokens,
                    timeout=args.timeout,
                ),
                samples,
            )
        )
    report = {
        "config": {
            "base_url": args.base_url,
            "input": args.input,
            "samples": len(samples),
            "concurrency": args.concurrency,
            "max_new_tokens": args.max_new_tokens,
        },
        "summary": summarize(results, time.perf_counter() - started),
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
