from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


WORKLOADS = ("jd_parse", "resume_parse", "match", "batch_parse", "batch_match")


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


def request_json(
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    request = Request(
        url,
        data=(json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None),
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    return request_json(url, payload, timeout)


def workload_request(
    workload: str,
    samples: list[dict[str, str]],
    index: int,
    *,
    batch_size: int,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any], list[str]]:
    sample = samples[index % len(samples)]
    if workload == "jd_parse":
        return "/api/parse", {"task": "jd_parse", "text": sample["jd_text"], "max_new_tokens": max_new_tokens}, [sample["id"]]
    if workload == "resume_parse":
        return "/api/parse", {"task": "resume_parse", "text": sample["resume_text"], "max_new_tokens": max_new_tokens}, [sample["id"]]
    if workload == "match":
        return "/api/match", {"jd_text": sample["jd_text"], "resume_text": sample["resume_text"], "max_new_tokens": max_new_tokens}, [sample["id"]]
    selected = [samples[(index * batch_size + offset) % len(samples)] for offset in range(batch_size)]
    ids = [item["id"] for item in selected]
    if workload == "batch_parse":
        return "/api/batch_parse", {"task": "jd_parse", "texts": [item["jd_text"] for item in selected], "max_new_tokens": max_new_tokens}, ids
    if workload == "batch_match":
        return "/api/batch_match", {"items": [{"jd_text": item["jd_text"], "resume_text": item["resume_text"]} for item in selected], "max_new_tokens": max_new_tokens}, ids
    raise ValueError(f"Unsupported workload: {workload}")


def run_request(
    *,
    base_url: str,
    workload: str,
    samples: list[dict[str, str]],
    index: int,
    batch_size: int,
    max_new_tokens: int,
    timeout: float,
    sender: Callable[[str, dict[str, Any], float], dict[str, Any]] = post_json,
) -> dict[str, Any]:
    endpoint, payload, ids = workload_request(
        workload,
        samples,
        index,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    started = time.perf_counter()
    try:
        response = sender(f"{base_url.rstrip('/')}{endpoint}", payload, timeout)
        execution = response.get("execution") or {}
        return {
            "request_index": index,
            "sample_ids": ids,
            "batch_size": len(ids),
            "ok": bool(response.get("ok")),
            "wall_seconds": time.perf_counter() - started,
            "server_seconds": float(response.get("latency_seconds") or 0.0),
            "backend": str(execution.get("backend") or "unknown"),
            "parse_mode": str(execution.get("parse_mode") or execution.get("mode") or "unknown"),
            "output_tokens": response.get("usage", {}).get("completion_tokens"),
        }
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "request_index": index,
            "sample_ids": ids,
            "batch_size": len(ids),
            "ok": False,
            "wall_seconds": time.perf_counter() - started,
            "server_seconds": 0.0,
            "backend": "unknown",
            "parse_mode": "unknown",
            "output_tokens": None,
            "error_type": type(exc).__name__,
        }


def summarize(results: list[dict[str, Any]], total_wall_seconds: float) -> dict[str, Any]:
    successful = [result for result in results if result.get("ok")]
    wall_values = [float(result["wall_seconds"]) for result in successful]
    server_values = [float(result["server_seconds"]) for result in successful]
    completed_samples = sum(int(result.get("batch_size") or 1) for result in successful)
    token_counts = [result.get("output_tokens") for result in successful]
    tokens_reliable = bool(successful) and all(isinstance(value, int) for value in token_counts)
    total_tokens = sum(int(value) for value in token_counts if isinstance(value, int))
    return {
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "completed_samples": completed_samples,
        "total_wall_seconds": round(total_wall_seconds, 3),
        "throughput_requests_per_second": round(len(successful) / total_wall_seconds, 4) if total_wall_seconds else 0.0,
        "throughput_samples_per_second": round(completed_samples / total_wall_seconds, 4) if total_wall_seconds else 0.0,
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
        "tokens_per_second": round(total_tokens / total_wall_seconds, 3) if tokens_reliable and total_wall_seconds else None,
        "tokens_per_second_status": "reported_by_api" if tokens_reliable else "unavailable_api_does_not_report_usage",
        "backend_distribution": Counter(result["backend"] for result in results).most_common(),
        "parse_mode_distribution": Counter(result["parse_mode"] for result in results).most_common(),
        "error_type_distribution": Counter(str(result.get("error_type") or "") for result in results if not result.get("ok")).most_common(),
    }


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"status": "unavailable", "gpus": []}
    gpus = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 6:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "uuid": parts[2],
                    "memory_total_mib": int(parts[3]),
                    "memory_used_mib": int(parts[4]),
                    "utilization_percent": int(parts[5]),
                }
            )
    return {"status": "nvidia_smi", "gpus": gpus}


def run_workload(
    *,
    base_url: str,
    workload: str,
    samples: list[dict[str, str]],
    request_count: int,
    concurrency: int,
    batch_size: int,
    max_new_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(concurrency, request_count)) as executor:
        results = list(
            executor.map(
                lambda index: run_request(
                    base_url=base_url,
                    workload=workload,
                    samples=samples,
                    index=index,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    timeout=timeout,
                ),
                range(request_count),
            )
        )
    return {"summary": summarize(results, time.perf_counter() - started), "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark all JobMatchTune API workloads")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--input", default="data/eval/match_manual_eval_seed.jsonl")
    parser.add_argument("--samples", type=int, default=8, help="input rows loaded from the common set")
    parser.add_argument("--requests", type=int, default=4, help="requests per workload")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out", default="outputs/benchmarks/api_backend_benchmark.json")
    args = parser.parse_args()
    if min(args.samples, args.requests, args.concurrency, args.batch_size) <= 0 or args.warmup < 0:
        raise ValueError("samples, requests, concurrency and batch-size must be positive; warmup must be non-negative")

    samples = load_samples(args.input, args.samples)
    health = request_json(f"{args.base_url.rstrip('/')}/health", None, args.timeout)
    for workload in args.workloads:
        for index in range(args.warmup):
            run_request(base_url=args.base_url, workload=workload, samples=samples, index=index, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, timeout=args.timeout)

    gpu_before = gpu_snapshot()
    workloads = {
        workload: run_workload(
            base_url=args.base_url,
            workload=workload,
            samples=samples,
            request_count=args.requests,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            timeout=args.timeout,
        )
        for workload in args.workloads
    }
    report = {
        "benchmark_status": "measured",
        "config": {
            "base_url": args.base_url,
            "input": args.input,
            "samples": len(samples),
            "requests_per_workload": args.requests,
            "concurrency": args.concurrency,
            "batch_size": args.batch_size,
            "warmup_requests_per_workload": args.warmup,
            "max_new_tokens": args.max_new_tokens,
        },
        "runtime": {
            "backend": health.get("backend"),
            "model": health.get("model_path") or health.get("vllm_model"),
            "adapter": health.get("adapter_path"),
            "match_parse_mode": health.get("match_parse_mode"),
        },
        "gpu_before": gpu_before,
        "gpu_after": gpu_snapshot(),
        "workloads": workloads,
        "limitations": [
            "tokens/sec is null unless the API exposes completion token usage",
            "GPU memory is an nvidia-smi process-wide snapshot, not per-request allocation",
            "quality must be evaluated separately; latency alone cannot promote a backend",
        ],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
