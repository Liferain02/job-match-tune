import json
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/eval/benchmark_match_api.py"
SPEC = importlib.util.spec_from_file_location("benchmark_match_api", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
load_samples = MODULE.load_samples
percentile = MODULE.percentile
summarize = MODULE.summarize
workload_request = MODULE.workload_request


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    assert percentile([], 0.95) == 0.0


def test_load_samples_filters_incomplete_rows(tmp_path) -> None:
    path = tmp_path / "match.jsonl"
    rows = [
        {"id": "bad", "jd_text": "JD"},
        {"id": "good", "jd_text": "JD", "resume_text": "Resume"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assert load_samples(path, 1) == [
        {"id": "good", "jd_text": "JD", "resume_text": "Resume"}
    ]


def test_summarize_does_not_include_request_text() -> None:
    report = summarize(
        [
            {
                "id": "sample",
                "ok": True,
                "wall_seconds": 2.0,
                "server_seconds": 1.5,
                "backend": "vllm",
                "parse_mode": "parallel",
            }
        ],
        total_wall_seconds=2.0,
    )
    assert report["successful_requests"] == 1
    assert report["throughput_requests_per_second"] == 0.5
    assert report["throughput_samples_per_second"] == 0.5
    assert report["tokens_per_second"] is None
    assert report["parse_mode_distribution"] == [("parallel", 1)]


def test_all_workloads_use_the_same_input_set() -> None:
    samples = [
        {"id": "a", "jd_text": "JD A", "resume_text": "Resume A"},
        {"id": "b", "jd_text": "JD B", "resume_text": "Resume B"},
    ]
    assert workload_request("jd_parse", samples, 0, batch_size=2, max_new_tokens=64)[2] == ["a"]
    assert workload_request("resume_parse", samples, 0, batch_size=2, max_new_tokens=64)[2] == ["a"]
    assert workload_request("match", samples, 0, batch_size=2, max_new_tokens=64)[2] == ["a"]
    assert workload_request("batch_parse", samples, 0, batch_size=2, max_new_tokens=64)[2] == ["a", "b"]
    assert workload_request("batch_match", samples, 0, batch_size=2, max_new_tokens=64)[2] == ["a", "b"]
