import json

from jobmatch_tune.dataset.import_public_match_data import convert_rows, load_sources, read_rows


def test_load_match_sources(tmp_path):
    manifest = tmp_path / "match_sources.yaml"
    manifest.write_text(
        "sources:\n"
        "  - name: demo\n"
        "    schema: match_pair_rows\n"
        "    path: demo.jsonl\n",
        encoding="utf-8",
    )
    sources = load_sources(manifest)
    assert sources[0]["schema"] == "match_pair_rows"


def test_read_match_json_rows(tmp_path):
    path = tmp_path / "match.json"
    path.write_text(json.dumps([{"resume": "A", "job_description": "B"}]), encoding="utf-8")
    rows = read_rows(path)
    assert rows == [{"resume": "A", "job_description": "B"}]


def test_convert_match_pair_rows():
    source = {
        "name": "match_demo",
        "schema": "match_pair_rows",
        "path": "demo.jsonl",
        "mapping": {
            "jd_text": "job_description",
            "resume_text": "resume",
            "label": "label",
            "score": "score",
        },
    }
    rows = [
        {
            "job_description": "岗位名称：后端开发工程师",
            "resume": "目标岗位：后端开发",
            "label": "fit",
            "score": 0.88,
        }
    ]
    converted = convert_rows(source, rows)
    assert converted[0]["task"] == "match"
    assert converted[0]["label"]["raw_label"] == "fit"
