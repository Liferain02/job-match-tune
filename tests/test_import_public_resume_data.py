import json

from jobmatch_tune.dataset.import_public_resume_data import convert_rows, load_sources, read_rows


def test_load_resume_sources(tmp_path):
    manifest = tmp_path / "resume_sources.yaml"
    manifest.write_text(
        "sources:\n"
        "  - name: demo\n"
        "    schema: resume_parse_rows\n"
        "    path: demo.jsonl\n",
        encoding="utf-8",
    )
    sources = load_sources(manifest)
    assert sources[0]["name"] == "demo"


def test_read_resume_jsonl_rows(tmp_path):
    path = tmp_path / "resume.jsonl"
    path.write_text(json.dumps({"text": "A"}) + "\n", encoding="utf-8")
    rows = read_rows(path)
    assert rows == [{"text": "A"}]


def test_convert_resume_parse_rows():
    source = {
        "name": "faircv_demo",
        "schema": "resume_parse_rows",
        "path": "demo.jsonl",
        "mapping": {
            "text": "text",
            "target_job": "label.目标岗位",
            "education": "label.教育背景",
            "skills": "label.核心技能",
            "internships": "label.实习经历",
            "projects": "label.项目经历",
            "strengths": "label.优势标签",
        },
    }
    rows = [
        {
            "text": "姓名：张三\n目标岗位：后端开发",
            "label": {
                "目标岗位": "后端开发",
                "教育背景": ["本科，计算机科学与技术"],
                "核心技能": ["Java", "MySQL"],
                "实习经历": ["参与支付服务开发"],
                "项目经历": ["订单中心重构"],
                "优势标签": ["微服务"],
            },
        }
    ]
    converted = convert_rows(source, rows)
    assert converted[0]["task"] == "resume_parse"
    assert converted[0]["label"]["目标岗位"] == "后端开发"


def test_convert_resume_ner_rows():
    source = {
        "name": "resume_ner_demo",
        "schema": "resume_ner_rows",
        "path": "demo.jsonl",
        "mapping": {"tokens": "tokens", "tags": "ner_tags"},
    }
    rows = [{"tokens": ["张", "三"], "ner_tags": ["B-NAME", "I-NAME"]}]
    converted = convert_rows(source, rows)
    assert converted[0]["task"] == "resume_ner"
    assert converted[0]["text"] == "张三"


def test_convert_resume_ner_rows_accepts_array_like_values():
    class ArrayLike:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return self.values

    source = {
        "name": "resume_ner_demo",
        "schema": "resume_ner_rows",
        "path": "demo.parquet",
        "mapping": {"tokens": "tokens", "tags": "ner_tags"},
    }
    converted = convert_rows(source, [{"tokens": ArrayLike(["张", "三"]), "ner_tags": ArrayLike([1, 2])}])

    assert converted[0]["tokens"] == ["张", "三"]
    assert converted[0]["ner_tags"] == ["1", "2"]
