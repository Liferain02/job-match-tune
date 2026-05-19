from jobmatch_tune.dataset.build_resume_train_pool_synthetic import build_rows


def test_build_rows_renders_multiple_resume_variants():
    rows = [
        {
            "id": "resume_1",
            "text": "目标岗位：后端开发",
            "label": {
                "目标岗位": "后端开发",
                "教育背景": ["本科，计算机科学与技术"],
                "核心技能": ["Java", "MySQL"],
                "实习经历": ["参与接口开发。"],
                "项目经历": ["负责订单中心重构。"],
                "优势标签": ["微服务架构"],
            },
        }
    ]
    built = build_rows(rows)
    assert len(built) >= 10
    assert all(row["task"] == "resume_parse" for row in built)
