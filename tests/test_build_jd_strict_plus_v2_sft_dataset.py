from pathlib import Path

import yaml

from jobmatch_tune.dataset.build_jd_strict_plus_v2_sft_dataset import build_rows


def test_build_rows_accepts_repairable_pool_row() -> None:
    rows = [
        {
            "id": "repairable_pool_1",
            "source": "careers.tencent.com",
            "job_title": "腾讯会议-Android研发工程师",
            "company": "腾讯",
            "location": "深圳",
            "raw_text": "岗位职责：负责会议 Android 客户端功能研发、性能优化、稳定性建设和多端协同能力落地。\n任职要求：本科及以上，三年以上工作经验，熟悉 Android 开发、性能分析和工程化实践。",
            "meta": {"language": "", "pool_origin": "repairable_candidate"},
        }
    ]
    with Path("configs/label_schema.yaml").open("r", encoding="utf-8") as fp:
        schema = yaml.safe_load(fp)
    built = build_rows(rows, schema)
    assert len(built) == 1
    assert built[0]["labels"]["岗位方向"] == "客户端开发"
