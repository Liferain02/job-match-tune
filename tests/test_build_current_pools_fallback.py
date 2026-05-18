from pathlib import Path
from tempfile import TemporaryDirectory

from jobmatch_tune.dataset.build_jd_train_pool_combined import main as jd_main
from jobmatch_tune.dataset.build_match_train_pool_combined import main as match_main
from jobmatch_tune.dataset.build_resume_train_pool_combined import main as resume_main


def test_combined_pool_builders_allow_missing_public_inputs(monkeypatch):
    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        manual_resume = base / "resume_manual.jsonl"
        manual_match = base / "match_manual.jsonl"
        manual_jd = base / "jd_manual.jsonl"
        manual_resume.write_text('{"id":"r1","task":"resume_parse","source_type":"text","text":"姓名：张三\\n目标岗位：后端开发工程师\\n教育背景：本科，计算机科学与技术\\n核心技能：Java、MySQL\\n项目经历：订单中心重构与缓存优化。","label":{"目标岗位":"后端开发","教育背景":["本科，计算机科学与技术"],"核心技能":["Java","MySQL"],"项目经历":["订单中心重构与缓存优化。"]}}\n', encoding="utf-8")
        manual_match.write_text('{"id":"m1","task":"match","source_type":"text","jd_text":"岗位名称：后端开发工程师\\n岗位职责：负责服务开发与治理。\\n任职要求：熟悉 Java、MySQL、Redis，负责高并发接口开发和稳定性建设。","resume_text":"目标岗位：后端开发\\n教育背景：本科，计算机科学与技术\\n核心技能：Java、MySQL、Redis\\n项目经历：订单中心重构与缓存优化，参与支付服务接口开发。","label":{"匹配等级":"高匹配"}}\n', encoding="utf-8")
        manual_jd.write_text('{"id":"j1","source":"zhaopin.jd.com","language":"zh","job_title":"后端开发工程师","company":"示例公司","location":"北京","salary":"20-30K","clean_text":"岗位职责：负责交易链路服务开发与治理，推进缓存优化、接口稳定性建设和告警治理。\\n任职要求：本科及以上，熟悉 Java、MySQL、Redis。\\n技能要求：熟悉 Linux、SQL 和日志排障。","sections":{"responsibilities":"负责交易链路服务开发与治理。","requirements":"本科及以上，熟悉 Java、MySQL、Redis。","bonus":""},"labels":{"岗位方向":"后端开发","必备技能":["Java","MySQL","Redis"],"学历要求":"本科"},"sft_ready":true}\n', encoding="utf-8")

        out_resume = base / "resume_combined.jsonl"
        out_match = base / "match_combined.jsonl"
        out_jd = base / "jd_combined.jsonl"

        monkeypatch.setattr("sys.argv", ["resume", "--manual-input", str(manual_resume), "--public-input", str(base / "missing_resume.jsonl"), "--out", str(out_resume)])
        resume_main()
        monkeypatch.setattr("sys.argv", ["match", "--manual-input", str(manual_match), "--public-input", str(base / "missing_match.jsonl"), "--out", str(out_match)])
        match_main()
        monkeypatch.setattr("sys.argv", ["jd", "--manual-input", str(manual_jd), "--public-input", str(base / "missing_jd.jsonl"), "--out", str(out_jd)])
        jd_main()

        assert out_resume.exists()
        assert out_match.exists()
        assert out_jd.exists()
