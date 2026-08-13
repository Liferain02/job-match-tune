# Readiness、Source Admission、隐私、泄漏

## 核心原则

“文件能下载”“格式能训练”和“允许进入训练”是三件事。Readiness 把许可、来源独立性、隐私、标签证据、分布完整性和泄漏检查变成训练前硬门禁，而不是训练后的免责声明。

## 实现链路

- Source Admission：`report_resume_source_admission.py` 与各 `public_*_sources.yaml` 检查来源是否启用、许可、用途和转换器。
- 隐私：`resume/privacy.py`、`report_resume_privacy_readiness.py` 扫描 PII；原始简历放 `data/private` 且由 `.gitignore` 排除。
- 数据审计：`audit_public_{jd,resume,match}_data.py`、`audit_sft_dataset.py` 汇总来源、标签和异常。
- 泄漏：`audit_match_gold.py::audit_match_gold` 比较 Pair/JD/Resume hash、source group、entity 以及近重复；`grouped_split.py` 防训练内跨 split 泄漏。
- 汇总：`report_data_readiness.py`、`report_product_readiness.py`；`assert_training_readiness.py` 和 shell gate 在训练入口前执行。

输入为 registry、manifest、训练 split、Gold 和审计文件；输出每项 check 的 pass/fail、理由、样本统计和整体 ready。调用方是训练脚本、人审和冻结报告。

## 失败与边界

任何必需报告缺失、过期或 fail，都应该阻断训练；不会自动降低标准。当前 Resume 独立来源、真实 Match Pair、年限正例及偏好 holdout 不足，因此相关 readiness=false。

`tests/test_report_data_readiness.py`、`test_report_resume_source_admission.py`、`test_report_resume_privacy_readiness.py`、`test_audit_match_gold.py` 覆盖门禁。

边界：hash 零重叠不能证明语义完全独立；正则隐私扫描也不能证明所有 PII 已移除。必须结合来源实体分组、人工抽查和许可记录。外部条件未变化时，不应靠更多合成样本把 fail 改成 pass。
