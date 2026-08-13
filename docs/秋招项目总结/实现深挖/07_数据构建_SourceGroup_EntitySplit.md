# 数据构建、SourceGroup、EntitySplit

## 为什么先分实体再配对

同一个 JD 或简历经过模板改写、OCR 扰动或正负配对后仍属于同一信息实体。若先随机配对再切分，近重复实体会同时进入 train/valid/test，指标会虚高。因此匹配数据在 `build_match_train_pool_synthetic.py::partition_match_entities` 中先按 JD/Resume 实体 hash 分组切分，再在分区内配对。

## 数据入口与主要函数

- 来源登记：`configs/public_job_sources*.yaml`、`public_resume_sources.yaml`、`public_match_sources.yaml` 和 `dataset_registry.yaml`。
- 下载：`download_verified_sources.py::verify_or_download_source` 校验 URL、许可元数据与 SHA256。
- 导入：`import_public_job_data.py`、`import_public_resume_data.py`、`import_public_match_data.py` 转换字段并保留 source/source_group/source_entity_id。
- 清洗：`clean_text.py`、`deduplicate.py::deduplicate_rows`；近重复使用 normalized shingles/Jaccard，确定重复用 fingerprint。
- 分组切分：`grouped_split.py::split_linked_samples`；多任务数据由 `build_multitask_sft_dataset.py::source_group_key` 继续保持来源组隔离。
- 池构建：JD、Resume、Match 各自 `build_*_train_pool_*`；SFT builder 再转成 messages。

输入是带原始文本、来源、许可、实体 ID 的 JSONL；中间结果增加 normalized text、label、content hash、split 和质量标签；输出是 train/validation/holdout JSONL 与 profile/audit。训练脚本只消费门禁认可的 split。

## 例子、降级与测试

一份简历生成原文版和 OCR 版，两者必须共享 group；同一 JD 与两个负样本配对，也不能跨 split。来源缺失、许可不清、真实 Pair 证据不足的记录进入 candidate/quarantine，而不是训练集。

测试包括 `tests/test_grouped_split.py`、`test_build_match_train_pool_synthetic.py`、`test_build_multitask_sft_dataset.py`、`test_deduplicate.py` 和各数据审计测试。

当前事实边界：Resume SFT 主要来自 bootstrap，4,798 个 Match Pair 是规则合成，显式年限样本缺少“满足”正例。数据量大不能抵消来源和标签证据不足，所以 readiness 仍为 false。
