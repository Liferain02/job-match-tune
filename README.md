# JobMatchTune

面向招聘 JD / 简历结构化抽取的 Qwen3 QLoRA 微调项目。当前默认服务版本为 `Qwen3-14B + LoRA adapter + 规则后处理`。

## 当前默认版本

- 基座模型：`models/Qwen3-14B`
- Adapter：`outputs/checkpoints/qwen3-14b-jobmatch-qlora`
- 服务默认入口：
  - API: [src/jobmatch_tune/api/server.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/api/server.py)
  - 启动脚本: [scripts/serve/start_api.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/serve/start_api.sh)
  - vLLM 服务脚本: [scripts/serve/start_vllm_server.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/serve/start_vllm_server.sh)
- 50 条人工 holdout 最新报告：
  - [outputs/eval_reports/manual_eval_50_qwen3_14b_v3_report.json](/share/home/lifr/workspace/code/job-match-tune/outputs/eval_reports/manual_eval_50_qwen3_14b_v3_report.json)

## 项目结构

详细说明见 [docs/project_structure.md](/share/home/lifr/workspace/code/job-match-tune/docs/project_structure.md)。

核心目录：

- `src/jobmatch_tune/`
  - `crawler/`：公开 JD 抓取
  - `preprocess/`：清洗、去重、规则抽取
  - `dataset/`：SFT / DPO 数据构造
  - `train/`：QLoRA / DPO 训练
  - `inference/`：推理与后处理
  - `api/`：FastAPI 服务
  - `eval/`：人工评估与指标
- `scripts/data/`：抓取、导入、重建数据
- `scripts/train/`：14B 训练入口
- `scripts/serve/`：API / vLLM / 前端启动
- `scripts/dev/`：环境与模型下载
- `scripts/research/`：研究辅助脚本
- `scripts/legacy/`：历史 1.7B 实验脚本归档
- `configs/`：训练、爬取、标签 schema
- `frontend/`：Vue 3 ESM 前端（无构建步骤）
- `docs/`：实验记录与口径文档

## 环境

```bash
conda create -n tune-demo python=3.11 -y
conda activate tune-demo
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

查看 `gpu03` 资源：

```bash
ssh -n gpu03 nvidia-smi
```

## 数据链路

初始化数据库：

```bash
python -m jobmatch_tune.init_db --db data/jobmatch_tune.sqlite3
```

抓取腾讯公开招聘 JD：

```bash
python -m jobmatch_tune.crawler.tencent_careers \
  --keywords-file configs/tencent_keywords.txt \
  --limit 3000 \
  --page-size 50 \
  --max-pages 30 \
  --interval-seconds 0.5 \
  --category 技术 \
  --out data/raw/tencent_jd_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
```

抓取百度公开招聘 JD：

```bash
python -m jobmatch_tune.crawler.baidu_talent \
  --keywords-file configs/baidu_keywords.txt \
  --interval-seconds 0.5 \
  --out data/raw/baidu_jd_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
```

抓取京东公开招聘 JD：

```bash
python -m jobmatch_tune.crawler.jd_careers \
  --out data/raw/jd_careers_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
```

抓取携程公开招聘 JD：

```bash
python -m jobmatch_tune.crawler.ctrip_careers \
  --out data/raw/ctrip_jd_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
```

抓取 Moka 托管招聘官网 JD：

```bash
python -m jobmatch_tune.crawler.moka_careers \
  --sources configs/moka_sources.yaml \
  --out data/raw/moka_jd_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
```

如需一键刷新腾讯数据：

```bash
bash scripts/data/refresh_tencent_data.sh auto
```

如需一键刷新百度数据：

```bash
bash scripts/data/refresh_baidu_data.sh
```

如需一键刷新京东数据：

```bash
bash scripts/data/refresh_jd_data.sh
```

如需一键刷新携程数据：

```bash
bash scripts/data/refresh_ctrip_data.sh
```

如需一键刷新小米数据：

```bash
bash scripts/data/refresh_xiaomi_data.sh
```

当前小米抓取已同时覆盖：
- 旧版研发职位列表页 `8-0-2`
- 关键词搜索页：`开发 / 算法 / 前端 / 后端 / 客户端 / 测试 / 数据 / Java`

如需一键刷新美团数据：

```bash
bash scripts/data/refresh_meituan_data.sh
```

当前美团公开招聘 API 已验证可用：
- 列表：`/api/official/job/getJobList`
- 详情：`/api/official/job/getJobDetail`
- 本轮接入后新增 `18` 条 tech-like raw，进入 `JD strict` 主集 `5` 条

如需一键刷新滴滴数据：

```bash
bash scripts/data/refresh_didi_data.sh
```

当前滴滴公开招聘 API 已验证可用：
- 列表：`/recruit-portal-service/api/job/front/list`
- 详情：`/recruit-portal-service/api/job/front/view/{jdId}`
- 本轮接入后新增 `540` 条 tech-like raw，进入 dedup `528` 条，进入 `JD strict` 主集 `395` 条

当前携程公开招聘 API 已验证可用：
- 列表：`/api/hrrecruit/getJobAd`
- 当前已抓取 `656` 条 tech-like raw，已进入 `jd_clean` 层 `656` 条
- 按现有 `strict` 准入规则单独评估，这批样本约有 `193` 条可进入 `JD strict`

如需一键刷新 Moka 招聘官网数据：

```bash
bash scripts/data/refresh_moka_data.sh
```

如需探测 Feishu ATS 招聘官网（例如得物）公开接口可用性：

```bash
bash scripts/data/probe_feishu_ats.sh https://poizon.jobs.feishu.cn \
  outputs/eval_reports/poizon_feishu_probe.json
```

说明：

- 这个脚本不会把数据直接接进训练集。
- 它会探测：
  - `websiteInfo`
  - 页面脚本 bundle URL
  - 主 bundle 中暴露的 `/api/...` 路径
  - `config/job/filters/{portal_type}`
  - `job/posts/{id}`
  - `search/job/posts`
- 适合用来判断一个 Feishu ATS 站点是否值得继续接成正式 crawler。
- 当前得物站点重跑后的结论是：
  - 详情与 filters 接口可访问
  - bundle 扫描只暴露了 `/api/embed/error-page/`
  - 仍未恢复出职位列表接口

如需探测蚂蚁招聘公开接口可用性：

```bash
bash scripts/data/probe_ant_careers.sh \
  outputs/eval_reports/ant_probe.json
```

说明：

- 这个脚本当前不会直接抓取 JD。
- 它会探测：
  - `/api/searchCondition/list`
  - `/api/searchCondition/listPositionGroup`
  - `/api/searchCondition/listTalentPlan`
  - `/api/social/position/search`
  - `/api/position/searchPositionIdsByQuery`
- 当前已确认筛选枚举接口匿名可用，并能拿到 `totalPositions` 和 `技术类` 数量。
- probe 现在会自动尝试多组 `social/position/search` payload 变体，区分：
  - 参数结构错误（`400 Bad Request`）
  - 缺少必填字段（`param_can_not_be_null`）
- `social/position/search` 仍需要继续恢复正确 payload，现阶段更适合作为 probe 而不是正式 crawler。

如需探测拼多多校园招聘官网的 Next/接口线索：

```bash
bash scripts/data/probe_pdd_campus.sh \
  outputs/eval_reports/pdd_campus_probe.json
```

说明：

- 这个脚本当前不会直接抓取 JD。
- 它会探测：
  - `__NEXT_DATA__`
  - 页面脚本 bundle URL
  - bundle 中暴露的 `/api/...` 路径线索
  - `/api/` 根路径返回
- 适合用来判断拼多多校园招聘站点是否存在可继续恢复的职位列表/详情接口。

如需探测小红书招聘官网公开接口线索：

```bash
bash scripts/data/probe_xiaohongshu_careers.sh \
  outputs/eval_reports/xiaohongshu_probe.json
```

说明：

- 这个脚本当前不会直接抓取 JD。
- 它会探测：
  - 页面主 bundle URL
  - bundle 中暴露的 `/api/...` 路径
  - 候选数据接口 `/api/store/jpd/main` 的 GET / POST 返回
  - `/api/data`、`/api/bizInUrl` 这类候选接口的返回
- 适合用来判断小红书招聘站点是否存在可继续恢复的职位列表接口。

如需一键刷新腾讯 + 百度 + 京东 + 携程 + 小米 + 美团 + 滴滴 + Moka 并重建下游：

```bash
bash scripts/data/refresh_official_job_data.sh
```

说明：

- `auto`：先尝试抓取，失败则直接用现有 raw 数据重建下游
- `crawl`：强制抓取后再重建
- `rebuild`：只重建清洗、去重和 SFT 数据

当前按最新 `strict` 口径重建后，`JD strict` 主集已到：

- `train / valid / test = 2653 / 331 / 333`
- 总计 `3317`

当前贡献最大的高信任官网源：

- `careers.tencent.com = 853`
- `zhaopin.jd.com = 731`
- `talent.baidu.com = 400`
- `talent.didiglobal.com = 395`
- `careers.ctrip.com = 193`

导入公开职位导出文件并扩充原始语料：

```bash
bash scripts/data/import_public_job_exports.sh
```

导入大规模中文招聘学历数据：

```bash
bash scripts/data/import_chinese_job_exports.sh
```

审计公开 JD 数据：

```bash
bash scripts/data/audit_public_jd_data.sh --input data/raw/public_job_datasets_raw.jsonl
```

从公开 JD 导入语料中筛高质量候选池：

```bash
bash scripts/data/build_public_jd_candidate_pool.sh
```

把默认严格 JD 与公开 JD 候选池合并成统一训练池：

```bash
bash scripts/data/build_jd_train_pool_combined.sh
```

导入公开 resume 数据：

```bash
bash scripts/data/import_public_resume_exports.sh
```

导入公开 match 数据：

```bash
bash scripts/data/import_public_match_exports.sh
```

审计公开 resume 数据：

```bash
bash scripts/data/audit_public_resume_data.sh --input data/external/public_resume_imports.jsonl
```

审计公开 match 数据：

```bash
bash scripts/data/audit_public_match_data.sh --input data/external/public_match_imports.jsonl
```

把人工 resume 训练池与可用公开 `resume_parse` 样本合并：

```bash
bash scripts/data/build_resume_train_pool_combined.sh
```

把人工 `match` 训练池与可用公开匹配样本合并：

```bash
bash scripts/data/build_match_train_pool_combined.sh
```

一键跑公开 `resume` 导入、审计和合并建池：

```bash
bash scripts/data/prepare_public_resume_pipeline.sh
```

一键跑公开 `match` 导入、审计和合并建池：

```bash
bash scripts/data/prepare_public_match_pipeline.sh
```

一键跑公开 `JD` 审计和候选池构造：

```bash
bash scripts/data/prepare_public_jd_pipeline.sh
```

输出统一的数据就绪报告：

```bash
bash scripts/data/report_data_readiness.sh
```

输出当前三个数据池的分布画像：

```bash
bash scripts/data/report_pool_profiles.sh
```

输出外部数据落盘状态报告：

```bash
bash scripts/data/report_external_data_status.sh
```

一键跑当前所有公共数据流水线：

```bash
bash scripts/data/prepare_all_public_pipelines.sh
```

如果外部公开文件还没落盘，但想先基于当前已有数据把三条池子生成出来：

```bash
bash scripts/data/build_current_data_pools.sh
```

当前基于仓库已有数据实际生成出的池子规模：

- `data/eval/public_jd_candidate_pool.jsonl`: `679`
- `data/eval/jd_train_pool_supplemental.jsonl`: `2`
- `data/eval/jd_train_pool_weak_structured.jsonl`: `35477`
- `data/eval/jd_train_pool_combined.jsonl`: `37796`
- `data/eval/resume_train_pool_synthetic.jsonl`: `3200`
- `data/eval/resume_train_pool_from_sft.jsonl`: `3200`
- `data/eval/resume_train_pool_bootstrap.jsonl`: `2600`
- `data/eval/resume_train_pool_combined.jsonl`: `3137`
- `data/eval/match_train_pool_combined.jsonl`: `1176`

当前统一就绪报告结论：

- `JD`: combined pool 已过线，但默认 `sft` 规模仍未过线
- `resume`: 已达到当前训练门槛
- `match`: 已达到当前训练门槛

也就是说，当前仍然不应该开 SFT。

如果要看三个池子现在的来源和分布，不只看总数：

- `JD`：来源 / 标题 / 公司分布
- `resume`：目标岗位 / 核心技能分布
- `match`：匹配等级和原始标签覆盖

直接执行：

```bash
bash scripts/data/report_pool_profiles.sh
```

当前训练集规模：

- `data/sft/train.jsonl`: `2111`
- `data/sft/valid.jsonl`: `263`
- `data/sft/test.jsonl`: `265`
- `data/sft_jd_bootstrap/train.jsonl`: `1371`
- `data/sft_jd_bootstrap/valid.jsonl`: `171`
- `data/sft_jd_bootstrap/test.jsonl`: `172`
- `data/sft_resume/train.jsonl`: `2560`
- `data/sft_resume/valid.jsonl`: `320`
- `data/sft_resume/test.jsonl`: `320`
- `data/sft_match/train.jsonl`: `929`
- `data/sft_match/valid.jsonl`: `132`
- `data/sft_match/test.jsonl`: `115`

当前这条链路会导入三类补充源：

- GitHub `jhcoco/bosszp` CSV
- GitHub `WorkAggregation` CSV
- Hugging Face `open-apply-jobs` 的 Greenhouse / Ashby / Lever parquet 分片
- Hugging Face `job-educational-parser-dataset-08-0-0805` 中文 parquet
- 百度 / 京东 / Moka 招聘官网公开职位抓取

注意：

- 这一步配合腾讯、百度、京东、Moka 官网抓取后，当前 `jd_clean / jd_clean_dedup` 已经达到 `292167 / 267949`。
- 当前去重后语言分布约为：
  - 中文：`221402`
  - 英文：`51330`
  - 其他 / 未知：`927`
- 默认 `data/sft/` 现在是严格质量版：`2111 / 263 / 265`。
- `data/sft_expanded/` 是扩展实验版：`4524 / 565 / 566`。
- 默认训练不再追求先凑满 2 万，而是优先保留高信任官网中文技术岗；弱标注样本只进入扩展实验集，不再直接混入默认集。当前 `20000` 目标只属于扩展实验链路，不代表默认高质量集规模。

如果要把 `JD combined pool` 转成一条独立的 bootstrap SFT 数据线，而不覆盖当前严格高质量集：

```bash
bash scripts/data/build_jd_bootstrap_sft_dataset.sh
```

如果要生成介于 `strict` 和 `bootstrap` 之间的 `JD strict_plus` 数据线：

```bash
bash scripts/data/build_jd_strict_plus_sft_dataset.sh
```

如果要单独抽出“原文信息足够、但当前还能修”的 `JD repairable pool`：

```bash
bash scripts/data/build_jd_train_pool_repairable.sh
```

如果要基于这条修复池再生成实验性的 `JD strict_plus_v2`：

```bash
bash scripts/data/build_jd_strict_plus_v2_sft_dataset.sh
```

当前 `strict_plus` 规模为：

- `train / valid / test = 1188 / 148 / 149`
- 总计 `1485`

当前 `repairable pool` 和 `strict_plus_v2` 的规模为：

- `data/eval/jd_train_pool_repairable.jsonl = 310`
- `data/sft_jd_strict_plus_v2/train|valid|test = 2 / 1 / 1`

结论：真正“原文足够、但可通过小修复救回”的高信任 JD 确实存在，但数量远小于几十万 raw 的直觉规模。这条链路适合作为精修候选池，不适合作为主训练数据来源。

如果要直接比较 `JD strict` 和 `JD bootstrap` 的质量指标：

```bash
bash scripts/data/compare_jd_sft_tracks.sh
```

如果要审计高信任官网样本为什么没有进入 `JD strict` 主集：

```bash
bash scripts/data/report_jd_strict_rejections.sh
```

如果要只看其中“标题本身像技术岗”的拒绝样本：

```bash
bash scripts/data/report_jd_strict_tech_candidates.sh
```

当前 `JD bootstrap` 相对 `strict` 的主要差异：

- 样本量：`2426 -> 1714`
- `json_valid_rate`：都为 `1.0`
- `avg_responsibility_count`：`4.45 -> 3.92`
- `avg_skill_count`：`1.07 -> 1.37`
- `education_coverage`：`0.58 -> 1.00`
- `experience_coverage`：`0.66 -> 0.67`

结论：最新 `bootstrap` 已经明显收紧，职责密度和技能密度都在可用区间。它在学历覆盖上仍然更完整，但经验覆盖已经和 `strict` 接近。因此它更适合作为第二阶段增强集，而不是直接替代 `strict` 主集。

当前 `JD strict_plus` 相对 `strict` 和 `bootstrap` 的位置：

- 样本量：`2426 -> 1485 -> 1714`
- `avg_responsibility_count`：`4.45 -> 3.75 -> 3.92`
- `avg_skill_count`：`1.07 -> 1.52 -> 1.37`
- `education_coverage`：`0.58 -> 1.00 -> 1.00`
- `experience_coverage`：`0.66 -> 0.78 -> 0.67`

结论：最新 `strict_plus` 已经从“数量优先”转成“经验字段更完整”的增强集，`experience_coverage` 高于 `strict`，技能密度也更高，但样本量最小。它适合做第二阶段补强，不适合做唯一主集。

当前 `JD strict` 拒绝审计的主要结论：

- `total_rejected = 4455`
- Top 3 原因：
  - `missing_direction = 2420`
  - `sft_not_ready = 1094`
  - `language_not_zh = 348`
- 说明当前 `strict` 的主瓶颈已经不是单纯经验字段缺失，而是：
  - 岗位方向规则没有覆盖到足够多的高信任官网样本
  - 以及一批样本被上游标成 `sft_not_ready`

进一步过滤掉明显业务岗后，真正值得继续回收的技术候选拒绝样本约为：

- `total_tech_like_rejected = 680`
- `total_tech_like_rejected = 453`
- Top reasons:
  - `missing_direction = 180`
  - `missing_edu_exp_skill = 82`
  - `sft_not_ready = 75`
  - `excluded_title = 67`
  - `clean_text_too_short = 21`

补了一条 `careers.tencent.com` 技术短 JD 例外后，腾讯高信任技术岗里的 `clean_text_too_short` 已经从 `238` 压到 `25`。后续随着滴滴、携程等官网源并入并按最新 `strict` 口径重建，`strict` 主集已经进一步提升到 `3317`。

清洗与构造训练集：

```bash
python -m jobmatch_tune.preprocess.normalize_jd \
  --db data/jobmatch_tune.sqlite3 \
  --out data/interim/jd_clean.jsonl \
  --schema configs/label_schema.yaml

python -m jobmatch_tune.dataset.build_sft_dataset \
  --jd data/interim/jd_clean.jsonl \
  --out-dir data/sft \
  --quality-profile strict

python -m jobmatch_tune.dataset.build_sft_dataset \
  --jd data/interim/jd_clean.jsonl \
  --out-dir data/sft_expanded \
  --include-weak-tech \
  --quality-profile expanded \
  --target-total 20000
```

构造万级中英混合弱标注 SFT：

```bash
bash scripts/data/build_multilingual_weak_sft.sh
```

说明：

- `data/sft/` 是默认高质量中文集，只保留高信任中文官网样本。
- `data/sft_expanded/` 是扩展实验集，允许少量高置信弱标注样本进入。
- `data/sft_multilingual_weak/` 是规模优先的中英混合弱标注集，适合做第二阶段扩量实验，不建议直接替换默认 demo 版本。

当前中文数据最多的来源：

1. Hugging Face `job-educational-parser-dataset-08-0-0805`
   - 当前导入：`232064` 条中文职位样本
2. 京东公开招聘
   - 当前抓取：`3054` 条
3. 腾讯公开招聘
   - 当前抓取：`935` 条
4. 百度公开招聘
   - 当前抓取：`577` 条
5. Moka 招聘官网公开 API
   - 当前抓取：`2662` 条

## 训练

14B smoke：

```bash
bash scripts/train/train_qwen3_14b_smoke.sh
```

14B 正式训练：

```bash
bash scripts/train/train_qwen3_14b_full.sh
```

如需下载模型快照：

```bash
bash scripts/dev/download_qwen_models_python.sh 14B
```

轻量回退模型：

```bash
bash scripts/dev/download_qwen_models_python.sh 1.7B
```

## 偏好优化

从人工评估预测结果生成偏好数据：

```bash
bash scripts/data/build_preference_dataset.sh
```

14B DPO 训练：

```bash
bash scripts/train/train_qwen3_14b_dpo.sh
```

说明：

- 当前环境中 `trl==1.4.0` 可直接使用 `DPOTrainer`
- `ORPOTrainer` 当前环境不可直接用，所以仓库先接入了 `DPO`
- `GRPO` 属于更重的在线后训练，不是当前第一优先级
- 当前 DPO adapter 评估报告：
  - [outputs/eval_reports/manual_eval_50_qwen3_14b_dpo_report.json](/share/home/lifr/workspace/code/job-match-tune/outputs/eval_reports/manual_eval_50_qwen3_14b_dpo_report.json)

## 推理与评估

单条推理：

```bash
python -m jobmatch_tune.inference.predict \
  --model models/Qwen3-14B \
  --adapter outputs/checkpoints/qwen3-14b-jobmatch-qlora \
  --task jd_parse \
  --input examples/jd_ai_app.txt \
  --load-4bit
```

匹配分析推理：

```bash
python -m jobmatch_tune.inference.predict \
  --model models/Qwen3-14B \
  --adapter outputs/checkpoints/qwen3-14b-jobmatch-qlora \
  --task match \
  --input examples/jd_ai_app.txt \
  --resume-input examples/resume_llm_app.txt \
  --rule-result '{"匹配分数":82,"匹配等级":"较匹配","岗位方向匹配":true,"学历匹配":true,"经验匹配":true,"命中技能":["Python","FastAPI"],"缺失技能":["RAG"],"命中项目":["负责企业知识库问答系统开发"]}' \
  --load-4bit
```

50 条人工评估：

```bash
PYTHONPATH=src python -m jobmatch_tune.eval.run_manual_eval \
  --dataset data/eval/jd_manual_eval_50.jsonl \
  --model models/Qwen3-14B \
  --adapter outputs/checkpoints/qwen3-14b-jobmatch-qlora \
  --out outputs/eval_reports/manual_eval_50_qwen3_14b_v3_report.json \
  --predictions-out outputs/eval_reports/manual_eval_50_qwen3_14b_v3_predictions.jsonl \
  --load-4bit
```

简历解析人工评估集构造与评估：

```bash
bash scripts/data/build_resume_eval_dataset.sh

PYTHONPATH=src python -m jobmatch_tune.eval.run_manual_eval \
  --dataset data/eval/resume_manual_eval_seed.jsonl \
  --model models/Qwen3-14B \
  --adapter outputs/checkpoints/qwen3-14b-jobmatch-qlora \
  --out outputs/eval_reports/resume_manual_eval_seed_report.json \
  --predictions-out outputs/eval_reports/resume_manual_eval_seed_predictions.jsonl \
  --load-4bit
```

人岗匹配人工评估集构造与评估：

```bash
bash scripts/data/build_match_eval_dataset.sh

bash scripts/data/run_match_eval.sh \
  --dataset data/eval/match_manual_eval_seed.jsonl \
  --model models/Qwen3-14B \
  --adapter outputs/checkpoints/qwen3-14b-jobmatch-qlora \
  --out outputs/eval_reports/match_eval_report.json \
  --predictions-out outputs/eval_reports/match_eval_predictions.jsonl \
  --load-4bit
```

当前已分层生成：

- `data/eval/resume_manual_eval_text_seed.jsonl`
- `data/eval/resume_manual_eval_ocr_seed.jsonl`

`text_seed` 用于文本简历评估，`ocr_seed` 用于 OCR-like 噪声文本评估。

简历解析 SFT 数据构造：

```bash
bash scripts/data/build_resume_sft_dataset.sh
```

当前输出：

- `data/sft_resume/train.jsonl`: `1920`
- `data/sft_resume/valid.jsonl`: `240`
- `data/sft_resume/test.jsonl`: `240`

这是从人工简历种子集出发，生成多种简历写法后的高质量 bootstrap 集，适合先打通 `resume_parse` 训练链路，不适合被误认为最终规模数据集。

匹配评估与训练数据构造：

```bash
bash scripts/data/build_match_sft_dataset.sh
```

当前输出：

- `data/eval/match_manual_eval_seed.jsonl`: `64`
- `data/eval/match_manual_train_pool.jsonl`: `128`
- `data/sft_match/train.jsonl`: `97`
- `data/sft_match/valid.jsonl`: `12`
- `data/sft_match/test.jsonl`: `19`

这批 `match` 数据目前仍然是高质量人工种子扩写后的 bootstrap 集，适合先打通 `match` 训练和评估链路，不适合被误认为正式规模数据。

Resume 专项增量训练：

```bash
bash scripts/train/train_qwen3_14b_resume_sft.sh
```

简历原始文件接入：

```bash
bash scripts/data/resume_ingest.sh <resume-file-or-dir>
```

如果图片或扫描件已经有人先做了 OCR，可传 sidecar 目录：

```bash
bash scripts/data/resume_ingest.sh <resume-file-or-dir> data/resume_raw/resume_ingest.jsonl <ocr-dir>
```

sidecar 约定文件名支持：

- `<原文件名>.ocr.txt`
- `<文件 stem>.ocr.txt`

自动生成 OCR sidecar：

```bash
bash scripts/data/resume_ocr_sidecar.sh <image-or-pdf-file-or-dir>
```

默认输出到：

- `data/resume_ocr_text/`

简历中间层规范化：

```bash
bash scripts/data/resume_normalize.sh \
  --input data/resume_raw/resume_ingest.jsonl \
  --out data/resume_interim/resume_clean.jsonl \
  --only-parse-ok
```

这一步会把 `resume_raw` 统一整理为可直接喂给 `resume_parse` 的 `normalized_text`。

批量 resume pipeline 评估：

```bash
bash scripts/data/run_resume_pipeline_eval.sh \
  --dataset data/eval/resume_manual_eval_text_seed.jsonl \
  --model models/Qwen3-14B \
  --adapter outputs/checkpoints/qwen3-14b-jobmatch-qlora \
  --out outputs/eval_reports/resume_pipeline_text_report.json \
  --predictions-out outputs/eval_reports/resume_pipeline_text_predictions.jsonl \
  --load-4bit
```

OCR-like 对照评估：

```bash
bash scripts/data/run_resume_pipeline_eval.sh \
  --dataset data/eval/resume_manual_eval_ocr_seed.jsonl \
  --model models/Qwen3-14B \
  --adapter outputs/checkpoints/qwen3-14b-jobmatch-qlora \
  --out outputs/eval_reports/resume_pipeline_ocr_report.json \
  --predictions-out outputs/eval_reports/resume_pipeline_ocr_predictions.jsonl \
  --load-4bit
```

PDF 接入时会进一步区分：

- `text_pdf`
- `weak_text_pdf`
- `scanned_pdf`

其中 `weak_text_pdf / scanned_pdf` 在没有 sidecar OCR 时会被标记为 `needs_ocr=true`，不会直接假装可用于结构化抽取。

## 前后端分离应用

启动后端：

```bash
source /share/home/lifr/miniconda3/etc/profile.d/conda.sh
conda activate tune-demo
cd /share/home/lifr/workspace/code/job-match-tune
bash scripts/serve/start_api.sh
```

如需切到 `vLLM + OpenAI-compatible API + JSON Schema structured outputs`：

```bash
bash scripts/serve/start_vllm_server.sh

export JOBMATCH_INFERENCE_BACKEND=vllm
export JOBMATCH_VLLM_BASE_URL=http://127.0.0.1:8010/v1
export JOBMATCH_VLLM_MODEL=jobmatch-lora
bash scripts/serve/start_api.sh
```

启动前端：

```bash
cd /share/home/lifr/workspace/code/job-match-tune
bash scripts/serve/start_frontend.sh
```

说明：

- 默认从 `5173` 开始找端口
- 如果端口已占用，会自动顺延到下一个可用端口
- 启动时会打印实际访问地址

前端当前结构：

- `frontend/index.html`
- `frontend/src/main.js`
- `frontend/src/App.js`
- `frontend/src/components/`
- `frontend/src/services/`
- `frontend/src/config/`
- `frontend/src/utils/`
- `frontend/src/styles/`

说明：

- 当前前端使用 Vue 3 ESM 浏览器构建版本
- 不依赖 Vite / webpack，本地直接通过静态文件服务启动
- 目录已经按组件、服务、配置、工具和样式拆分，不再是单一 `html/js/css` 三文件结构

端口转发：

```bash
ssh -L 5173:localhost:5173 -L 8000:localhost:8000 gpu03
```

浏览器打开 `http://localhost:5173`。

前端当前支持三个工作模式：

- `JD 解析`
- `简历解析`
- `人岗匹配`
  - 双输入：`JD 文本 + 简历文本`
  - 输出：`规则匹配结果 + 模型分析结论`

前端同时支持两种请求方式：

- `单条`
- `批量`
  - 使用 `---` 分隔多条样本
  - 可直接调用 `/api/batch_parse` 和 `/api/batch_match`

在 `简历解析 -> 单条` 模式下，前端支持直接上传 `txt / docx / pdf / 图片` 简历文件，并调用 `/api/resume_file_parse`。

当前后端接口：

- `POST /api/parse`
  - 支持 `jd_parse`
  - 支持 `resume_parse`
- `POST /api/match`
  - 输入 `jd_text + resume_text`
  - 返回 `jd_parse + resume_parse + rule_result + analysis`
- `POST /api/batch_parse`
  - 输入 `task + texts[]`
  - 返回批量结构化结果和逐条状态
- `POST /api/batch_match`
  - 输入 `items[{jd_text,resume_text}]`
  - 返回批量匹配结果和逐条状态
- `POST /api/resume_file_parse`
  - 输入 `multipart/form-data`
  - 支持 `txt / docx / pdf / png / jpg`
  - 对 `text_pdf` 直接解析
  - 对 `weak_text_pdf / scanned_pdf / image`，可额外传 `ocr_text` 作为 OCR sidecar 文本

如需切回 1.7B：

```bash
export JOBMATCH_MODEL_PATH=models/Qwen3-1.7B
export JOBMATCH_ADAPTER_PATH=outputs/checkpoints/qwen3-1.7b-dft-lr1e-4
bash scripts/serve/start_api.sh
```

## 文档索引

- 技术方案原文：[微调方案.md](/share/home/lifr/workspace/code/job-match-tune/%E5%BE%AE%E8%B0%83%E6%96%B9%E6%A1%88.md)
- 项目阶段路线图：[docs/project_roadmap.md](/share/home/lifr/workspace/code/job-match-tune/docs/project_roadmap.md)
- 项目实现与迭代总览：[docs/implementation_and_evolution.md](/share/home/lifr/workspace/code/job-match-tune/docs/implementation_and_evolution.md)
- 数据处理全流程：[docs/data_pipeline_full.md](/share/home/lifr/workspace/code/job-match-tune/docs/data_pipeline_full.md)
- 三条数据链路说明：[docs/data_tracks_explained.md](/share/home/lifr/workspace/code/job-match-tune/docs/data_tracks_explained.md)
- 简历处理链路：[docs/resume_pipeline.md](/share/home/lifr/workspace/code/job-match-tune/docs/resume_pipeline.md)
- 公开招聘/简历/匹配数据源清单：[docs/public_dataset_inventory.md](/share/home/lifr/workspace/code/job-match-tune/docs/public_dataset_inventory.md)
- 简历写法与项目亮点：[docs/resume_project_highlights.md](/share/home/lifr/workspace/code/job-match-tune/docs/resume_project_highlights.md)
- 数据来源：[docs/data_sources.md](/share/home/lifr/workspace/code/job-match-tune/docs/data_sources.md)
- 字节招聘 API 研究记录：[docs/bytedance_api_research.md](/share/home/lifr/workspace/code/job-match-tune/docs/bytedance_api_research.md)
- 岗位方向标注口径：[docs/job_direction_policy.md](/share/home/lifr/workspace/code/job-match-tune/docs/job_direction_policy.md)
- 历史实验记录：
  - [docs/history/experiment_results_2026-05-11.md](/share/home/lifr/workspace/code/job-match-tune/docs/history/experiment_results_2026-05-11.md)
  - [docs/history/incremental_sft_2026-05-13.md](/share/home/lifr/workspace/code/job-match-tune/docs/history/incremental_sft_2026-05-13.md)
  - [docs/history/manual_eval_2026-05-13.md](/share/home/lifr/workspace/code/job-match-tune/docs/history/manual_eval_2026-05-13.md)
