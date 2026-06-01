# 公开招聘/简历/匹配数据源清单

更新时间：2026-05-18

## JD / 招聘数据

1. `edwarddgao/open-apply-jobs`
   - 类型：公开招聘职位聚合
   - 规模：百万级
   - 当前状态：已接入部分技术岗位分片
   - 适用：raw job pool 扩量，不直接作为默认高质量 `JD parse` 集
   - 链接：<https://huggingface.co/datasets/edwarddgao/open-apply-jobs>

2. `wangzihaogithub/job-educational-parser-dataset-08-0-0805`
   - 类型：中文岗位描述 -> 学历要求
   - 规模：288,627
   - 当前状态：已作为弱监督中文岗位池的一部分使用
   - 适用：学历字段专项监督、弱标注补量
   - 链接：<https://huggingface.co/datasets/wangzihaogithub/job-educational-parser-dataset-08-0-0805>

3. `Job-SDF`
   - 类型：中国招聘技能需求多粒度数据集
   - 规模：基于 10.35M 公共招聘广告构建
   - 当前状态：可作为技能词表/趋势研究参考，不直接适配当前 JSON parse 任务
   - 链接：<https://job-sdf.github.io/>

## Resume / 简历数据

1. `OhMyKing/FairCV`
   - 类型：中文模拟简历
   - 规模：100K<n<1M，仓库存储约 6.86 GB
   - 当前状态：已接入轻量采样入口，当前拉取 1000 条进入 resume 候选池
   - 风险：字段口径和当前 schema 需要映射，且包含偏见研究变量
   - 链接：<https://huggingface.co/datasets/OhMyKing/FairCV>

2. `PassbyGrocer/resume-ner`
   - 类型：中文简历 NER
   - 规模：train 3821 / validation 463 / test 477
   - 当前状态：已下载 train parquet，共 3821 条；进入外部语料审计，不直接混入 resume_parse SFT
   - 风险：是 token classification，不是完整 JSON 结构化标签
   - 链接：<https://huggingface.co/datasets/PassbyGrocer/resume-ner>

## JD-Resume Match / 匹配数据

1. `med2425/resume-job-fit-merged-v1`
   - 类型：简历-JD 匹配分类
   - 规模：93,733
   - 当前状态：已确认可用，并已补本地导入器入口，但主要是英文数据
   - 适用：弱监督匹配分类、匹配任务结构参考
   - 风险：不适合作为默认中文人岗匹配训练集
   - 链接：<https://huggingface.co/datasets/med2425/resume-job-fit-merged-v1>

2. `cnamuangtoun/resume-job-description-fit`
   - 类型：简历-JD 匹配
   - 当前状态：可作为后续英文弱监督匹配补充来源，已补本地导入器入口
   - 链接：<https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit>

## 当前仓库中的导入入口

- Resume：
  - [download_public_resume_samples.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/dataset/download_public_resume_samples.py)
  - [download_public_resume_samples.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/data/download_public_resume_samples.sh)
  - [configs/public_resume_sources.yaml](/share/home/lifr/workspace/code/job-match-tune/configs/public_resume_sources.yaml)
  - [import_public_resume_data.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/dataset/import_public_resume_data.py)
  - [import_public_resume_exports.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/data/import_public_resume_exports.sh)
- Match：
  - [configs/public_match_sources.yaml](/share/home/lifr/workspace/code/job-match-tune/configs/public_match_sources.yaml)
  - [import_public_match_data.py](/share/home/lifr/workspace/code/job-match-tune/src/jobmatch_tune/dataset/import_public_match_data.py)
  - [import_public_match_exports.sh](/share/home/lifr/workspace/code/job-match-tune/scripts/data/import_public_match_exports.sh)

## 当前取舍

- 默认高质量训练仍然优先：
  - 中文公开官网 JD
  - 可控人工/模板 resume
  - 可控配对 match 数据
- 外部公开数据集当前主要用于：
  - 扩技能/字段边界
  - 扩简历样式覆盖
  - 扩匹配任务弱监督或对照实验
- 在没有完成 schema 映射和质量审查前，不直接混入默认 `SFT` 主集
