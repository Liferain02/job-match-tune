我会优先找“能直接服务 JD 解析 / 简历解析 / 人岗匹配”的公开数据，而不是泛泛的中文语料；同时会区分哪些适合主 SFT，哪些只适合弱监督或专项训练。

已思考 1m 34s

有，而且比你现在自己硬爬综合招聘平台更值得优先尝试。但我建议你不要把公开数据集一股脑混进默认 SFT，而是按用途分层：JD 技能抽取、简历解析、人岗匹配、职业/技能词表、弱监督扩量分别用。

最推荐先看的公开数据
1. Chinese-SkillSpan：最适合你项目的中文 JD 技能抽取数据

这是我现在最建议你优先接入的公开数据。它是 2026 年 4 月发布的中文招聘文本技能抽取数据集，包含 2 万多条实例，来自 2014–2025 年四个主流招聘平台，标注目标对齐 ESCO，并把能力分成 knowledge、skill、transversal competence、language competence 四类。作者也明确说这是面向中文招聘文本的 JobSkillNER 数据集。

对你的项目用途：

适合：
- 技能字段抽取
- 技能 span 边界学习
- 技能标准化 / 技能类型分类
- 构造 skill_extract 子任务

不适合直接当作：
- 完整 JD parse 数据
- 岗位方向、学历、经验、职责字段的完整监督数据

你可以把它转换成类似：

{
  "instruction": "从以下招聘描述中抽取技能，并按类型归类。",
  "input": "...招聘文本...",
  "output": {
    "skills": [
      {"name": "Python", "type": "skill"},
      {"name": "机器学习", "type": "knowledge"}
    ]
  }
}

这批数据应该进你的 专项 SFT / multi-task SFT，不要直接混进默认完整 JSON parse 集。

2. Job-SDF：适合做中文技能词表、弱监督和行业覆盖扩展

Job-SDF 是中文劳动力市场方向的数据集，基于 2021–2023 年中国主要招聘平台的 1035 万条公开招聘广告，覆盖 2324 种技能、521 家公司，主要用于技能需求预测和多粒度 benchmark。

它对你很有价值，但不是“拿来就训完整 JD 解析”的数据。

适合：
- 扩充技能词表
- 发现中文招聘高频技能
- 公司 / 行业 / 地区维度的数据分布参考
- 做弱监督标签
- 做技能需求趋势分析

谨慎：
- 如果它释放的是聚合后的 skill demand 数据，而不是完整 JD 原文，就不能直接做完整 SFT

我建议你把 Job-SDF 用作：

1. 生成 skill_aliases.yaml
2. 生成岗位方向 -> 技能先验
3. 校验你规则抽取的技能是否漏掉长尾词
4. 做 weak label，不做 gold label
3. Chinese Resume NER / Resume NER：适合简历解析，不适合 JD 解析

中文 Resume NER 数据在中文 NER 论文里长期被当作 benchmark 使用，例如 Lattice LSTM 和后续中文 NER 工作都会在 MSRA、Weibo、Resume 等数据集上实验。

对你项目用途：

适合：
- 简历中的姓名、组织、学校、职位、时间等实体抽取
- 简历解析模块
- 简历字段边界学习

不适合：
- JD 职责/要求抽取
- JD 岗位方向分类
- 技能需求抽取的主数据

如果你的项目后面要做“JD + 简历匹配”，这类数据很有用；但如果当前主线是 JD 结构化，优先级低于 Chinese-SkillSpan。

4. CLUENER2020：可作为中文 NER 辅助数据，不是招聘专用数据

CLUENER2020 是中文细粒度 NER benchmark，包含 10 类实体，覆盖 person、organization、location 等更真实的中文实体场景。

它不招聘专用，但可以用来增强模型对中文组织名、地址、产品名等实体边界的稳定性。我的建议是：不要混进主 SFT，只做辅助 NER 子任务或预热训练。

5. SkillSpan：英文技能抽取数据，可借鉴 schema 和训练方式

SkillSpan 是英文 job posting 技能抽取数据，包含 14.5K 句子和 12.5K+ span 标注，区分 hard skill 和 soft skill。

对中文项目来说，它不是主训练数据，但有两个用途：

1. 借鉴标注规范：什么算 skill，什么不算 skill
2. 做少量英文 / 中英混合岗位的技能抽取训练

如果你的目标只服务中文岗位，可以暂时不用它；如果未来要支持中英混合 JD，可以加 5%–10%。

6. RJDB：适合人岗匹配，不适合直接提升 JD parse

RJDB，也就是 Resume-Job Description Benchmark，包含超过 5 万组三元组：JD、匹配简历、不匹配简历，面向简历-JD 匹配、解释、技能/经验抽取等 HR 任务。它是用 LLM 和 skill-occupation graph 蒸馏构造的 benchmark。

它适合你项目后半段：

适合：
- JD-简历匹配
- 匹配理由生成
- 简历改写建议
- hard negative 训练

不适合：
- 中文 JD 完整结构化主训练

另外它是合成/蒸馏数据，训练时要防止模型学到过于模板化的解释。

7. JobResQA：适合做 HR 场景评估集

JobResQA 是 2026 年发布的多语言简历 + JD 问答 benchmark，包含 105 组合成简历-JD 对、581 个 QA 对，语言包含中文。

它数据量不大，不适合当主训练集，但非常适合做：

- 中文 HR 场景问答评估
- JD + 简历交叉理解评估
- LLM 是否能根据 JD 和简历回答事实问题

你可以把它放到 eval_hr_qa/，而不是 train/。

职业/技能词表类：建议一定接
ESCO

ESCO 是欧盟维护的多语言职业、技能、资格分类体系，v1.2 包含约 3039 个职业，并按 ISCO-08 分类；它本质上不是训练样本，而是标准化 taxonomy。

对你项目来说，它适合做：

- 技能标准化
- occupation / skill ontology
- 技能 alias 合并
- 中文技能映射到英文标准名

Chinese-SkillSpan 本身也对齐 ESCO，所以你接 Chinese-SkillSpan 时，最好顺手接 ESCO。

O*NET

O*NET 是美国劳工部相关体系下的免费职业信息数据库，包含职业定义、技能、任务、工作活动等信息，并持续更新。

它适合做职业画像和技能先验，但因为是英文和美国职业体系，中文互联网岗位要做映射，不要直接当中文标签标准。

我建议你的接入优先级

按你现在“高质量中文数据不到 2000 条”的情况，我会这样排：

第一优先级：
1. Chinese-SkillSpan
2. Job-SDF
3. 你自己已有的中文官网 JD / Moka / 腾讯 / 百度 / 京东数据

第二优先级：
4. Chinese Resume NER
5. CLUENER2020
6. ESCO / O*NET taxonomy

第三优先级：
7. SkillSpan 英文
8. RJDB
9. JobResQA
不建议直接用的数据

我不建议你把下面这些直接混进训练：

1. Kaggle / GitHub 上来路不明的真实简历数据
   - 很可能有个人隐私、授权不清、脱敏不完整的问题

2. 51job / 智联 / BOSS / 拉勾历史爬虫 dump
   - 版权和平台条款风险高
   - 字段质量不稳定
   - 过期岗位非常多

3. 纯职位标题数据
   - 可以做岗位方向分类
   - 不能训练完整 JD parse

4. 聚合型技能趋势数据
   - 可以做词表和弱监督
   - 不能当完整 JD 标签
对你项目最实用的组合方案

你可以这样组织数据：

data/sft/
- 只放你自己清洗过、可信来源、完整字段的高质量 JD
- 目标：3000–5000 条

data/sft_tasks/skill_extract/
- Chinese-SkillSpan
- SkillSpan 少量英文
- 目标：1–2 万条技能抽取子任务

data/weak/skill_taxonomy/
- Job-SDF
- ESCO
- O*NET
- 用于扩 skill_aliases、岗位方向先验、弱标签

data/eval/
- 自己人工标注 JD eval
- Chinese-SkillSpan holdout
- JobResQA 中文部分
- 边界岗位专项 eval

训练混合比例可以先用：

60% 你的高质量完整 JD parse
25% Chinese-SkillSpan 技能抽取
10% 简历 / HR QA / 人岗匹配辅助任务
5% 英文 SkillSpan / ESCO prompt 数据

最关键的是：Chinese-SkillSpan 和 Job-SDF 是你现在最值得接的两个公开数据源。 前者补“中文招聘技能 span 标注”，后者补“中文招聘技能覆盖和弱监督规模”。你的完整 JD JSON 训练集，还是应该以自己可信来源清洗出来的数据为主。