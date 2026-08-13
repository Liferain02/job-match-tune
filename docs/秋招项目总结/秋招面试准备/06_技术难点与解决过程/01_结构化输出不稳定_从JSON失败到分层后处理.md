# 结构化输出不稳定：从 JSON 失败到分层后处理

## 1. 难点不是“让模型输出 JSON”这么简单

项目需要模型完成 JD 解析、简历解析和匹配解释。这三类任务都要求稳定字段，因为下游规则和前端不是在阅读自然语言，而是在访问固定键。

最初看起来只要在 Prompt 中写“只输出 JSON”即可，实际会遇到三类完全不同的问题：

```text
语法问题：thinking、Markdown 围栏、中文引号、尾逗号、括号缺失
结构问题：字段类型错误、列表变字符串、职责与要求混放
语义问题：JSON 完全合法，但把加分项技能当成必备技能
```

如果把它们都统计为“JSON 准确率”，既无法定位，也无法选择正确修复层。

## 2. 第一版思路为什么不够

第一版直觉方案是：

```text
模型生成 → json.loads → 成功则返回，失败则报错
```

这个方案有两个问题。

第一，模型可能只因为多输出一段说明或一个尾逗号而失败，用户无法得到本来可以恢复的结果。

第二，更危险的是 `json.loads` 成功会被误认为业务正确。合法 JSON 里的字段仍可能错位，直接交给规则层会产生稳定但错误的分数。

## 3. 如何确认根因不在同一层

调试时保留三个快照：

```text
raw_output：模型原文
raw_data：不修复时严格解析得到的对象
data：修复并归一化后的产品对象
```

对同一失败样本逐层比较：

1. `raw_json_ok=false`：首先是格式遵循问题；
2. `raw_json_ok=true` 但字段错：是结构或语义问题；
3. `data` 正确而 `raw_data` 错：后处理产生了真实纠错；
4. `data` 仍错：需要修 schema、证据规则或训练数据；
5. `rule_result` 正确但 `analysis` 错：属于解释生成，而不是解析。

这一步把“模型不行”拆成了可以写测试的具体问题。

## 4. 最终方案一：生成阶段尽量约束

Transformers 路径采用确定性生成：

```text
enable_thinking=false
do_sample=false
不设置 temperature/top_p 采样
```

vLLM 路径使用 `temperature=0`，并通过 `build_response_format(task)` 传入对应 JSON Schema。

三个 schema 分别约束：

- JDParseResult；
- ResumeParseResult；
- MatchAnalysisResult。

这能降低格式错误，但不能替代后处理，因为不同后端、模型版本和输出截断仍会产生异常。

## 5. 最终方案二：只做有限 JSON 修复

`repair_json_text` 的设计原则不是“猜出任何损坏 JSON”，而是只修历史真实出现、语义风险较低的模式：

1. 删除成对 `<think>...</think>`；
2. 截取第一个 `{` 到最后一个 `}`；
3. 中文引号规范化；
4. 删除对象或数组结束前的尾逗号；
5. 修一个已经观察到的匹配结论尾部错位模式。

没有使用 `eval`，也没有自动补任意字段或任意括号。原因是过度修复会把不可确定的模型错误静默伪装成正确结果。

## 6. 最终方案三：业务字段归一化

JSON 修复解决“能读”，归一化解决“能用”。

简历侧处理：

- 字符串、字典、嵌套列表统一成列表；
- 目标岗位映射到标准方向；
- 技能 alias 和 OCR 形式映射到标准名；
- 优势标签去模板化前后缀；
- 项目按分号切分并补句末标点。

JD 侧处理：

- 从职责中移出“任职要求”“加分项”“学历”“经验”行；
- 从完整上下文补抽取学历和经验；
- 对模型截断的职责只在前缀可信时补回；
- 根据原文章节证据重建必备技能，而不是直接相信模型列表。

## 7. 为什么不把所有规则写进 Prompt

Prompt 仍然负责告诉模型字段语义，但关键约束必须由代码复核：

- Prompt 遵循不是确定性的；
- 业务规则需要回归测试；
- 线上发现新错例时，代码修复比重新训练快；
- 评测需要区分模型能力和产品后处理贡献。

同时也没有把开放式总结全部规则化。匹配建议仍由模型生成，因为它不是一个适合枚举的确定性字段。

## 8. 测试如何防止修复再次退化

结构化输出测试验证三个任务都生成正确 JSON Schema，并检查关键字段存在。

后处理测试覆盖：

- thinking 移除；
- 尾逗号和中文引号；
- resume 字段类型统一；
- JD 职责/要求错位；
- 必备技能证据重建；
- OCR alias 和短词边界。

评测又把结果分为 raw、normalized、product final 三层。这样一次后处理改动即使提高产品结果，也不能掩盖 raw model 是否退化。

## 9. 方案代价

这套设计增加了一层业务代码，也带来维护成本：

- schema 和 alias 需要更新；
- 规则可能只覆盖已知错误模式；
- normalized 指标可能明显高于 raw，需要诚实解释；
- 新模型后端仍要验证 response format 兼容性。

但相比“模型直接生成最终结果”，换来的是可定位、可测试和可回归。

## 10. 仍未解决的边界

- 未闭合 thinking 标签不能稳定恢复；
- 多个 JSON 对象同时出现时不会猜正确对象；
- 无章节、无换行的长 JD，职责与要求边界仍弱；
- 字段全缺但 JSON 合法时，任务类型启发式可能误判；
- 在线接口尚未强制阻断所有 explanation grounding 问题。

这些问题需要真实失败样本驱动，而不是继续堆通用修复器。

## 11. 面试回答结构

> 结构化输出的难点不只是 JSON 语法，而是语法、字段结构和业务语义三层错误。我最初如果只用 `json.loads`，既会拒绝可恢复输出，也会把合法但语义错误的 JSON 当正确。后来保留 raw output 和 strict parse，只做有限语法修复，再做任务字段归一化；JD 必备技能还会回到原文章节证据重建。评测分别报告 raw、normalized 和 product final，因此后处理贡献可见，也不会把产品指标冒充模型原生能力。

## 12. 对应代码和验证

```text
src/jobmatch_tune/inference/structured_output.py
src/jobmatch_tune/inference/postprocess_json.py
src/jobmatch_tune/preprocess/jd_skill_evidence.py
tests/test_structured_output.py
tests/test_postprocess_json.py
tests/test_jd_skill_evidence.py
```
