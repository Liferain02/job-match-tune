# 技能证据与 OCR 噪声：从误召回到可追溯判断

## 1. 两种相反错误同时存在

简历和 JD 经过 PDF/OCR 后可能出现：

```text
Py thon
Kubernet es
C + +
My SOL
```

如果只做精确匹配，会漏召回；如果简单删除所有空格再做子串匹配，又会误召回：

```text
C 命中 CUDA、C++、C# 或 C语言
R 命中普通英文单词
Go 命中更长 token 的一部分
pytest 命中 pytest-like 复合词
```

这不是单纯调一个 fuzzy threshold 能解决的问题，因为短词、符号技能和长技术名的风险不同。

## 2. 早期简单 alias 方案的问题

最容易想到的是为每种噪声手写 alias：

```yaml
Python: [Py thon]
Kubernetes: [Kubernet es]
```

它能修当前例子，但有三个问题：

- OCR 断点位置组合太多，词表会不断膨胀；
- alias 只解决已知写法，不形成一般边界；
- 短词若也做同样扩展，会显著增加假阳性。

因此最终没有把 OCR 问题完全变成词表维护问题。

## 3. 第一层：Unicode 与标准词表

`_text` 先做 NFKC，再压缩空白。NFKC 处理全角半角等兼容形式，避免视觉相同字符产生不同 key。

schema 为每个 canonical skill 保存别名。构建两张表：

```text
exact：标准化字符串 → canonical name
compact：去空格/连字符 key → 唯一 canonical name
```

compact key 只有在该 key 唯一对应一个 canonical name 时才保留，避免两个技能压缩后冲突却被任意选择。

## 4. 第二层：精确候选边界

对于英文开头和结尾的技能，正则增加 ASCII 字母数字边界：

```text
左侧不能紧邻 A-Z/a-z/0-9
右侧不能紧邻 A-Z/a-z/0-9/连字符
```

这避免 Python 命中 `CPythonX` 一类更长字符串。

`C` 被单独处理：右侧如果是 `+`、`#` 或“语言”，不算独立 C 技能。这样 C、C++、C# 和 C语言不会被双重计数。

## 5. 第三层：OCR 弹性匹配门槛

不是所有技能都允许字符之间插入空格或连字符。`_allows_ocr_flex` 要求：

```text
压缩 key 长度至少为 4
或者技能包含 +、#、. 等技术符号
```

所以 Python、Pytest、C++ 可以生成 OCR pattern；C、R、Go 等短词不会无条件放宽。

OCR pattern 允许字符间出现 0～3 个空格或连字符，同时仍检查左右边界。

## 6. 为什么没有使用通用编辑距离

Levenshtein 或 embedding 相似度能提高召回，但这里是事实技能匹配：

- Java 与 JavaScript 字符相似但能力不同；
- React 与 React Native 有包含关系但不完全等价；
- PyTorch 与 TensorFlow 可迁移但不能算已掌握；
- 短技能的一个字符差异比例很大。

当前选择“已知词表 + 有界 OCR 容错”，优先保证硬匹配可解释。可迁移能力可以放到建议层，不混入命中技能。

## 7. 更难的问题：技能出现不等于技能必备

即使识别出 Kubernetes，还要判断它出现在哪个语义位置：

```text
任职要求：熟悉 Kubernetes      → requirement evidence
岗位职责：使用 Kubernetes      → responsibility evidence
加分项：了解 Kubernetes        → bonus evidence
公司平台基于 Kubernetes        → other context evidence
```

`SkillEvidence.required` 只有在存在 requirement evidence 时为真。职责和上下文出现的技术不会自动成为候选人硬门槛。

## 8. 章节与句内 cue 的冲突怎么处理

优先级是章节高于句内词语。

例如：

```text
加分项：具备 Kubernetes 使用经验
```

虽然“具备”“经验”是 requirement cue，但明确处于加分项章节，所以仍归为 bonus。

没有章节时才根据“熟悉/掌握/具备/要求”和“负责/参与/开发/维护”等词判断。

## 9. 根因定位的真实顺序

当用户说“技能判错”时，不直接修改 alias：

1. 原文是否真的含该技能；
2. `extract_known_skills` 是否识别；
3. 识别使用 exact 还是 OCR pattern；
4. canonical name 是否唯一；
5. 该行属于哪个章节；
6. `SkillEvidence` 有哪些证据来源；
7. `required_skills_from_evidence` 是否取对；
8. 简历集合是否识别同一 canonical name；
9. 最终错在漏召回、误召回还是 required 角色错判。

## 10. 测试矩阵

至少同时包含正例和反例：

```text
Python     ↔ Py thon             应命中
Kubernetes ↔ Kubernet es         应命中
C++        ↔ C + +               应命中
C          ↔ CUDA                不应命中
C          ↔ C++                 不应重复命中
Pytest     ↔ pytest-like         不应命中
职责 Agent                        不应成为必备
要求 Agent                        应成为必备
加分项 Kafka                      不应成为必备
```

`tests/test_jd_skill_evidence.py` 还验证同一技能同时出现在职责和要求时，两种证据都保留且最终为 required。

## 11. 当前代价与边界

- 遍历整个技能词表做正则，词表极大时性能会下降；
- schema 外的新技能在 JD 侧可能漏召回；
- 完全粘连且字符被识错的 OCR 仍处理不了；
- “A 或 B 至少一项”无法由当前二元 required 表达；
- alias 更新需要人工审计，不能自动吸收模型输出。

当前词表规模和请求量下，准确可解释比引入复杂 NER/向量检索更重要。

## 12. 面试回答结构

> OCR 技能匹配同时有漏召回和误召回风险。最初逐个加 `Py thon` alias 只能修个例。我后来把它拆成 NFKC 标准化、唯一 compact key、英文 token 边界和有门槛的 OCR 弹性匹配；长度不足四的短词不会放宽，C 还单独排除 C++、C# 和 C语言。识别之后再按任职要求、职责、加分项和其他上下文保存证据，只有 requirement evidence 才进入硬匹配。这样每次命中不仅有标准名，还有原文片段和语义角色。

## 13. 对应代码和验证

```text
src/jobmatch_tune/preprocess/skill_canonicalization.py
src/jobmatch_tune/preprocess/jd_skill_evidence.py
configs/label_schema.yaml
tests/test_jd_skill_evidence.py
tests/test_postprocess_json.py
```
