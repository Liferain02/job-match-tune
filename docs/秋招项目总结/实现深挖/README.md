# 实现深挖索引

更新时间：2026-08-13

本目录面向代码审查和面试追问。内容以当前实现为准，不把规划写成已完成能力，也不把启发式匹配分解释为录用概率。建议按以下顺序阅读：

1. [请求生命周期与完整调用链](01_请求生命周期与完整调用链.md)
2. [JD 解析实现逐层拆解](02_JD解析实现逐层拆解.md)
3. [Resume 解析、文件、OCR、隐私实现](03_Resume解析_文件_OCR_隐私实现.md)
4. [Normalization 与证据约束实现](04_Normalization与证据约束实现.md)
5. [Match 规则与 Direction 兼容实现](05_Match规则与Direction兼容实现.md)
6. [技能 Canonicalization 与 OCR 实现](06_技能Canonicalization与OCR实现.md)
7. [数据构建、SourceGroup、EntitySplit](07_数据构建_SourceGroup_EntitySplit.md)
8. [SFT、QLoRA、DFT 训练实现](08_SFT_QLoRA_DFT训练实现.md)
9. [DPO、Preference 与暂停门禁](09_DPO_Preference与暂停门禁.md)
10. [Transformers 与 vLLM 推理实现](10_Transformers与vLLM推理实现.md)
11. [Gold 评测与三层 Evaluator 实现](11_Gold评测与三层Evaluator实现.md)
12. [Readiness、Source Admission、隐私、泄漏](12_Readiness_SourceAdmission_隐私_泄漏.md)
13. [FastAPI、文件上传、批量、前端](13_FastAPI_文件上传_批量_前端.md)
14. [从一个 Case 贯穿整个系统](14_从一个Case贯穿整个系统.md)
15. [踩坑到代码修复映射](15_踩坑到代码修复映射.md)
16. [面试深挖追问链](16_面试深挖追问链.md)

阅读代码时优先核对文中给出的路径和函数。历史实验数字只用于复盘；能否继续训练，以当前 readiness 报告和训练门禁为准。
