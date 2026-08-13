# 训练与后训练

本目录解释 SFT、QLoRA、DFT 与 DPO 的顺序、数学含义和执行边界。

1. [SFT、QLoRA、DFT 与 DPO 实现](01_SFT_QLoRA_DFT与DPO实现.md)：训练入口、配置和历史实验。
2. [QLoRA、SFT 与 DFT 从张量到配置逐项推导](02_QLoRA_SFT与DFT从张量到配置逐项推导.md)：矩阵、loss、mask、显存和版本风险。
3. [DPO 偏好样本构造、目标函数与暂停原因](03_DPO偏好样本构造_目标函数与暂停原因.md)：chosen/rejected、DPO公式和恢复条件。

训练失败与复现风险的过程性复盘见[训练资源与版本复现](../06_技术难点与解决过程/05_训练资源与版本复现_从能跑到可信实验.md)。
