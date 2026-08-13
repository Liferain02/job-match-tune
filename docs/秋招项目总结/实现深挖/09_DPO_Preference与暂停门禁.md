# DPO、Preference 与暂停门禁

## DPO 在流程中的位置

DPO 通常位于 SFT 之后：它需要一个已经能完成任务的 policy/reference，再用同一 Prompt 下的 chosen/rejected 学输出偏好。它不是“补救缺数据”的步骤，也不能替代 SFT 的格式学习。

## 数据和训练实现

- `build_preference_dataset.py::build_preference_row` 从人工/模型对比结果构建 prompt、chosen、rejected；Match preference 还注入冻结规则事实。
- `build_preference_bootstrap_dataset.py` 能制造缺字段、串字段、方向错误等 rejected，用于链路验证，但这种合成偏好不等于真实人工偏好。
- `split_rows` 及 `report_preference_readiness.py::audit_preference_files` 检查 prompt hash 跨 train/valid/holdout 重叠。
- `train/train_dpo.py::main` 读取 `configs/train_qwen3_14b_dpo.yaml`；脚本先经过 `_training_readiness_gate.sh` 和 `_dpo_pause_gate.sh`。

`chosen/rejected` 必须针对完全相同的 Prompt，chosen 有可验证证据且优于 rejected。输出仍是 adapter、checkpoint 和 manifest，由同一推理服务加载。

## “人工 holdout”是什么

人工 holdout 是在训练和调参前就隔离的一组人工复核样本。训练、DPO pair 构建、规则修改和阈值选择都不能读取其标签；只有方案冻结后用于一次性验收。它不是普通 validation，也不是“从训练数据随机留 10%”。这里门禁还要求来源/实体不与训练集重叠。

## 为什么当前暂停

Preference readiness 会核对偏好来源、人工复核、独立 holdout、样本量和泄漏。当前真实人工 Pair 与独立偏好证据不足，所以暂停是正确结果。失败时脚本明确退出，不自动改用 bootstrap。

测试覆盖 `tests/test_build_preference_dataset.py`、`test_build_preference_bootstrap_dataset.py`、`test_report_preference_readiness.py`。边界是：形式正确的 preference 仍可能语义错误；只有人工证据和独立 Gold 能证明收益。
