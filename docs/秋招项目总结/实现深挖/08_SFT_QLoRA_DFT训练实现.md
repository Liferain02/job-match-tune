# SFT、QLoRA、DFT 训练实现

## 训练对象与入口

SFT 教模型遵循三类结构化任务格式；QLoRA 用 4-bit 基座加低秩适配器降低显存；配置中的 DFT 是训练损失/权重选择，不是另一条产品链。入口为 `train/train_lora.py::main`，脚本包括 `train_qwen3_14b_{smoke,full,multitask_sft,resume_sft}.sh`，核心配置在 `configs/train_qwen3_14b_qlora.yaml`。

## 参数与执行步骤

配置提供 model、train/valid 数据、输出目录、max sequence length、batch、gradient accumulation、learning rate、epoch、LoRA rank/alpha/dropout、4-bit 量化和 `loss_type`。脚本先执行 `_training_readiness_gate.sh`；门禁失败就停止，不加载大模型。

通过后，训练器读取 messages，tokenizer 应用 chat template，assistant 部分作为目标；PEFT 把 LoRA 注入目标线性层，bitsandbytes 负责量化基座，TRL/Transformers 执行梯度累积、评估和保存 adapter。`train/run_manifest.py::write_run_manifest` 记录 Git、配置、数据 SHA256、行数、readiness 和运行事实。

输入样本形如 system/user/assistant 三段 messages，输出是 adapter、tokenizer/配置、checkpoint、日志和 manifest；推理侧通过 `ADAPTER_PATH` 消费 adapter。

## 失败、验证和限制

- readiness、文件不存在、CUDA/量化库不兼容应在训练前或初始化时失败，不静默退回另一数据集。
- smoke 只验证链路，不证明模型效果；是否晋级必须看独立 Gold 的 Raw、Normalized、Product Final。
- 测试覆盖配置和 manifest（`tests/test_run_manifest.py` 等），但训练本身不在单元测试执行。
- 本轮明确不训练：当前 Resume/Match/Multitask 数据门禁未通过。历史 adapter 可复现推理，但不能据此宣称当前数据已 ready。
