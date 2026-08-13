# SFT、QLoRA、DFT 与 DPO 实现

## 一、训练在项目中的作用

训练目标不是让模型学习真实录用概率，而是提升三个结构化任务的基础能力：

1. `jd_parse`：从 JD 抽取方向、职责、要求、技能、学历和经验。
2. `resume_parse`：从简历抽取目标岗位、教育、技能、实习、项目和优势。
3. `match`：根据 JD、Resume 和确定性规则事实生成结构化解释。

最终 Match 等级仍由规则计算。SFT 改善解析和解释能力，不直接替代评分策略。

## 二、为什么选择 Qwen3-14B

项目早期从更小模型验证链路，后来使用 Qwen3-14B 作为默认基座。选择依据主要是中文能力、指令遵循、结构化生成和本地可用性，而不是因为14B本身构成项目创新。

14B 的代价是显存和延迟，因此训练采用 QLoRA，推理默认4-bit加载；当前历史运行环境是3张 NVIDIA L20，但正式配置使用 `device_map="auto"`，没有手工写死层分配。

面试时不要说“14B一定优于所有小模型”，因为项目没有完成当前 Gold 上的严格基座规模 A/B。准确说法是：14B 是历史工程选择，核心贡献在数据和评测闭环。

## 三、SFT 数据格式

训练数据采用 conversational messages：system、user、assistant。`train_lora.py` 的 `formatting_func` 调 tokenizer chat template，关闭 generation prompt，并根据配置决定是否启用 thinking。

`SFTConfig` 设置 `assistant_only_loss=True`。原因是结构化任务只需要监督 assistant JSON；若对 system/user token 同样计算目标，会浪费优化容量并鼓励模型复述输入。

## 四、QLoRA 细节

配置文件是 `configs/train_qwen3_14b_qlora.yaml`。

### 4-bit 基座

`BitsAndBytesConfig` 使用：

- `load_in_4bit=true`；
- `bnb_4bit_quant_type=nf4`；
- compute dtype `bfloat16`；
- double quantization。

基座权重以量化形式加载，训练的主要参数是 LoRA adapter。这样显著降低14B训练显存，但不等于模型所有计算都变成4-bit；计算仍使用 BF16 等较高精度。

### LoRA 注入

当前参数：

- rank `r=8`；
- alpha `16`；
- dropout `0.05`；
- target modules：`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`。

同时覆盖 attention 和 MLP 投影，提供足够适配容量。项目没有进行大规模 rank 搜索，因此不能声称 r=8 是全局最优，只能说它是显存、训练速度和任务规模下的稳定配置。

### 训练参数

- sequence length 768；
- per-device batch 1；
- gradient accumulation 16；
- 1 epoch；
- learning rate `1e-4`；
- cosine scheduler；
- BF16；
- gradient checkpointing；
- seed 42；
- eval 最多320条。

有效优化 batch 不能只看 per-device=1，还要结合梯度累积和实际设备并行方式。配置没有显式 DeepSpeed/FSDP，因此不要声称使用了复杂分布式训练框架。

## 五、为什么 packing=false

短结构样本通常适合 packing，可以减少 padding。但当前 attention 是 SDPA，历史 TRL 警告 packing 在不受支持的 attention 实现下可能造成样本间信息串扰。

因此正式配置保持 `packing=false`。这是一种正确性优先的选择：除非安装并验证兼容的 FlashAttention 和边界 mask，否则不为吞吐打开有污染风险的优化。

## 六、DFT 是什么、项目做了什么

配置通过 TRL `SFTConfig(loss_type="dft")` 启用 DFT。项目没有自行实现一种新的损失函数，而是使用安装版本 TRL 提供的 loss option；其具体数学语义应以对应 TRL 版本为准。

面试中最安全的表达：

> 我把 DFT 作为 SFT loss 对照项接入，历史上与 NLL 使用相同9,800条 train、320条 eval、1 epoch和613个 optimizer steps 比较。两种 loss 定义不同，所以不直接比较 loss 数字，而看结构化任务和产品评测。

历史耗时约8,050秒与8,029秒，结果接近。这个实验说明工程链路能支持 loss 对照，不足以证明 DFT 对所有任务显著更好。

## 七、SFT 训练代码

`src/jobmatch_tune/train/train_lora.py::main` 的执行顺序：

1. 解析 YAML 和 CLI override。
2. 在 heavy import 前完成基本配置读取，使数据工具无需加载训练依赖。
3. 写 `run_manifest.json`。
4. 加载 tokenizer，并在无 pad token 时使用 eos。
5. 构造4-bit config。
6. 加载模型；若传入 adapter，则以可训练方式继续增量 SFT，否则创建新 LoRA。
7. 用 Hugging Face Datasets 加载 train/validation JSONL。
8. 按 max samples 固定裁剪评测集。
9. 创建 `SFTConfig` 与 `SFTTrainer`。
10. 支持 `resume_from_checkpoint`。
11. 落盘 train/eval metrics、trainer state 和 adapter。

### Run Manifest

`train/run_manifest.py` 记录：

- stage；
- Git commit、branch、dirty status；
- config 路径与 SHA256；
- train/valid 文件、行数、hash、任务分布、source group；
- readiness report hash 和 summary；
- CLI overrides。

这解决“一个 checkpoint 到底由哪份数据和代码产生”的追溯问题。它不保证历史所有环境字节级复现，因为 CUDA/PyTorch wheel 仍依赖宿主机。

## 八、DPO 在 SFT 后面的原因

SFT 先让模型学会任务和输出格式；DPO 再让 policy 在同一个 Prompt 下偏好 chosen 而不是 rejected。若模型连合法 JSON 都不会，直接做 DPO 会把格式学习和偏好学习混在一起。

DPO 也不是强化学习在线交互。当前使用 TRL `DPOTrainer`，数据是静态 chosen/rejected pair，优化目标通过 reference policy 的相对 log probability 表达偏好。

## 九、DPO 配置与代码

`configs/train_qwen3_14b_dpo.yaml` 使用：

- 基座 Qwen3-14B；
- 从历史 SFT adapter 继续；
- learning rate `5e-6`，显著低于 SFT；
- beta `0.1`；
- max length 1024；
- batch 1、accumulation 16；
- 1 epoch；
- eval 最多128条；
- 同样4-bit NF4、BF16和 gradient checkpointing。

`train_dpo.py` 加载基座后，把 SFT adapter 设为 trainable；`DPOTrainer(ref_model=None)` 按 PEFT/TRL 路径处理 reference。训练后同样保存指标和 adapter。

### 历史 DPO 结果

历史实验使用4,400条对话偏好、1 epoch、275 steps，末期 train loss 0.2099、eval loss 0.1749、reward accuracy 0.9297、margin 3.3643。

这些只说明模型能区分构造偏好，不能证明用户匹配结果变好。旧64条 Match 评测来自16个基础样例变体且基础样例进入过训练池，已经失效。

## 十、为什么当前 DPO 暂停

训练脚本首先 source `_dpo_pause_gate.sh`。除非显式设置 `JOBMATCH_ALLOW_DPO=1`，脚本以退出码2停止。之后还要通过对应 readiness stage。

当前暂停有三层原因：

1. preference 多为结构化 synthetic negative，不是足量 human preference；
2. preference pipeline 已 stale；
3. Resume、Match、Multitask readiness=false，继续产品 DPO 缺少可信上游基础。

即使技术上可以绕过 gate，也不代表应该运行。面试中应把 gate 解释为实验治理，而不是“代码跑不起来”。

## 十一、人工 holdout 的准确含义

人工 holdout 是在训练、规则调参和模型选择前隔离的人工复核集合。开发者不能根据其中错误反复修改后仍称其为 blind test。

普通 validation 用于调参和早停；holdout 用于方案冻结后的少量验收；Gold 还要求标签定义、审计和版本冻结。随机从训练数据留10%不自动成为人工 holdout。

## 十二、训练如何晋级

新 adapter 不能因为 loss 下降就设为默认。产品晋级至少需要：

1. 数据 Readiness 通过；
2. run manifest 完整；
3. JD/Resume/Match 绝对产品阈值通过；
4. 相对默认 adapter 的关键指标回退不超过阈值；
5. 文件/API/UI 主流程 smoke；
6. 新错误分析没有严重语义回退。

历史默认最大允许回退为0.005。当前因为数据门禁未通过，不进入这条晋级流程。

## 十三、训练方面最重要的反思

- loss 是优化信号，不是产品指标。
- 格式增强是 robustness，不是来源多样性。
- 合成 rejected 是工程起点，不是真实偏好。
- packing、Liger、FlashAttention 等优化应在正确性确认后开启。
- 配置可运行不等于数据允许训练。
- 没有独立业务评测时，继续训练可能只是在拟合规则或已知 Gold。

## 十四、常见追问简答

为什么不用全量微调？14B 全参成本高，当前任务更偏格式与领域适配，QLoRA 足以建立基线；没有证据证明全参收益值得成本。  
为什么1 epoch？当前样本模板重复较多，更多 epoch 增加过拟合风险；通过独立评测决定是否增加。  
为什么学习率不同？DPO 从已可用 SFT adapter 出发，目标是小步调整偏好，因此使用更低学习率。  
为什么不继续调 rank？当前瓶颈是数据真实性而非 adapter 容量，调参无法补真实 Pair。  
为什么不用多卡框架？历史环境可由 device map 使用多 GPU，但项目没有引入 DeepSpeed/FSDP；当前重点不是规模化训练基础设施。
