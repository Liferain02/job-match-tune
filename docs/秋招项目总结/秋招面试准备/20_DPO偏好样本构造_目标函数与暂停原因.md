# DPO 偏好样本构造、目标函数与暂停原因

DPO 在这个项目中是 SFT 之后的可选阶段，不是当前必须执行的下一步。代码路径完整不等于数据已准备好，本章把二者分开。

## 1. 训练顺序的代码证据

DPO 配置中的：

```yaml
adapter_path: outputs/checkpoints/qwen3-14b-jobmatch-dft-20260601
```

说明策略模型从已有 SFT/DFT adapter 继续训练，而不是从裸基座直接开始。典型顺序是：

```text
Base model
  → SFT/DFT：学任务、字段和基本输出格式
  → DPO：在相同 prompt 下偏好 chosen 而非 rejected
```

若模型连 JSON 和字段都不会，DPO pair 很难高效教会完整任务，因此先 SFT 是当前项目的合理顺序。

## 2. 一条 DPO 数据长什么样

```json
{
  "id": "sample_direction_mismatch",
  "task_type": "jd_parse",
  "prompt": [
    {"role": "system", "content": "……"},
    {"role": "user", "content": "解析以下 JD……"}
  ],
  "chosen": [
    {"role": "assistant", "content": "{正确 JSON}"}
  ],
  "rejected": [
    {"role": "assistant", "content": "{方向错误的 JSON}"}
  ],
  "meta": {
    "provenance": "synthetic_structured_hard_negative",
    "rejection_strategy": "direction_mismatch"
  }
}
```

DPO 学的是同一 prompt 下两个完整回答的相对偏好，不是给单个回答打标签。

## 3. prediction mismatch 偏好数据

`build_preference_dataset.py` 从评测 prediction 构造：

```text
chosen   = gold label 序列化后的 JSON
rejected = 模型 parsed 结果；若无 parsed 则用 raw prediction
```

若 chosen 与 rejected 相同或 rejected 为空，该行被跳过。其价值是错误来自真实模型分布，能针对现有失败。

风险：如果 Gold 已被开发者看过并反复修规则，偏好数据会混入回归污染；普通随机行切分也可能让同一实体跨 train/valid。

## 4. synthetic hard negative 如何构造

bootstrap builder 从正确 SFT answer 做一种确定性破坏：

### JD 任务

- 添加未约定字段；
- 改错岗位方向；
- 删除一条职责；
- 把职责泄漏到技能；
- 把学历混入经验。

### Resume 任务

- 添加未约定字段；
- 删除优势；
- 删除项目；
- 把教育混入技能。

### Match explanation 任务

- 添加未约定字段；
- 删除短板；
- 交换优势与短板；
- 删除建议。

`_stable_index` 对 row id 做 SHA-1，再决定从哪种破坏开始；如果该破坏不适用，就循环尝试下一种。相同 row id 每次得到相同 rejected，保证构建可复现。

## 5. synthetic negative 的价值和局限

价值：

- 低成本构造明确错误边界；
- 覆盖字段污染、遗漏和结构错误；
- chosen/rejected 差异可解释；
- 可用于训练流程 smoke test。

局限：

- rejected 过于规则化，模型可能只学习表面模式；
- 不是人类真实偏好；
- 很少覆盖“两个回答都合理但一个更有帮助”的细粒度偏好；
- match negative 多为删字段/交换字段，未证明建议能提高投递效果。

因此它叫 bootstrap preference，不应包装为 RLHF 人类偏好数据。

## 6. DPO 目标函数

对 prompt `x`、chosen `y_w`、rejected `y_l`，定义策略与参考模型的 log-ratio 差：

```text
Δθ = [log πθ(y_w|x) - log πθ(y_l|x)]
     - [log πref(y_w|x) - log πref(y_l|x)]
```

标准 sigmoid DPO loss：

```text
L_DPO = -log σ(β Δθ)
```

直觉：策略相对参考模型，应更提高 chosen 的相对概率，而不是无约束地漂离初始策略。

项目 `beta=0.1`。β 控制偏好 logit 的尺度/相对参考约束强度，其实际效果要靠 chosen reward、rejected reward、margin 和下游评测比较，不能只看 loss。

## 7. `ref_model=None` 是什么意思

代码把 `ref_model=None` 交给 TRL `DPOTrainer`。在 PEFT 场景下，Trainer 可按其版本策略使用参考行为，例如禁用 adapter 得到参考策略，避免额外常驻一份完整模型。

这个行为是依赖版本的。仓库没有自己实现 reference forward，因此正式运行前必须在目标 TRL 版本确认日志和显存行为，不能笼统说“完全不需要参考模型”。

## 8. DPO 中 LoRA 的两种分支

### 已有 adapter_path

```python
PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
```

继续更新 SFT adapter，`peft_config=None`。

### 没有 adapter_path

创建新的 `LoraConfig`，由 Trainer 包装基础模型。

当前配置走第一种，所以它确实是 SFT adapter 之后继续对齐。

## 9. DPO 配置为何更保守

```text
learning rate: 5e-6   比 SFT 的 1e-4 小 20 倍
epoch: 1
max length: 1024
micro batch: 1
accumulation: 16
beta: 0.1
```

偏好优化容易破坏原有任务能力，小学习率和单 epoch 是保守选择。长度 1024 需要容纳 prompt + chosen/rejected；但仍应统计截断率。

## 10. 为什么当前 DPO 被暂停

Readiness 报告同时出现：

```text
ready_for_dpo = true
dpo_paused_by_quality_goal = true
dpo_execution_ready = false
dpo_pipeline_fresh = false
```

前两个“ready”主要说明文件结构或旧门槛满足，不代表现在应该执行。真正执行结论看 `dpo_execution_ready`：当前 preference 产物早于上游新 SFT 数据，管线 stale；同时质量目标要求真实 preference 证据，而现有主要是 synthetic hard negative。

所以正确决策是暂停，不是看到脚本能跑就烧 GPU。

## 11. 启动 DPO 前的最小门槛

1. 重建 preference，使它与最新多任务 SFT 对齐；
2. preference train/valid 做实体级切分审计；
3. 抽样人工检查 chosen 确实优于 rejected；
4. 统计 rejection strategy 分布，避免单一破坏占主导；
5. 使用新 blind holdout，而不是历史已检查 Gold 选 checkpoint；
6. 与 SFT adapter 做严格 A/B；
7. 检查 JD、resume、match 三任务是否出现遗忘；
8. 只有结构/语义指标改善且无明显回退才保留 DPO adapter。

## 12. 什么叫人工 holdout

`holdout` 是训练和调参期间不参与学习、不用于挑规则的一组保留样本。人工 holdout 指这些样本的标签由人审核，而不是由当前规则或模型自动生成。

真正有效还需要：

- 标注前定义口径；
- 与训练实体隔离；
- 开发期间不反复查看答案；
- 最好多人复核或记录争议；
- 只在预先约定的最终比较时打开。

如果看完 holdout 错例后修改系统，它就从 blind holdout 变成历史回归集，之后仍有价值，但不能再证明新的泛化。

## 13. 如何做最小 DPO 实验

不需要一次全量训练。先固定：

```text
Base/SFT checkpoint 相同
训练样本小子集相同
seed 相同
推理参数相同
```

比较：

```text
A: SFT adapter
B: SFT + DPO adapter
```

观察：strict JSON、字段指标、Gold 回归、blind boundary、解释 grounding、任务间回退。若 B 只在 synthetic preference valid loss 上好看而产品指标无提升，应停止。

## 14. 面试口述模板

> DPO 在项目里是 SFT/DFT adapter 后的可选步骤。chosen 可以来自人工 Gold，rejected 可以来自模型真实错例；目前更多数据是通过删除职责、改错方向、字段泄漏等方式合成的 hard negative，它适合验证流程但不等于人类偏好。标准目标是让策略相对参考模型提高 chosen 对 rejected 的 log-ratio。当前 preference 管线已 stale，且缺少新的人工 blind holdout，所以 readiness 明确把 DPO execution 设为 false，我选择暂停训练而不是为了完整技术栈强行运行。

