# Transformers 与 vLLM 推理实现

## 两个后端的定位

Transformers 是默认、依赖最少的单机验证路径，适合低并发和调试；vLLM 面向并发、连续批处理和稳定服务吞吐。哪个“更好”必须在同一模型、adapter、Prompt、输入集、token 上限和硬件上同时比较质量与性能，不能只凭框架名判断。

## 调用与参数

`api/server.py::ModelService` 读取 `INFERENCE_BACKEND`、模型/adapter 路径、生成参数和并发限制。Transformers 通过 `load_model` 加载 tokenizer、基座和 PEFT adapter，`predict_loaded` 调 `generate`；为避免共享模型并发问题使用全局生成锁。

vLLM 路径由 `scripts/serve/start_vllm_server.sh` 启动 OpenAI-compatible server，API 使用消息、temperature、max tokens，并在可用时提交 `structured_output.py::build_response_format` 的 JSON schema。健康接口暴露实际 backend/model/adapter/parse mode。

## 统一基准

`scripts/eval/benchmark_match_api.py` 对同一输入集运行 `jd_parse`、`resume_parse`、`match`、`batch_parse`、`batch_match`，配置 samples、requests、concurrency、batch size、warmup。输出 mean/p50/p95/max、requests/s、samples/s、后端分布和前后 GPU 快照。API 未报告 token usage 时，tokens/s 明确为 null，不估算伪数据。

示例：先分别启动两个后端，再使用相同 `--input data/eval/match_manual_eval_seed.jsonl --samples 8 --requests 4`，输出到不同文件比较。

## 失败、测试与限制

远端超时/HTTP/JSON 错误逐请求计数；GPU 快照是进程级 nvidia-smi 观察，不是请求独占显存。测试为 `tests/test_benchmark_match_api.py`、`test_server.py` 和推理相关测试。

当前环境未安装 vLLM，因此真实 A/B 性能报告仍为外部条件阻塞；默认 Transformers 更可复现，但不能据此宣称吞吐更优。后端晋级还必须通过同一 Gold 质量回归。
