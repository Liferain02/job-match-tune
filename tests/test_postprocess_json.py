from jobmatch_tune.inference.postprocess_json import parse_json_output, remove_thinking


def test_remove_thinking():
    assert remove_thinking("<think>abc</think>{\"a\":1}") == '{"a":1}'


def test_parse_json_output_repairs_trailing_comma():
    result = parse_json_output("说明 {“技能”:[\"Python\",],} 结束")
    assert result["ok"] is True
    assert result["data"]["技能"] == ["Python"]


def test_parse_json_output_deduplicates_lists():
    result = parse_json_output('{"加分项":["LoRA","LoRA","QLoRA"]}')
    assert result["ok"] is True
    assert result["data"]["加分项"] == ["LoRA", "QLoRA"]


def test_parse_json_output_normalizes_job_direction():
    result = parse_json_output('{"岗位方向":"AI开发"}')
    assert result["ok"] is True
    assert result["data"]["岗位方向"] == "AI应用开发"


def test_parse_json_output_canonicalizes_free_form_job_direction():
    result = parse_json_output('{"岗位方向":"游戏开发","核心职责":["负责项目/模块的测试流程优化"]}')
    assert result["ok"] is True
    assert result["data"]["岗位方向"] == "测试开发"


def test_parse_json_output_overrides_ai_application_to_algorithm_when_context_is_inference():
    result = parse_json_output('{"岗位方向":"AI应用开发","核心职责":["优化大模型推理性能，提升吞吐并控制成本"]}')
    assert result["ok"] is True
    assert result["data"]["岗位方向"] == "算法工程"


def test_parse_json_output_uses_context_text_for_frontend_direction():
    result = parse_json_output('{"岗位方向":"算法工程","核心职责":["负责业务场景上的落地"]}', context_text="岗位名称：混元多模态前端开发工程师")
    assert result["ok"] is True
    assert result["data"]["岗位方向"] == "前端开发"


def test_parse_json_output_uses_context_text_for_ai_application_direction():
    result = parse_json_output('{"岗位方向":"算法工程","核心职责":["负责业务场景上的落地"]}', context_text="岗位名称：ima copilot-大模型应用算法工程师")
    assert result["ok"] is True
    assert result["data"]["岗位方向"] == "AI应用开发"


def test_parse_json_output_keeps_backend_priority_for_ai_application_backend_title():
    result = parse_json_output(
        '{"岗位方向":"算法工程","核心职责":["优化模型调用链路"]}',
        context_text="岗位名称：企业微信-AI应用后台开发工程师-AI大模型应用（成都/北京）",
    )
    assert result["ok"] is True
    assert result["data"]["岗位方向"] == "后端开发"


def test_parse_json_output_keeps_test_priority_for_ai_eval_title():
    result = parse_json_output(
        '{"岗位方向":"算法工程","核心职责":["构建评测集并推进评测执行"]}',
        context_text="岗位名称：资深测试开发工程师（AI评测方向）",
    )
    assert result["ok"] is True
    assert result["data"]["岗位方向"] == "测试开发"


def test_parse_json_output_uses_algorithm_context_for_application_algorithm_title():
    result = parse_json_output(
        '{"岗位方向":"AI应用开发","核心职责":["负责元宝Post-training研发与应用"]}',
        context_text="岗位名称：大模型应用算法工程师",
    )
    assert result["ok"] is True
    assert result["data"]["岗位方向"] == "算法工程"


def test_parse_json_output_uses_application_context_for_copilot_application_algorithm_title():
    result = parse_json_output(
        '{"岗位方向":"算法工程","核心职责":["搭建任务框架，优化任务规划、工具调用和记忆能力"]}',
        context_text="岗位名称：ima copilot-大模型应用算法工程师",
    )
    assert result["ok"] is True
    assert result["data"]["岗位方向"] == "AI应用开发"


def test_parse_json_output_uses_algorithm_for_plain_application_algorithm_title_with_post_training():
    result = parse_json_output(
        '{"岗位方向":"AI应用开发","核心职责":["负责元宝Post-training研发与应用"]}',
        context_text="岗位名称：大模型应用算法工程师",
    )
    assert result["ok"] is True
    assert result["data"]["岗位方向"] == "算法工程"


def test_parse_json_output_uses_ai_application_for_innovation_application_engineer():
    result = parse_json_output(
        '{"岗位方向":"算法工程","核心职责":["设计Prompt并开发实际应用"]}',
        context_text="岗位名称：元宝-大模型创新应用工程师",
    )
    assert result["ok"] is True
    assert result["data"]["岗位方向"] == "AI应用开发"


def test_parse_json_output_uses_ai_application_for_agentic_engineer():
    result = parse_json_output(
        '{"岗位方向":"后端开发","核心职责":["设计Agent工作流并优化生成质量"]}',
        context_text="岗位名称：游戏Agentic Engineer",
    )
    assert result["ok"] is True
    assert result["data"]["岗位方向"] == "AI应用开发"


def test_parse_json_output_canonicalizes_and_filters_skills():
    result = parse_json_output('{"必备技能":["python","LangChain","模型剪枝"]}', context_text="岗位名称：大模型工程师")
    assert result["ok"] is True
    assert result["data"]["必备技能"] == []


def test_parse_json_output_keeps_only_evidence_backed_skills():
    result = parse_json_output(
        '{"必备技能":["JavaScript","React","TypeScript","Canvas","WebGL","Agent","Java"]}',
        context_text="岗位职责：\n1.对接LLM API，参与AI对话界面及Agent工作流前端实现。"
    )
    assert result["ok"] is True
    assert result["data"]["必备技能"] == ["Agent"]


def test_parse_json_output_recognizes_agent_inside_rl_agent_context():
    result = parse_json_output(
        '{"必备技能":[]}',
        context_text="岗位职责：\n1.负责大模型测试执行工作，包含 RL+agent 流程校验、链路质量和推理性能专项测试；",
    )
    assert result["ok"] is True
    assert result["data"]["必备技能"] == ["Agent"]


def test_parse_json_output_backfills_missing_responsibility_lines_from_context():
    result = parse_json_output(
        '{"核心职责":["1.负责方案设计","2.推进功能开发"]}',
        context_text="岗位职责：\n1.负责方案设计\n2.推进功能开发\n3.补充收尾职责\n经验要求：三年以上工作经验",
    )
    assert result["ok"] is True
    assert result["data"]["核心职责"] == ["1.负责方案设计", "2.推进功能开发", "3.补充收尾职责"]


def test_parse_json_output_moves_requirement_fields_out_of_responsibilities():
    result = parse_json_output(
        '{"核心职责":["1.负责模型研发","经验要求：三年以上工作经验","学历要求：本科及以上","任职要求：熟悉Python和LangChain"],"必备技能":[],"加分项":[]}'
    )
    assert result["ok"] is True
    assert result["data"]["核心职责"] == ["1.负责模型研发"]
    assert result["data"]["经验要求"] == "三年以上工作经验"
    assert result["data"]["学历要求"] == "本科及以上"
    assert "熟悉Python和LangChain" in result["data"]["任职要求"]
    assert "Python" in result["data"]["必备技能"]
    assert "LangChain" in result["data"]["必备技能"]
