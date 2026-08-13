from jobmatch_tune.inference.postprocess_json import parse_json_output, remove_thinking


def test_remove_thinking():
    assert remove_thinking("<think>abc</think>{\"a\":1}") == '{"a":1}'


def test_parse_json_output_repairs_trailing_comma():
    result = parse_json_output("说明 {“技能”:[\"Python\",],} 结束")
    assert result["ok"] is True
    assert result["data"]["技能"] == ["Python"]


def test_parse_json_output_repairs_match_conclusion_without_key():
    result = parse_json_output(
        '{"匹配优势":["经验背景满足岗位要求"],"主要短板":["学历不足"],'
        '"简历优化建议":["补充学历说明"],"JD 与简历匹配度较低。"}'
    )
    assert result["ok"] is True
    assert result["data"]["匹配结论"] == "JD 与简历匹配度较低。"
    assert set(result["data"]) == {"匹配结论", "匹配优势", "主要短板", "简历优化建议", "推荐投递岗位方向"}


def test_parse_json_output_does_not_rewrite_valid_list_tail():
    result = parse_json_output('{"简历优化建议":["补充项目成果","补充量化指标"]}')
    assert result["ok"] is True
    assert result["data"]["简历优化建议"] == ["补充项目成果", "补充量化指标"]


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


def test_parse_json_output_recognizes_client_direction_from_context():
    result = parse_json_output(
        '{"岗位方向":"后端开发","核心职责":["负责客户端功能实现与性能优化"]}',
        context_text="岗位名称：Unity 客户端开发工程师",
    )
    assert result["ok"] is True
    assert result["data"]["岗位方向"] == "客户端开发"


def test_parse_json_output_recognizes_embedded_direction_from_context():
    result = parse_json_output(
        '{"岗位方向":"后端开发","核心职责":["负责固件开发与驱动调试"]}',
        context_text="岗位名称：嵌入式固件开发工程师",
    )
    assert result["ok"] is True
    assert result["data"]["岗位方向"] == "嵌入式开发"


def test_parse_json_output_recognizes_sre_direction_from_context():
    result = parse_json_output(
        '{"岗位方向":"后端开发","核心职责":["建设运维平台并保障系统稳定性"]}',
        context_text="岗位名称：SRE工程师",
    )
    assert result["ok"] is True
    assert result["data"]["岗位方向"] == "运维开发"


def test_parse_json_output_recognizes_security_direction_from_context():
    result = parse_json_output(
        '{"岗位方向":"后端开发","核心职责":["负责漏洞研究与安全攻防"]}',
        context_text="岗位名称：安全工程师",
    )
    assert result["ok"] is True
    assert result["data"]["岗位方向"] == "安全工程"


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
    assert result["data"]["必备技能"] == []
    assert result["data"]["技能证据"][0]["证据来源"] == ["responsibility_evidence"]


def test_parse_json_output_recognizes_agent_inside_rl_agent_context():
    result = parse_json_output(
        '{"必备技能":[]}',
        context_text="岗位职责：\n1.负责大模型测试执行工作，包含 RL+agent 流程校验、链路质量和推理性能专项测试；",
    )
    assert result["ok"] is True
    assert result["data"]["必备技能"] == []


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


def test_parse_json_output_normalizes_resume_schema():
    result = parse_json_output(
        '{"目标岗位":"AI 应用开发工程师","教育背景":"本科，计算机科学与技术",'
        '"核心技能":["Python","RAG"],"实习经历":"参与知识库开发。",'
        '"项目经历":"搭建检索链路；接入业务系统","优势标签":"LLM应用落地"}'
    )
    assert result["ok"] is True
    assert result["data"] == {
        "目标岗位": "AI应用开发",
        "教育背景": ["本科，计算机科学与技术"],
        "核心技能": ["Python", "RAG"],
        "实习经历": ["参与知识库开发。"],
        "项目经历": ["搭建检索链路。", "接入业务系统。"],
        "优势标签": ["LLM应用落地"],
    }


def test_parse_json_output_canonicalizes_resume_skill_aliases():
    result = parse_json_output(
        '{"目标岗位":"后端开发","教育背景":[],"核心技能":["Golang","SpringBoot","自研框架"],'
        '"实习经历":[],"项目经历":[],"优势标签":[]}'
    )

    assert result["ok"] is True
    assert result["data"]["核心技能"] == ["Go", "Spring Boot", "自研框架"]


def test_parse_json_output_normalizes_structured_resume_items():
    result = parse_json_output(
        '{"目标岗位":"后端开发工程师","教育背景":{"学历":"本科","专业":"软件工程"},'
        '"核心技能":"Python","实习经历":[{"公司":"示例科技","内容":"参与服务开发"}],'
        '"项目经历":[{"名称":"订单平台","内容":"负责 API 开发"}]}'
    )
    assert result["ok"] is True
    assert result["data"]["教育背景"] == ["本科，软件工程"]
    assert result["data"]["实习经历"] == ["示例科技，参与服务开发"]
    assert result["data"]["项目经历"] == ["订单平台，负责 API 开发。"]


def test_parse_json_output_keeps_canonical_resume_direction():
    for direction in ("硬件研发", "网络与基础设施", "AI Infra"):
        result = parse_json_output(f'{{"目标岗位":"{direction}","教育背景":[]}}')
        assert result["ok"] is True
        assert result["data"]["目标岗位"] == direction


def test_parse_json_output_normalizes_resume_strength_tags():
    result = parse_json_output(
        '{"目标岗位":"后端开发","教育背景":[],'
        '"优势标签":["熟悉稳定性工程、自动化运维和监控体系。","具备 API 设计能力"]}'
    )
    assert result["ok"] is True
    assert result["data"]["优势标签"] == ["稳定性工程", "自动化运维", "监控体系", "API设计"]


def test_parse_json_output_canonicalizes_resume_strength_aliases():
    result = parse_json_output(
        '{"目标岗位":"AI应用开发","教育背景":[],'
        '"优势标签":["具备 LLM 应用落地经验","关注交互体验和性能优化",'
        '"自动化测试框架建设和质量保障经验","高并发场景优化"]}'
    )
    assert result["ok"] is True
    assert result["data"]["优势标签"] == [
        "LLM应用落地",
        "交互体验优化",
        "性能优化",
        "自动化测试框架",
        "质量保障",
        "高并发优化",
    ]
