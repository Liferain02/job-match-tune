from __future__ import annotations

import re
from typing import Any


EDUCATION_PATTERNS = [
    r"(博士(?:研究生)?(?:及以上)?)",
    r"(硕士(?:研究生)?(?:及以上)?)",
    r"(本科(?:及以上)?)",
    r"(大专(?:及以上)?)",
    r"(全日制本科(?:及以上)?)",
    r"(统招本科(?:及以上)?)",
]

EXPERIENCE_PATTERNS = [
    r"((?:[一二三四五六七八九十两0-9]+|[0-9]+\+?)年以上[^，。；;\n]*经验)",
    r"((?:[一二三四五六七八九十两0-9]+|[0-9]+\+?)年[^，。；;\n]*工作经验)",
    r"(经验要求[：:]\s*[^，。；;\n]+)",
    r"(工作经验[：:]\s*[^，。；;\n]+)",
    r"(经验不限)",
    r"(工作经验不限)",
]

JOB_DIRECTION_RULES = [
    ("前端开发", ["前端", "react", "vue", "typescript", "javascript", "web 前端"]),
    ("客户端开发", ["客户端", "ios", "android", "u3d", "unity", "ue", "cocos", "移动端"]),
    ("嵌入式开发", ["嵌入式", "firmware", "固件", "驱动", "单片机", "bsp", "rtos"]),
    ("运维开发", ["sre", "运维", "devops", "infra", "平台运维", "云原生", "可靠性", "idc", "网络交付"]),
    ("安全工程", ["安全工程师", "渗透测试", "安全研发", "漏洞研究", "攻防", "数据安全", "安全运营", "安全运营工程师"]),
    ("测试开发", ["测试开发", "测试工程师", "自动化测试", "性能测试", "测试流程", "测试方案", "测试任务", "代码质量", "质量保障", "qa"]),
    ("后端开发", ["后端", "后台开发", "服务端", "java", "spring", "golang", "c++", "数据库", "分布式", "引擎", "游戏开发", "全栈开发"]),
    ("数据开发", ["数据开发", "数仓", "etl", "spark", "flink", "数据工程师", "数据平台"]),
    ("算法工程", ["算法", "algorithm engineer", "机器学习", "深度学习", "推理", "推理加速", "模型训练", "模型蒸馏", "world model", "视频生成", "多模态", "aigc", "生成模型", "大模型", "nlp", "tts", "asr", "对齐策略"]),
    ("AI应用开发", ["ai应用", "大模型应用", "llm", "rag", "agent", "langchain", "langgraph", "智能体"]),
]

TITLE_PRIORITY_PATTERNS = [
    ("前端开发", [r"前端开发", r"web前端", r"前端工程师"]),
    ("客户端开发", [r"客户端开发", r"ios开发", r"android开发", r"移动端开发", r"unity开发", r"u3d开发", r"ue开发", r"cocos"]),
    ("嵌入式开发", [r"嵌入式", r"固件", r"驱动开发", r"\bbsp\b"]),
    ("运维开发", [r"\bsre\b", r"运维开发", r"devops", r"平台运维", r"基础设施", r"\bidc\b", r"网络交付工程师", r"网络工程师"]),
    ("安全工程", [r"安全工程师", r"渗透测试", r"漏洞研究", r"安全研发", r"攻防", r"安全运营工程师", r"安全运营专家"]),
    ("测试开发", [r"测试开发", r"测试工程师", r"测试负责人", r"\bqa\b"]),
    ("后端开发", [r"后台开发", r"后端开发", r"服务端", r"后台工程师"]),
    ("算法工程", [r"algorithm engineer", r"\bnlp\b", r"\btts\b", r"\basr\b", r"对齐策略研发工程师", r"自然语言处理"]),
]

ALGORITHM_CONTEXT_PATTERNS = [
    r"post-training",
    r"后训练",
    r"\brl\b",
    r"\brm\b",
    r"强化学习",
    r"推理优化",
    r"推理加速",
    r"模型训练",
    r"蒸馏",
    r"算法研究",
]

AI_APPLICATION_CONTEXT_PATTERNS = [
    r"落地实践",
    r"业务场景落地",
    r"业务场景落地",
    r"场景服务",
    r"任务自动化",
    r"智能对话",
    r"推荐与搜索场景",
    r"个性化推荐",
    r"精准搜索",
    r"智能客服",
    r"投研分析",
    r"风险预警",
    r"应用能力",
    r"ai paas",
    r"prompt",
    r"demo",
    r"工作流",
    r"知识库问答",
    r"任务规划",
    r"工具调用",
    r"记忆能力",
    r"badcase",
    r"问答链路",
]

NON_TECH_TITLE_PATTERNS = [
    r"公关",
    r"客服",
    r"销售",
    r"采销",
    r"店长",
    r"招商主管",
    r"招商",
    r"品牌",
    r"渠道经理",
    r"商品经理",
    r"客户经理",
    r"结算",
    r"物流",
    r"采购",
    r"运营",
    r"商务合作",
    r"人力",
    r"\bhr\b",
    r"财务",
    r"法务",
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_bullets(text: str) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-•\*\t ]+", "", line).strip()
        if line:
            lines.append(line)
    return lines


def extract_education_requirement(text: str) -> str:
    for pattern in EDUCATION_PATTERNS:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1)
    return ""


def extract_experience_requirement(text: str) -> str:
    for pattern in EXPERIENCE_PATTERNS:
        match = re.search(pattern, text, flags=re.I)
        if match:
            value = match.group(1)
            return re.sub(r"^(经验要求|工作经验)[：:]\s*", "", value).strip()
    return ""


def extract_skills_from_text(text: str, schema: dict[str, Any]) -> list[str]:
    found = []
    lower_text = text.lower()
    for canonical, aliases in schema.get("skill_alias", {}).items():
        candidates = [canonical, *aliases]
        if any(candidate.lower() in lower_text for candidate in candidates):
            found.append(canonical)
    return found


def infer_job_direction(title: str, text: str, schema: dict[str, Any]) -> str:
    normalized_title = _normalize_text(title).lower()
    normalized_text = _normalize_text(text).lower()
    if re.search(r"(创新应用工程师|agentic engineer)", normalized_title, flags=re.I):
        return "AI应用开发"
    for direction, patterns in TITLE_PRIORITY_PATTERNS:
        if any(re.search(pattern, normalized_title, flags=re.I) for pattern in patterns):
            return direction
    if any(re.search(pattern, normalized_title, flags=re.I) for pattern in NON_TECH_TITLE_PATTERNS):
        return ""

    if re.search(r"(应用算法工程师|算法应用)", normalized_title, flags=re.I):
        if re.search(r"(copilot|创新应用|agentic engineer|应用工程师)", normalized_title, flags=re.I):
            return "AI应用开发"
        if re.search(r"^大模型应用算法工程师$", normalized_title, flags=re.I):
            if any(re.search(pattern, normalized_text, flags=re.I) for pattern in ALGORITHM_CONTEXT_PATTERNS):
                return "算法工程"
            return "AI应用开发"
        if any(re.search(pattern, normalized_text, flags=re.I) for pattern in ALGORITHM_CONTEXT_PATTERNS):
            if any(re.search(pattern, normalized_text, flags=re.I) for pattern in AI_APPLICATION_CONTEXT_PATTERNS):
                return "AI应用开发"
            return "算法工程"
        if any(re.search(pattern, normalized_text, flags=re.I) for pattern in AI_APPLICATION_CONTEXT_PATTERNS):
            return "AI应用开发"
        return "AI应用开发"

    title_rules = [
        ("前端开发", ["前端", "web前端", "web 前端"]),
        ("客户端开发", ["客户端", "ios", "android", "unity", "u3d", "ue", "cocos"]),
        ("嵌入式开发", ["嵌入式", "固件", "驱动", "bsp", "firmware"]),
        ("运维开发", ["运维", "sre", "devops", "infra", "可靠性", "平台运维"]),
        ("安全工程", ["安全", "渗透", "攻防", "漏洞", "dba 安全"]),
        ("测试开发", ["测试", "qa", "评测"]),
        ("后端开发", ["后台", "后端", "服务端", "全栈", "引擎", "存储", "框架研发", "平台研发", "dba", "数据库"]),
        ("算法工程", ["算法", "研究员", "推理", "训练", "蒸馏", "强化学习", "rl", "多模态", "aigc"]),
        ("AI应用开发", ["应用开发工程师", "application engineer", "应用架构师", "应用研究工程师", "agent开发工程师", "agent 应用", "llm application"]),
        ("产品经理", ["产品经理", "技术产品经理"]),
    ]
    for direction, keywords in title_rules:
        if any(keyword in normalized_title for keyword in keywords):
            return direction

    haystack = f"{title}\n{text}".lower()
    best_direction = ""
    best_score = 0
    for direction, keywords in JOB_DIRECTION_RULES:
        score = sum(1 for keyword in keywords if keyword.lower() in haystack)
        if score > best_score:
            best_score = score
            best_direction = direction
    if best_score > 0:
        return best_direction
    return ""


def canonicalize_job_direction(direction: str, context: str, schema: dict[str, Any]) -> str:
    normalized = _normalize_text(direction)
    if not normalized:
        return infer_job_direction("", context, schema)
    title_match = re.search(r"岗位名称[：:]\s*([^\n]+)", context)
    title = title_match.group(1).strip() if title_match else ""
    title_first_direction = infer_job_direction(title or normalized, context, schema)
    if title_first_direction:
        return title_first_direction
    if normalized.lower() == "ai应用开发" and re.search(r"(算法|推理|模型训练|模型蒸馏|world model|世界模型)", context, flags=re.I):
        return "算法工程"
    haystack = f"{normalized}\n{context}"
    return infer_job_direction(title or normalized, haystack, schema)


def merge_unique(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        normalized = _normalize_text(item)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result
