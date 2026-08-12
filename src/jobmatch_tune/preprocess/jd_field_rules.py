from __future__ import annotations

import re
from typing import Any


EDUCATION_PATTERNS = [
    r"(博士(?:研究生)?(?:及以上)?(?:\s*(?:学历|学位))?\s*(?:优先)?)",
    r"(硕士(?:研究生)?(?:及以上)?(?:\s*(?:学历|学位))?\s*(?:优先)?)",
    r"(研究生\s*(?:及以上|以上)?\s*学历(?:优先)?)",
    r"(专科\s*(?:及以上|以上)?\s*学历(?:优先)?)",
    r"(学士(?:及学士以上|及以上|以上)?(?:的)?(?:学历|学位)(?:优先)?)",
    r"(本科(?:及以上)?)",
    r"(大专(?:及以上)?)",
    r"(全日制本科(?:及以上)?)",
    r"(统招本科(?:及以上)?)",
]

EXPERIENCE_NUMBER = r"[一二三四五六七八九十两0-9]+"
EXPERIENCE_PATTERNS = [
    r"(经验要求[：:]\s*[^，。；;\n]+)",
    r"(工作经验[：:]\s*[^，。；;\n]+)",
    r"(经验不限)",
    r"(工作经验不限)",
    rf"(({EXPERIENCE_NUMBER})\s*[-~～至]\s*({EXPERIENCE_NUMBER})\s*年(?:工作)?经验)",
    rf"(({EXPERIENCE_NUMBER})\s*[-~～至]\s*({EXPERIENCE_NUMBER})\s*年)",
    rf"((?:至少|不少于|不低于|具备|拥有|具有|需要)?\s*{EXPERIENCE_NUMBER}\s*年\s*"
    rf"(?:以上|及以上|或以上|\+|左右|以内|以下)?(?:的)?[^，。；;\n]{{0,50}}(?:经验|经历|背景))",
    rf"((?:工作经验|相关经验|经验要求|工作年限)[^，。；;\n]{{0,20}}"
    rf"{EXPERIENCE_NUMBER}\s*(?:[-~至～]\s*{EXPERIENCE_NUMBER}\s*)?年)",
]

JOB_DIRECTION_RULES = [
    ("前端开发", ["前端", "react", "vue", "typescript", "javascript", "web 前端", "node", "next.js", "nextjs"]),
    ("客户端开发", ["客户端", "ios", "android", "u3d", "unity", "ue", "ue4", "ue5", "unreal engine", "cocos", "移动端", "桌面端", "sdk", "音视频引擎"]),
    ("嵌入式开发", ["嵌入式", "firmware", "固件", "驱动", "单片机", "bsp", "rtos"]),
    ("硬件研发", ["硬件开发", "硬件工程师", "电力电子", "功率硬件", "结构设计", "系统集成", "npi工程师", "声学工程师", "电子电器", "硬件结构", "音响开发"]),
    ("网络与基础设施", ["网络规划", "网络开发", "网络工程师", "网络交付", "数据中心网络", "基础架构", "机房网络"]),
    ("AI Infra", ["ai infra", "ai infrastructure", "机器学习平台", "训练平台", "推理平台", "训推平台", "模型训练平台", "模型推理平台", "算力平台", "rl infra"]),
    ("高性能计算", ["高性能计算", "hpc", "gpu 集群", "gpu集群", "计算集群", "分布式计算"]),
    ("汽车软件/智驾研发", ["智驾系统", "驾驶辅助", "泊车功能", "智能行车系统", "车控软件", "整车控制软件", "智驾软件集成", "底盘集成控制系统", "底盘电控功能开发", "感知质量开发", "转向电控", "制动电控", "发动机软件及标定", "混动系统开发", "混动电驱控制器"]),
    ("运维开发", ["sre", "运维", "devops", "infra", "平台运维", "云原生", "可靠性", "idc", "网络交付", "运营运维"]),
    ("安全工程", ["安全工程师", "渗透测试", "安全研发", "漏洞研究", "攻防", "数据安全", "安全运营", "安全运营工程师", "大模型安全"]),
    ("测试开发", ["测试开发", "测试工程师", "自动化测试", "性能测试", "测试流程", "测试方案", "测试任务", "代码质量", "质量保障", "qa", "开发质量工程师"]),
    ("后端开发", ["后端", "后台开发", "服务端", "java", "spring", "golang", "c++", "数据库", "分布式", "引擎", "游戏开发", "全栈开发", "架构师岗", "软件架构开发"]),
    ("数据开发", ["数据开发", "大数据开发", "数仓", "etl", "spark", "flink", "数据工程师", "数据平台"]),
    ("算法工程", ["算法", "algorithm engineer", "机器学习", "深度学习", "推理", "推理加速", "模型训练", "模型蒸馏", "world model", "视频生成", "多模态", "aigc", "生成模型", "大模型", "nlp", "tts", "asr", "对齐策略", "游戏ai"]),
    ("AI应用开发", ["ai应用", "大模型应用", "llm", "rag", "agent", "langchain", "langgraph", "智能体"]),
]

TITLE_PRIORITY_PATTERNS = [
    ("前端开发", [r"前端开发", r"web前端", r"前端工程师", r"react开发", r"vue开发", r"node开发"]),
    ("客户端开发", [r"客户端开发", r"ios开发", r"android开发", r"安卓开发", r"app开发", r"移动端开发", r"unity(?:3d)?开发", r"u3d开发", r"ue开发", r"\bue[45]\b.*(?:客户端|gameplay|引擎|渲染|图形|战斗|玩法)", r"cocos", r"桌面端开发工程师", r"音视频引擎sdk开发工程师", r"腾讯会议-(?:ios|android)研发工程师", r"软件开发工程师\s*\(sdk\)"]),
    ("嵌入式开发", [r"嵌入式", r"固件", r"驱动开发", r"\bbsp\b"]),
    ("硬件研发", [r"硬件开发工程师", r"硬件研发", r"功率硬件工程师", r"电力电子硬件开发工程师", r"系统集成工程师", r"结构设计工程师", r"硬件结构设计", r"音响开发工程师", r"\bnpi工程师\b", r"电子电器"]),
    ("网络与基础设施", [r"网络规划", r"网络开发", r"网络工程师", r"网络交付", r"基础架构工程师", r"数据中心网络", r"交换机软件研发工程师", r"云网络高级开发工程师", r"网络运营工程师", r"云接入网络运营工程师"]),
    ("AI Infra", [r"ai infra", r"ai infrastructure", r"机器学习平台", r"训练平台", r"推理平台", r"训推平台", r"rl infra"]),
    ("高性能计算", [r"高性能计算", r"\bhpc\b", r"ai编译优化工程师", r"编程语言&编译器工程师"]),
    ("汽车软件/智驾研发", [r"智驾系统架构工程师", r"驾驶辅助开发工程师", r"智能行车系统开发工程师", r"智驾泊车功能开发工程师", r"底盘集成控制系统开发", r"底盘电控功能开发工程师", r"感知质量开发工程师", r"转向电控工程师", r"制动电控工程师", r"发动机软件及标定工程师", r"混动系统开发工程师", r"混动电驱控制器工程师"]),
    ("运维开发", [r"\bsre\b", r"运维开发", r"devops", r"平台运维", r"\bidc\b", r"网络交付工程师", r"网络工程师", r"运营运维工程师", r"秒送物流sre", r"运营开发工程师", r"运营开发高级工程师"]),
    ("安全工程", [r"安全工程师", r"渗透测试", r"漏洞研究", r"安全研发", r"攻防", r"安全运营工程师", r"安全运营专家", r"安全运营岗", r"大模型安全运营"]),
    ("测试开发", [r"测试开发", r"测试工程师", r"测试负责人", r"\bqa\b", r"开发质量工程师（软件）"]),
    ("后端开发", [r"后台开发", r"后端开发", r"服务端", r"后台工程师", r"\bjava(?:web)?(?:软件)?开发(?:工程师|实习生)?", r"^java(?:工程师)?(?:\s|（|\(|$)", r"研发岗", r"资深架构师岗", r"软件架构开发工程", r"服务器开发工程师", r"服务器高级工程师", r"游戏服务器工程师", r"游戏玩法开发专家", r"操作系统高级研发工程师", r"linux内核高级研发工程师", r"云存储高级研发工程师"]),
    ("数据开发", [r"大数据开发工程师", r"数据开发工程师"]),
    ("算法工程", [r"algorithm engineer", r"\bnlp\b", r"\btts\b", r"\basr\b", r"对齐策略研发工程师", r"自然语言处理", r"游戏ai开发工程师"]),
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

BUSINESS_ROLE_PATTERNS = [
    r"项目经理",
    r"店长",
    r"招商",
    r"采销",
    r"客户经理",
    r"销售经理",
    r"销售岗",
    r"运营岗",
    r"运营经理",
    r"用户运营",
    r"营销",
    r"事务",
    r"拓展",
    r"关怀专家",
    r"服务总监",
    r"服务接待",
    r"公共安全专家",
    r"检验技师",
]

STRONG_TECH_TITLE_PATTERNS = [
    r"工程师",
    r"开发",
    r"算法",
    r"测试",
    r"研发",
    r"后端",
    r"前端",
    r"客户端",
    r"服务端",
    r"运维",
    r"\bsre\b",
    r"架构",
    r"\bsdk\b",
    r"编译",
    r"网络",
    r"内核",
    r"数据库",
    r"固件",
    r"驱动",
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _contains_direction_keyword(lowered_text: str, keyword: str) -> bool:
    """Match ASCII direction terms as tokens, not pieces of ordinary words."""
    lowered_keyword = keyword.lower()
    if not re.search(r"[a-z]", lowered_keyword) or re.search(r"[\u4e00-\u9fff]", lowered_keyword):
        return keyword.lower() in lowered_text
    offset = 0
    while (index := lowered_text.find(lowered_keyword, offset)) >= 0:
        end = index + len(lowered_keyword)
        left_char = lowered_text[index - 1] if index else ""
        right_char = lowered_text[end] if end < len(lowered_text) else ""
        left_ok = (
            not lowered_keyword[0].isalnum()
            or not left_char
            or not (left_char.isascii() and left_char.isalnum())
        )
        right_ok = (
            not lowered_keyword[-1].isalnum()
            or not right_char
            or not (right_char.isascii() and right_char.isalnum())
        )
        if left_ok and right_ok:
            return True
        offset = index + 1
    return False


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
    matches = []
    for pattern in EDUCATION_PATTERNS:
        match = re.search(pattern, text, flags=re.I)
        if match:
            matches.append(match)
    if not matches:
        return ""
    # The first qualification in the JD is the baseline requirement. A later
    # “硕士优先” must not overwrite an earlier “本科及以上学历”.
    return min(matches, key=lambda item: item.start()).group(1).strip()


def extract_experience_requirement(text: str) -> str:
    for pattern in EXPERIENCE_PATTERNS:
        match = re.search(pattern, text, flags=re.I)
        if match:
            value = match.group(1)
            year_values = [int(number) for number in re.findall(r"(\d+)\s*年", value)]
            if any(number > 50 for number in year_values):
                continue
            return re.sub(r"^(经验要求|工作经验)[：:]\s*", "", value).strip()
    return ""


def extract_experience_requirement_from_meta(meta: dict[str, Any] | None) -> str:
    if not isinstance(meta, dict):
        return ""
    for key in (
        "experience_requirement",
        "experience",
        "work_year",
        "workYear",
        "workExperience",
        "experienceRange",
    ):
        value = meta.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        if text in {"不限", "经验不限", "工作经验不限"}:
            return "经验不限"
        if re.search(r"^\d+\s*[-~至]\s*\d+\s*年$", text):
            return text
        if re.search(r"^\d+\+?\s*年$", text):
            return text
        if re.search(r"^\d+\s*年及以上$", text):
            return text
        if "经验" in text or "年" in text:
            return text
    return ""


def extract_skills_from_text(text: str, schema: dict[str, Any]) -> list[str]:
    found = []
    lowered_text = text.lower()
    for canonical, aliases in schema.get("skill_alias", {}).items():
        candidates = [canonical, *aliases]
        if any(_contains_skill_candidate(lowered_text, str(candidate)) for candidate in candidates):
            found.append(canonical)
    return found


def _contains_skill_candidate(lowered_text: str, candidate: str) -> bool:
    candidate = candidate.strip().lower()
    if not candidate:
        return False
    has_ascii_marker = any(
        char.isascii() and (char.isalnum() or char in "+#._/-") for char in candidate
    )
    if not has_ascii_marker:
        return candidate in lowered_text
    offset = 0
    while (index := lowered_text.find(candidate, offset)) >= 0:
        end = index + len(candidate)
        left_char = lowered_text[index - 1] if index else ""
        right_char = lowered_text[end] if end < len(lowered_text) else ""
        left_ok = not (left_char.isascii() and left_char.isalnum())
        right_ok = not (right_char.isascii() and right_char.isalnum())
        if candidate == "c":
            right_ok = right_ok and right_char != "+" and not re.match(
                r"\s*语言", lowered_text[end:]
            )
        if left_ok and right_ok:
            return True
        offset = index + 1
    return False


def infer_job_direction(title: str, text: str, schema: dict[str, Any]) -> str:
    normalized_title = _normalize_text(title).lower()
    normalized_text = _normalize_text(text).lower()
    if re.search(r"(创新应用工程师|agentic engineer)", normalized_title, flags=re.I):
        return "AI应用开发"
    # Explicit role phrases take precedence over department/domain prefixes.
    # Examples: "微信安全-数据算法工程师" is an algorithm role, while
    # "网络安全研究岗" is security rather than generic network infrastructure.
    if (
        re.search(
            r"(算法(?:高级|资深|首席)?(?:工程师|研究员|专家|研发)|评测算法(?:工程师|研究员)?)",
            normalized_title,
            flags=re.I,
        )
        and not re.search(r"(应用算法|算法应用|大模型应用)", normalized_title, flags=re.I)
    ):
        return "算法工程"
    explicit_security_title = re.search(
        r"(网络(?:与)?(?:信息|数据)?安全|信息安全|数据安全|安全研发|渗透测试|漏洞研究|安全攻防)",
        normalized_title,
        flags=re.I,
    )
    security_research_title = re.search(
        r"(安全[^\n]{0,8}研究员|研究员[^\n]{0,8}安全)", normalized_title, flags=re.I
    ) and not re.search(r"(食品|药物|生物|质量|生产)安全", normalized_title, flags=re.I)
    if explicit_security_title or security_research_title:
        return "安全工程"
    if "研究员" in normalized_title and not (
        any(re.search(pattern, normalized_title, flags=re.I) for pattern in STRONG_TECH_TITLE_PATTERNS)
        or re.search(
            r"(算法|人工智能|机器学习|深度学习|大模型|量化|嵌入式|仿真|机器视觉|计算机视觉|"
            r"自然语言处理|世界模型|图模型|机器人|医学影像|图像处理|系统智能|"
            r"(?<![a-z])ai(?:研究员|技术|安全|框架|智能|创新|融合|与|\s*os|\s*infra|\s*for\s*science)|"
            r"(?<![a-z])ai4science)",
            normalized_title,
            flags=re.I,
        )
        or (
            re.search(r"(大模型|\bllm\b)", normalized_text, flags=re.I)
            and re.search(r"(模型训练|多模态|扩散模型|生成模型|算法)", normalized_text, flags=re.I)
        )
    ):
        return ""
    if any(re.search(pattern, normalized_title, flags=re.I) for pattern in BUSINESS_ROLE_PATTERNS):
        if not any(re.search(pattern, normalized_title, flags=re.I) for pattern in STRONG_TECH_TITLE_PATTERNS):
            return ""
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
        ("前端开发", ["前端", "web前端", "web 前端", "react", "vue", "node"]),
        ("客户端开发", ["客户端", "ios", "android", "安卓", "app开发", "unity", "unity3d", "u3d", "ue", "unreal engine", "cocos", "sdk", "游戏客户端", "音视频引擎", "桌面端"]),
        ("嵌入式开发", ["嵌入式", "固件", "驱动", "bsp", "firmware"]),
        ("硬件研发", ["硬件开发", "硬件工程师", "功率硬件", "电力电子", "结构设计工程师", "系统集成工程师", "硬件结构", "音响开发", "npi工程师", "电子电器"]),
        ("网络与基础设施", ["网络规划", "网络开发", "网络工程师", "网络交付", "基础架构工程师", "数据中心网络", "交换机软件", "云网络", "网络运营", "云接入网络"]),
        ("AI Infra", ["ai infra", "机器学习平台", "训练平台", "推理平台", "训推平台", "算力平台", "rl infra"]),
        ("高性能计算", ["高性能计算", "hpc", "gpu集群", "gpu 集群", "编译优化", "编译器工程师"]),
        ("汽车软件/智驾研发", ["智驾系统", "驾驶辅助", "智能行车系统", "泊车功能开发", "底盘集成控制系统开发", "底盘电控功能开发工程师", "感知质量开发工程师", "智驾软件集成", "转向电控工程师", "制动电控工程师", "发动机软件及标定工程师", "混动系统开发工程师", "混动电驱控制器工程师"]),
        ("运维开发", ["运维", "sre", "devops", "infra", "可靠性", "平台运维", "运营运维", "运营开发"]),
        ("安全工程", ["安全", "渗透", "攻防", "漏洞", "dba 安全"]),
        ("测试开发", ["测试", "qa", "评测", "开发质量工程师"]),
        ("后端开发", ["后台", "后端", "服务端", "全栈", "引擎", "存储", "框架研发", "平台研发", "dba", "数据库", "研发岗", "架构师岗", "软件架构开发", "服务器开发", "游戏服务器", "游戏玩法开发", "操作系统研发", "linux内核", "云存储"]),
        ("数据开发", ["数据开发", "大数据开发", "数据平台", "数仓", "etl"]),
        ("产品经理", ["产品经理", "技术产品经理", "产品岗", "产品实习生"]),
        ("算法工程", ["算法", "自然语言处理", "世界模型", "图模型", "推理", "训练", "蒸馏", "强化学习", "rl", "多模态", "aigc", "游戏ai"]),
        ("AI应用开发", ["应用开发工程师", "application engineer", "应用架构师", "应用研究工程师", "agent开发工程师", "agent 应用", "llm application"]),
    ]
    for direction, keywords in title_rules:
        if any(_contains_direction_keyword(normalized_title, keyword) for keyword in keywords):
            return direction

    haystack = f"{title}\n{text}".lower()
    best_direction = ""
    best_score = 0
    for direction, keywords in JOB_DIRECTION_RULES:
        score = sum(1 for keyword in keywords if _contains_direction_keyword(haystack, keyword))
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
