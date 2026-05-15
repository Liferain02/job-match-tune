from __future__ import annotations


SYSTEM_PROMPT = "你是一个招聘文本解析助手。请严格基于输入内容输出 JSON，不要编造输入中没有的信息。"


def jd_parse_prompt(jd_text: str) -> str:
    return (
        "请解析以下招聘 JD，抽取岗位方向、核心职责、必备技能、加分项、经验要求和学历要求。\n"
        "只输出 JSON，不要输出解释。\n\n"
        f"JD：\n{jd_text}"
    )


def resume_parse_prompt(resume_text: str) -> str:
    return (
        "请解析以下简历，抽取候选人的目标岗位、教育背景、核心技能、实习经历、项目经历和优势标签。\n"
        "只输出 JSON，不要输出解释。\n\n"
        f"简历：\n{resume_text}"
    )


def match_prompt(jd_text: str, resume_text: str, rule_result: str) -> str:
    return (
        "请根据 JD、简历和规则评分结果生成岗位匹配分析。\n"
        "只输出 JSON，包含匹配优势、主要短板和简历优化建议，不要编造输入中没有的信息。\n\n"
        f"JD：\n{jd_text}\n\n简历：\n{resume_text}\n\n规则评分结果：\n{rule_result}"
    )
