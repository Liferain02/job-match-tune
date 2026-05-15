from __future__ import annotations

from pydantic import BaseModel, Field


class JDParseResult(BaseModel):
    job_direction: str = Field(default="", alias="岗位方向")
    responsibilities: list[str] = Field(default_factory=list, alias="核心职责")
    required_skills: list[str] = Field(default_factory=list, alias="必备技能")
    bonus: list[str] = Field(default_factory=list, alias="加分项")
    experience: str = Field(default="", alias="经验要求")
    education: str = Field(default="", alias="学历要求")


class ResumeParseResult(BaseModel):
    target_role: str = Field(default="", alias="目标岗位")
    education: list[str] = Field(default_factory=list, alias="教育背景")
    skills: list[str] = Field(default_factory=list, alias="核心技能")
    internships: list[str] = Field(default_factory=list, alias="实习经历")
    projects: list[str] = Field(default_factory=list, alias="项目经历")
    strengths: list[str] = Field(default_factory=list, alias="优势标签")


class MatchRuleResult(BaseModel):
    score: int = Field(default=0, alias="匹配分数")
    level: str = Field(default="", alias="匹配等级")
    matched_skills: list[str] = Field(default_factory=list, alias="命中技能")
    missing_skills: list[str] = Field(default_factory=list, alias="缺失技能")
    matched_projects: list[str] = Field(default_factory=list, alias="命中项目")
