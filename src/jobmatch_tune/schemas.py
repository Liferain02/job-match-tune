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
    direction_match: bool = Field(default=False, alias="岗位方向匹配")
    direction_relation: str = Field(default="mismatch", alias="岗位方向关系")
    direction_evidence: dict = Field(default_factory=dict, alias="岗位方向证据")
    education_match: bool = Field(default=False, alias="学历匹配")
    experience_match: bool = Field(default=False, alias="经验匹配")
    matched_skills: list[str] = Field(default_factory=list, alias="命中技能")
    missing_skills: list[str] = Field(default_factory=list, alias="缺失技能")
    matched_projects: list[str] = Field(default_factory=list, alias="命中项目")
    skill_evidence: list[dict] = Field(default_factory=list, alias="技能证据")
    score_breakdown: dict[str, int] = Field(default_factory=dict, alias="匹配分项")
    scoring_policy: dict = Field(default_factory=dict, alias="评分策略")


class MatchAnalysisResult(BaseModel):
    conclusion: str = Field(default="", alias="匹配结论")
    strengths: list[str] = Field(default_factory=list, alias="匹配优势")
    gaps: list[str] = Field(default_factory=list, alias="主要短板")
    suggestions: list[str] = Field(default_factory=list, alias="简历优化建议")
    recommended_roles: list[str] = Field(default_factory=list, alias="推荐投递岗位方向")
