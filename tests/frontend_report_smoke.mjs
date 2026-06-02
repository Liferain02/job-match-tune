import assert from "node:assert/strict";

import { buildMarkdownReport } from "../frontend/src/utils/text.js";

const payload = {
  jd_parse: {
    岗位方向: "AI应用开发",
    经验要求: "三年以上工作经验",
    学历要求: "本科及以上",
    必备技能: ["Python", "RAG"],
    核心职责: ["负责知识库问答系统开发"],
    加分项: ["LangChain"],
  },
  resume_parse: {
    目标岗位: "AI应用开发",
    教育背景: ["本科，计算机科学与技术"],
    核心技能: ["Python", "RAG"],
    实习经历: ["平台团队实习"],
    项目经历: ["企业知识库问答系统。"],
    优势标签: ["LLM应用落地"],
  },
  rule_result: {
    匹配等级: "高匹配",
    匹配分数: 90,
    岗位方向匹配: true,
    学历匹配: true,
    经验匹配: true,
    命中技能: ["Python", "RAG"],
    缺失技能: [],
    命中项目: ["企业知识库问答系统。"],
  },
  analysis: {
    匹配结论: "候选人与岗位整体高度匹配。",
    匹配优势: ["方向一致"],
    主要短板: ["暂无明显硬性短板"],
    简历优化建议: ["补充量化结果"],
    推荐投递岗位方向: ["同方向相近岗位"],
  },
};

const report = buildMarkdownReport(payload, "match", "single");

assert.match(report, /# 人岗匹配分析报告/);
assert.match(report, /## JD 结构化解析/);
assert.match(report, /## 简历结构化解析/);
assert.match(report, /## 匹配分析/);
assert.match(report, /候选人与岗位整体高度匹配/);
assert.match(report, /```json/);
