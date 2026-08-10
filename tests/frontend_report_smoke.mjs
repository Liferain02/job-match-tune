import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import AppHeader from "../frontend/src/components/AppHeader.js";
import ControlPanel from "../frontend/src/components/ControlPanel.js";
import ResultPanel from "../frontend/src/components/ResultPanel.js";
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

const appSource = await readFile(new URL("../frontend/src/App.js", import.meta.url), "utf8");
assert.match(appSource, /task: "match"/);
assert.doesNotMatch(appSource, /SummaryGrid/);
assert.match(AppHeader.template, /看清岗位与简历是否匹配/);
assert.match(ControlPanel.template, /核心流程/);
assert.match(ControlPanel.template, /开始匹配/);
assert.match(ControlPanel.template, /file-picker/);
assert.match(ControlPanel.computed.submitLabel.toString(), /加载模型并开始匹配/);
assert.match(ControlPanel.computed.submitLabel.toString(), /约需 2–3 分钟/);
assert.match(ControlPanel.template, /本次将优先读取文件/);
assert.match(ControlPanel.template, /高级设置/);
assert.match(ControlPanel.template, /推理后端/);
assert.match(ControlPanel.template, /JD \/ 简历并行/);
assert.match(ControlPanel.template, /请勿上传未经授权的真实简历/);

const conclusionIndex = ResultPanel.template.indexOf("综合结论");
const evidenceIndex = ResultPanel.template.indexOf("硬性条件依据");
const actionIndex = ResultPanel.template.indexOf("下一步：优化简历");
assert.ok(conclusionIndex > 0);
assert.ok(evidenceIndex > conclusionIndex);
assert.ok(actionIndex > evidenceIndex);
assert.match(ResultPanel.template, /分数来自可复算规则/);
assert.match(ResultPanel.template, /JD \/ 简历并行解析/);
assert.match(ResultPanel.template, /parse_wall_seconds/);
