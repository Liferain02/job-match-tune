export function splitBatchText(text) {
  return String(text || "")
    .split(/\n-{3,}\n/g)
    .map((item) => item.trim())
    .filter(Boolean);
}

export function prettyJson(value) {
  return JSON.stringify(value ?? {}, null, 2);
}

function listLines(items) {
  const values = Array.isArray(items) ? items.filter(Boolean) : [];
  if (!values.length) return "-";
  return values.map((item) => `- ${item}`).join("\n");
}

function inlineList(items) {
  const values = Array.isArray(items) ? items.filter(Boolean) : [];
  return values.length ? values.join("、") : "-";
}

function scalar(value) {
  if (value === true) return "是";
  if (value === false) return "否";
  return value ?? "-";
}

function buildJdSection(data) {
  return [
    "## JD 结构化解析",
    "",
    `- 岗位方向：${scalar(data?.["岗位方向"])}`,
    `- 经验要求：${scalar(data?.["经验要求"])}`,
    `- 学历要求：${scalar(data?.["学历要求"])}`,
    `- 必备技能：${inlineList(data?.["必备技能"])}`,
    "",
    "### 核心职责",
    listLines(data?.["核心职责"]),
    "",
    "### 加分项",
    listLines(data?.["加分项"]),
  ].join("\n");
}

function buildResumeSection(data) {
  return [
    "## 简历结构化解析",
    "",
    `- 目标岗位：${scalar(data?.["目标岗位"])}`,
    `- 核心技能：${inlineList(data?.["核心技能"])}`,
    `- 优势标签：${inlineList(data?.["优势标签"])}`,
    "",
    "### 教育背景",
    listLines(data?.["教育背景"]),
    "",
    "### 实习经历",
    listLines(data?.["实习经历"]),
    "",
    "### 项目经历",
    listLines(data?.["项目经历"]),
  ].join("\n");
}

function buildMatchSection(payload) {
  const rule = payload?.rule_result || {};
  const analysis = payload?.analysis || {};
  return [
    "## 匹配分析",
    "",
    `- 匹配等级：${scalar(rule["匹配等级"])}`,
    `- 启发式匹配分数：${scalar(rule["匹配分数"])}`,
    `- 岗位方向匹配：${scalar(rule["岗位方向匹配"])}`,
    `- 学历匹配：${scalar(rule["学历匹配"])}`,
    `- 经验匹配：${scalar(rule["经验匹配"])}`,
    `- 命中技能：${inlineList(rule["命中技能"])}`,
    `- 缺失技能：${inlineList(rule["缺失技能"])}`,
    "",
    "### 匹配结论",
    scalar(analysis["匹配结论"]),
    "",
    "### 匹配优势",
    listLines(analysis["匹配优势"]),
    "",
    "### 主要短板",
    listLines(analysis["主要短板"]),
    "",
    "### 简历优化建议",
    listLines(analysis["简历优化建议"]),
    "",
    "### 推荐投递岗位方向",
    listLines(analysis["推荐投递岗位方向"]),
    "",
    "### 命中项目",
    listLines(rule["命中项目"]),
  ].join("\n");
}

function buildBatchSection(payload, task) {
  const items = payload?.items || [];
  const title = task === "match" ? "批量匹配结果" : "批量解析结果";
  const rows = items.map((item) => {
    if (task === "match") {
      return [
        `### 样本 ${item.index + 1}`,
        "",
        `- 状态：${item.ok ? "成功" : "失败"}`,
        `- 匹配等级：${scalar(item.rule_result?.["匹配等级"])}`,
        `- 启发式匹配分数：${scalar(item.rule_result?.["匹配分数"])}`,
        `- 命中技能数：${(item.rule_result?.["命中技能"] || []).length}`,
        `- 缺失技能数：${(item.rule_result?.["缺失技能"] || []).length}`,
      ].join("\n");
    }
    const data = item.data || {};
    return [
      `### 样本 ${item.index + 1}`,
      "",
      `- 状态：${item.ok ? "成功" : "失败"}`,
      `- ${task === "resume_parse" ? "目标岗位" : "岗位方向"}：${scalar(
        task === "resume_parse" ? data["目标岗位"] : data["岗位方向"],
      )}`,
      `- 技能数：${((task === "resume_parse" ? data["核心技能"] : data["必备技能"]) || []).length}`,
    ].join("\n");
  });
  return [`## ${title}`, "", ...rows].join("\n\n");
}

export function buildMarkdownReport(payload, task, requestMode) {
  if (!payload) return "";
  const title = task === "match" ? "人岗匹配分析报告" : task === "resume_parse" ? "简历结构化解析报告" : "JD 结构化解析报告";
  const sections = [
    `# ${title}`,
    "",
    `生成时间：${new Date().toLocaleString("zh-CN")}`,
    "",
  ];
  if (requestMode === "batch") {
    sections.push(buildBatchSection(payload, task));
  } else if (task === "match") {
    sections.push(buildJdSection(payload.jd_parse || {}));
    sections.push("");
    sections.push(buildResumeSection(payload.resume_parse || {}));
    sections.push("");
    sections.push(buildMatchSection(payload));
  } else if (task === "resume_parse") {
    sections.push(buildResumeSection(payload.data || payload));
  } else {
    sections.push(buildJdSection(payload.data || payload));
  }
  sections.push("", "## 原始 JSON", "", "```json", prettyJson(payload), "```", "");
  return sections.join("\n");
}
