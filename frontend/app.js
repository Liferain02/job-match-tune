const examples = {
  jd_parse: `岗位名称：AI 应用开发工程师
工作地点：深圳
岗位职责：
负责基于大模型 API 的智能问答、知识库和 Agent 应用开发。
参与 RAG 检索链路设计、提示词优化和效果评估。
对接业务系统，完成后端接口开发和部署。
任职要求：
熟悉 Python，了解 LangChain、RAG、向量数据库等技术。
有后端开发经验，熟悉 HTTP API、MySQL、Redis。
本科及以上学历，计算机相关专业优先。
加分项：
有 LoRA / QLoRA 微调、模型评估或 LangGraph 项目经验。`,
  resume_parse: `姓名：张三
目标岗位：AI 应用开发工程师
教育经历：本科，计算机科学与技术
技能：Python、FastAPI、LangChain、RAG、MySQL、Redis、Docker
项目经历：
负责企业知识库问答系统开发，完成文档切分、向量检索、重排序和答案生成链路。
参与客服智能体项目，设计工具调用流程，并接入业务工单系统。`,
  match: {
    jd: `岗位名称：AI Infra 研发工程师
工作地点：上海
岗位职责：
负责训练平台、推理平台和 GPU 资源调度系统研发。
建设模型服务部署链路，优化容器化、监控、灰度和稳定性。
与算法团队协作，支撑大模型训练、推理和评测平台落地。
任职要求：
熟悉 Python 或 Go，了解 Kubernetes、Docker、Linux、CI/CD。
有后端或平台研发经验，熟悉 GPU、分布式训练或模型服务部署优先。
本科及以上学历，计算机相关专业优先。`,
    resume: `姓名：李四
目标岗位：平台研发 / AI Infra
教育背景：本科，软件工程
核心技能：Python、Go、Kubernetes、Docker、Linux、Prometheus、Redis、MySQL
项目经历：
1. 负责公司内部模型推理平台研发，完成容器化部署、GPU 调度接入和服务监控告警。
2. 参与训练任务编排系统开发，对接 Kubernetes Job 和对象存储，提高训练任务稳定性。
实习经历：
在云平台团队参与 CI/CD 与发布链路建设。`,
  },
};

const state = {
  task: "jd_parse",
  requestMode: "single",
  activeView: "structured",
  lastPayload: null,
};

const BATCH_DELIMITER = "\n---\n";

const apiBase = document.querySelector("#apiBase");
const inputText = document.querySelector("#inputText");
const jdInputText = document.querySelector("#jdInputText");
const resumeInputText = document.querySelector("#resumeInputText");
const inputLabel = document.querySelector("#inputLabel");
const singleInputField = document.querySelector("#singleInputField");
const matchInputGrid = document.querySelector("#matchInputGrid");
const status = document.querySelector("#status");
const latency = document.querySelector("#latency");
const parseButton = document.querySelector("#parseButton");
const exampleButton = document.querySelector("#exampleButton");
const warmupButton = document.querySelector("#warmupButton");
const copyJsonButton = document.querySelector("#copyJsonButton");
const jdMode = document.querySelector("#jdMode");
const resumeMode = document.querySelector("#resumeMode");
const matchMode = document.querySelector("#matchMode");
const singleRequestMode = document.querySelector("#singleRequestMode");
const batchRequestMode = document.querySelector("#batchRequestMode");
const structuredTab = document.querySelector("#structuredTab");
const rawTab = document.querySelector("#rawTab");
const structuredView = document.querySelector("#structuredView");
const rawView = document.querySelector("#rawView");
const modelPath = document.querySelector("#modelPath");
const adapterPath = document.querySelector("#adapterPath");
const gpuState = document.querySelector("#gpuState");
const loadState = document.querySelector("#loadState");
const backendName = document.querySelector("#backendName");
const inputStats = document.querySelector("#inputStats");
const taskHint = document.querySelector("#taskHint");
const overviewLabel1 = document.querySelector("#overviewLabel1");
const overviewLabel2 = document.querySelector("#overviewLabel2");
const overviewLabel3 = document.querySelector("#overviewLabel3");
const overviewLabel4 = document.querySelector("#overviewLabel4");
const overviewValue1 = document.querySelector("#overviewValue1");
const overviewValue2 = document.querySelector("#overviewValue2");
const overviewValue3 = document.querySelector("#overviewValue3");
const overviewValue4 = document.querySelector("#overviewValue4");
const batchTip = document.querySelector("#batchTip");

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function setStatus(text, mode = "") {
  status.textContent = text;
  status.className = `status-pill ${mode}`.trim();
}

function renderList(items) {
  if (!Array.isArray(items) || items.length === 0) {
    return '<div class="list-empty">-</div>';
  }
  return items.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function renderChipList(items) {
  if (!Array.isArray(items) || items.length === 0) {
    return '<div class="list-empty">-</div>';
  }
  return `<div class="chip-list">${items
    .map((item) => `<span class="chip">${escapeHtml(item)}</span>`)
    .join("")}</div>`;
}

function resetOverview() {
  if (state.requestMode === "batch") {
    overviewLabel1.textContent = "总样本";
    overviewLabel2.textContent = "成功数";
    overviewLabel3.textContent = "失败数";
    overviewLabel4.textContent = "任务类型";
    overviewValue1.textContent = "0";
    overviewValue2.textContent = "0";
    overviewValue3.textContent = "0";
    overviewValue4.textContent = state.task === "match" ? "批量匹配" : "批量解析";
    return;
  }
  const defaultLabels =
    state.task === "match"
      ? ["匹配等级", "匹配分数", "命中技能", "缺失技能"]
      : state.task === "resume_parse"
        ? ["目标岗位", "技能条目", "项目经历", "优势标签"]
        : ["岗位方向", "职责条目", "技能条目", "加分项"];
  [overviewLabel1.textContent, overviewLabel2.textContent, overviewLabel3.textContent, overviewLabel4.textContent] =
    defaultLabels;
  overviewValue1.textContent = "-";
  overviewValue2.textContent = "0";
  overviewValue3.textContent = "0";
  overviewValue4.textContent = "0";
}

function setMode(task) {
  state.task = task;
  jdMode.classList.toggle("active", task === "jd_parse");
  resumeMode.classList.toggle("active", task === "resume_parse");
  matchMode.classList.toggle("active", task === "match");

  const isMatch = task === "match";
  singleInputField.classList.toggle("hidden", isMatch);
  matchInputGrid.classList.toggle("hidden", !isMatch);

  if (task === "jd_parse") {
    inputLabel.textContent = "输入文本";
    taskHint.textContent = "当前任务：JD 结构化";
  } else if (task === "resume_parse") {
    inputLabel.textContent = "输入文本";
    taskHint.textContent = "当前任务：简历结构化";
  } else {
    taskHint.textContent = "当前任务：人岗匹配分析";
  }

  resetOverview();
  renderStructured(null);
  updateInputStats();
}

function setRequestMode(mode) {
  state.requestMode = mode;
  singleRequestMode.classList.toggle("active", mode === "single");
  batchRequestMode.classList.toggle("active", mode === "batch");
  batchTip.classList.toggle("hidden", mode !== "batch");
  resetOverview();
  renderStructured(null);
  updateInputStats();
}

function setView(view) {
  state.activeView = view;
  structuredTab.classList.toggle("active", view === "structured");
  rawTab.classList.toggle("active", view === "raw");
  structuredView.classList.toggle("active", view === "structured");
  rawView.classList.toggle("active", view === "raw");
}

function renderParseStructured(data, task) {
  if (!data || typeof data !== "object") {
    resetOverview();
    structuredView.innerHTML = '<div class="empty-state">等待结构化结果</div>';
    return;
  }

  if (task === "resume_parse") {
    const skills = data["核心技能"] || [];
    const projects = data["项目经历"] || [];
    const strengths = data["优势标签"] || [];

    overviewLabel1.textContent = "目标岗位";
    overviewLabel2.textContent = "技能条目";
    overviewLabel3.textContent = "项目经历";
    overviewLabel4.textContent = "优势标签";
    overviewValue1.textContent = data["目标岗位"] || "-";
    overviewValue2.textContent = String(skills.length);
    overviewValue3.textContent = String(projects.length);
    overviewValue4.textContent = String(strengths.length);

    structuredView.innerHTML = `
      <div class="result-grid">
        <section class="result-card compact">
          <span class="card-label">目标岗位</span>
          <strong>${escapeHtml(data["目标岗位"] || "-")}</strong>
        </section>
        <section class="result-card compact">
          <span class="card-label">教育背景条目</span>
          <strong>${escapeHtml(String((data["教育背景"] || []).length))}</strong>
        </section>
        <section class="result-card">
          <div class="card-head">
            <span class="card-label">核心技能</span>
          </div>
          ${renderChipList(skills)}
        </section>
        <section class="result-card">
          <div class="card-head">
            <span class="card-label">优势标签</span>
          </div>
          ${renderChipList(strengths)}
        </section>
        <section class="result-card span-two">
          <div class="card-head">
            <span class="card-label">教育背景</span>
          </div>
          <ul class="bullet-list">${renderList(data["教育背景"] || [])}</ul>
        </section>
        <section class="result-card span-two">
          <div class="card-head">
            <span class="card-label">实习经历</span>
          </div>
          <ul class="bullet-list">${renderList(data["实习经历"] || [])}</ul>
        </section>
        <section class="result-card span-two">
          <div class="card-head">
            <span class="card-label">项目经历</span>
          </div>
          <ul class="bullet-list">${renderList(projects)}</ul>
        </section>
      </div>
    `;
    return;
  }

  const responsibilities = data["核心职责"] || [];
  const skills = data["必备技能"] || [];
  const bonus = data["加分项"] || [];

  overviewLabel1.textContent = "岗位方向";
  overviewLabel2.textContent = "职责条目";
  overviewLabel3.textContent = "技能条目";
  overviewLabel4.textContent = "加分项";
  overviewValue1.textContent = data["岗位方向"] || "-";
  overviewValue2.textContent = String(responsibilities.length);
  overviewValue3.textContent = String(skills.length);
  overviewValue4.textContent = String(bonus.length);

  structuredView.innerHTML = `
    <div class="result-grid">
      <section class="result-card compact">
        <span class="card-label">岗位方向</span>
        <strong>${escapeHtml(data["岗位方向"] || "-")}</strong>
      </section>
      <section class="result-card compact">
        <span class="card-label">经验要求</span>
        <strong>${escapeHtml(data["经验要求"] || "-")}</strong>
      </section>
      <section class="result-card compact">
        <span class="card-label">学历要求</span>
        <strong>${escapeHtml(data["学历要求"] || "-")}</strong>
      </section>
      <section class="result-card">
        <div class="card-head">
          <span class="card-label">必备技能</span>
        </div>
        ${renderChipList(skills)}
      </section>
      <section class="result-card span-two">
        <div class="card-head">
          <span class="card-label">核心职责</span>
        </div>
        <ul class="bullet-list">${renderList(responsibilities)}</ul>
      </section>
      <section class="result-card span-two">
        <div class="card-head">
          <span class="card-label">加分项</span>
        </div>
        <ul class="bullet-list">${renderList(bonus)}</ul>
      </section>
    </div>
  `;
}

function renderMatchStructured(data) {
  if (!data || typeof data !== "object") {
    resetOverview();
    structuredView.innerHTML = '<div class="empty-state">等待匹配分析结果</div>';
    return;
  }

  if (state.requestMode === "batch") {
    renderBatchMatchStructured(data);
    return;
  }

  const jdParse = data.jd_parse || {};
  const resumeParse = data.resume_parse || {};
  const ruleResult = data.rule_result || {};
  const analysis = data.analysis || {};
  const matchedSkills = ruleResult["命中技能"] || [];
  const missingSkills = ruleResult["缺失技能"] || [];
  const matchedProjects = ruleResult["命中项目"] || [];

  overviewLabel1.textContent = "匹配等级";
  overviewLabel2.textContent = "匹配分数";
  overviewLabel3.textContent = "命中技能";
  overviewLabel4.textContent = "缺失技能";
  overviewValue1.textContent = ruleResult["匹配等级"] || "-";
  overviewValue2.textContent = String(ruleResult["匹配分数"] ?? "-");
  overviewValue3.textContent = String(matchedSkills.length);
  overviewValue4.textContent = String(missingSkills.length);

  structuredView.innerHTML = `
    <div class="result-grid">
      <section class="result-card compact">
        <span class="card-label">匹配等级</span>
        <strong>${escapeHtml(ruleResult["匹配等级"] || "-")}</strong>
      </section>
      <section class="result-card compact">
        <span class="card-label">匹配分数</span>
        <strong>${escapeHtml(String(ruleResult["匹配分数"] ?? "-"))}</strong>
      </section>
      <section class="result-card compact">
        <span class="card-label">岗位方向匹配</span>
        <strong>${ruleResult["岗位方向匹配"] ? "是" : "否"}</strong>
      </section>
      <section class="result-card compact">
        <span class="card-label">学历 / 经验</span>
        <strong>${ruleResult["学历匹配"] ? "学历匹配" : "学历缺口"} / ${ruleResult["经验匹配"] ? "经验匹配" : "经验缺口"}</strong>
      </section>
      <section class="result-card">
        <div class="card-head">
          <span class="card-label">命中技能</span>
        </div>
        ${renderChipList(matchedSkills)}
      </section>
      <section class="result-card">
        <div class="card-head">
          <span class="card-label">缺失技能</span>
        </div>
        ${renderChipList(missingSkills)}
      </section>
      <section class="result-card">
        <div class="card-head">
          <span class="card-label">匹配优势</span>
        </div>
        <ul class="bullet-list">${renderList(analysis["匹配优势"] || [])}</ul>
      </section>
      <section class="result-card">
        <div class="card-head">
          <span class="card-label">主要短板</span>
        </div>
        <ul class="bullet-list">${renderList(analysis["主要短板"] || [])}</ul>
      </section>
      <section class="result-card span-two">
        <div class="card-head">
          <span class="card-label">匹配结论</span>
        </div>
        <div class="paragraph-block">${escapeHtml(analysis["匹配结论"] || "-")}</div>
      </section>
      <section class="result-card">
        <div class="card-head">
          <span class="card-label">命中项目</span>
        </div>
        <ul class="bullet-list">${renderList(matchedProjects)}</ul>
      </section>
      <section class="result-card">
        <div class="card-head">
          <span class="card-label">推荐投递岗位方向</span>
        </div>
        ${renderChipList(analysis["推荐投递岗位方向"] || [])}
      </section>
      <section class="result-card span-two">
        <div class="card-head">
          <span class="card-label">简历优化建议</span>
        </div>
        <ul class="bullet-list">${renderList(analysis["简历优化建议"] || [])}</ul>
      </section>
      <section class="result-card">
        <div class="card-head">
          <span class="card-label">JD 摘要</span>
        </div>
        <ul class="bullet-list">
          <li>岗位方向：${escapeHtml(jdParse["岗位方向"] || "-")}</li>
          <li>经验要求：${escapeHtml(jdParse["经验要求"] || "-")}</li>
          <li>学历要求：${escapeHtml(jdParse["学历要求"] || "-")}</li>
        </ul>
      </section>
      <section class="result-card">
        <div class="card-head">
          <span class="card-label">简历摘要</span>
        </div>
        <ul class="bullet-list">
          <li>目标岗位：${escapeHtml(resumeParse["目标岗位"] || "-")}</li>
          <li>核心技能数：${escapeHtml(String((resumeParse["核心技能"] || []).length))}</li>
          <li>项目经历数：${escapeHtml(String((resumeParse["项目经历"] || []).length))}</li>
        </ul>
      </section>
    </div>
  `;
}

function renderBatchParseStructured(data) {
  const items = Array.isArray(data?.items) ? data.items : [];
  overviewLabel1.textContent = "总样本";
  overviewLabel2.textContent = "成功数";
  overviewLabel3.textContent = "失败数";
  overviewLabel4.textContent = "任务类型";
  overviewValue1.textContent = String(data?.total ?? items.length);
  overviewValue2.textContent = String(data?.success_count ?? 0);
  overviewValue3.textContent = String((data?.total ?? items.length) - (data?.success_count ?? 0));
  overviewValue4.textContent = data?.task === "resume_parse" ? "批量简历解析" : "批量 JD 解析";

  structuredView.innerHTML = `
    <div class="batch-result-list">
      ${items
        .map((item) => {
          const parsed = item.data || {};
          const primary =
            data?.task === "resume_parse"
              ? parsed["目标岗位"] || "-"
              : parsed["岗位方向"] || "-";
          const secondary =
            data?.task === "resume_parse"
              ? String((parsed["核心技能"] || []).length)
              : String((parsed["必备技能"] || []).length);
          return `
            <section class="batch-result-card">
              <div class="batch-result-head">
                <strong>样本 ${item.index + 1}</strong>
                <span class="status-pill ${item.ok ? "ok" : "error"}">${item.ok ? "成功" : "失败"}</span>
              </div>
              <div class="batch-result-meta">
                <span>${data?.task === "resume_parse" ? "目标岗位" : "岗位方向"}：${escapeHtml(primary)}</span>
                <span>${data?.task === "resume_parse" ? "技能数" : "必备技能数"}：${escapeHtml(secondary)}</span>
              </div>
            </section>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderBatchMatchStructured(data) {
  const items = Array.isArray(data?.items) ? data.items : [];
  overviewLabel1.textContent = "总样本";
  overviewLabel2.textContent = "成功数";
  overviewLabel3.textContent = "失败数";
  overviewLabel4.textContent = "批量任务";
  overviewValue1.textContent = String(data?.total ?? items.length);
  overviewValue2.textContent = String(data?.success_count ?? 0);
  overviewValue3.textContent = String((data?.total ?? items.length) - (data?.success_count ?? 0));
  overviewValue4.textContent = "批量匹配";

  structuredView.innerHTML = `
    <div class="batch-result-list">
      ${items
        .map((item) => {
          const rule = item.rule_result || {};
          const analysis = item.analysis || {};
          return `
            <section class="batch-result-card">
              <div class="batch-result-head">
                <strong>匹配样本 ${item.index + 1}</strong>
                <span class="status-pill ${item.ok ? "ok" : "error"}">${item.ok ? "成功" : "失败"}</span>
              </div>
              <div class="batch-result-meta">
                <span>匹配等级：${escapeHtml(rule["匹配等级"] || "-")}</span>
                <span>匹配分数：${escapeHtml(String(rule["匹配分数"] ?? "-"))}</span>
                <span>命中技能：${escapeHtml(String((rule["命中技能"] || []).length))}</span>
                <span>缺失技能：${escapeHtml(String((rule["缺失技能"] || []).length))}</span>
              </div>
              <div class="batch-mini-grid">
                <div class="batch-mini-item">
                  <span>岗位方向</span>
                  <strong>${escapeHtml(item.jd_parse?.["岗位方向"] || "-")}</strong>
                </div>
                <div class="batch-mini-item">
                  <span>简历目标岗位</span>
                  <strong>${escapeHtml(item.resume_parse?.["目标岗位"] || "-")}</strong>
                </div>
                <div class="batch-mini-item">
                  <span>匹配结论</span>
                  <strong>${escapeHtml(analysis["匹配结论"] || "-")}</strong>
                </div>
                <div class="batch-mini-item">
                  <span>推荐方向数</span>
                  <strong>${escapeHtml(String((analysis["推荐投递岗位方向"] || []).length))}</strong>
                </div>
              </div>
            </section>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderStructured(payload) {
  if (state.task === "match") {
    renderMatchStructured(payload);
  } else {
    if (state.requestMode === "batch") {
      renderBatchParseStructured(payload);
      return;
    }
    renderParseStructured(payload, state.task);
  }
}

function updateServiceMeta(data) {
  backendName.textContent = data.backend || "-";
  modelPath.textContent = data.model_path || "-";
  adapterPath.textContent = data.adapter_path || "-";
  gpuState.textContent = data.cuda_available ? "CUDA 可用" : "CUDA 不可用";
  loadState.textContent = data.loaded ? "已加载" : "未加载";
}

function updateInputStats() {
  if (state.task === "match") {
    const jdText = jdInputText.value.trim();
    const resumeText = resumeInputText.value.trim();
    if (state.requestMode === "batch") {
      const jdCount = splitBatchText(jdText).length;
      const resumeCount = splitBatchText(resumeText).length;
      inputStats.textContent = `JD ${jdCount} 条 / 简历 ${resumeCount} 条`;
      return;
    }
    inputStats.textContent = `JD ${jdText.length} 字 / 简历 ${resumeText.length} 字`;
    return;
  }
  const text = inputText.value.trim();
  if (state.requestMode === "batch") {
    inputStats.textContent = `${splitBatchText(text).length} 条`;
    return;
  }
  inputStats.textContent = `${text.length} 字`;
}

function splitBatchText(text) {
  return text
    .split(/\n-{3,}\n/g)
    .map((item) => item.trim())
    .filter(Boolean);
}

async function checkHealth() {
  try {
    const response = await fetch(`${apiBase.value}/api/status`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    updateServiceMeta(data);
    setStatus(data.loaded ? "模型已加载" : "服务可用", "ok");
  } catch {
    setStatus("未连接", "error");
    backendName.textContent = "-";
    modelPath.textContent = "-";
    adapterPath.textContent = "-";
    gpuState.textContent = "-";
    loadState.textContent = "-";
  }
}

async function warmupModel() {
  warmupButton.disabled = true;
  warmupButton.textContent = "预热中";
  setStatus("加载模型", "ok");
  try {
    const response = await fetch(`${apiBase.value}/api/warmup`, { method: "POST" });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`);
    }
    updateServiceMeta(data);
    latency.textContent = data.latency_seconds ? `${data.latency_seconds}s` : "-";
    setStatus("模型已加载", "ok");
  } catch (error) {
    setStatus("预热失败", "error");
    rawView.textContent = JSON.stringify({ ok: false, error: error.message }, null, 2);
    setView("raw");
  } finally {
    warmupButton.disabled = false;
    warmupButton.textContent = "预热模型";
  }
}

async function parseText() {
  const isBatch = state.requestMode === "batch";
  const url =
    state.task === "match"
      ? `${apiBase.value}/${isBatch ? "api/batch_match" : "api/match"}`
      : `${apiBase.value}/${isBatch ? "api/batch_parse" : "api/parse"}`;
  const payload = buildPayload();
  const hasText = hasValidPayload(payload);
  if (!hasText) {
    rawView.textContent =
      state.task === "match"
        ? isBatch
          ? "请输入批量 JD 和简历文本"
          : "请输入 JD 和简历文本"
        : isBatch
          ? "请输入批量文本"
          : "请输入文本";
    setView("raw");
    return;
  }

  parseButton.disabled = true;
  parseButton.textContent = state.task === "match" ? "分析中" : "处理中";
  latency.textContent = "-";
  setStatus(state.task === "match" ? "匹配分析中" : "推理中", "ok");

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`);
    }
    state.lastPayload = data;
    latency.textContent = data.latency_seconds ? `${data.latency_seconds}s` : "-";
    rawView.textContent = JSON.stringify(data, null, 2);
    renderStructured(isBatch || state.task === "match" ? data : data.data);
    setStatus(data.ok ? "处理完成" : "处理失败", data.ok ? "ok" : "error");
    setView("structured");
    await checkHealth();
  } catch (error) {
    state.lastPayload = { ok: false, error: error.message };
    rawView.textContent = JSON.stringify(state.lastPayload, null, 2);
    renderStructured(null);
    setStatus("请求失败", "error");
    setView("raw");
  } finally {
    parseButton.disabled = false;
    parseButton.textContent = "开始结构化";
  }
}

function buildPayload() {
  if (state.task === "match") {
    if (state.requestMode === "batch") {
      const jdItems = splitBatchText(jdInputText.value.trim());
      const resumeItems = splitBatchText(resumeInputText.value.trim());
      const size = Math.min(jdItems.length, resumeItems.length);
      return {
        items: Array.from({ length: size }, (_, index) => ({
          jd_text: jdItems[index],
          resume_text: resumeItems[index],
        })),
        max_new_tokens: 1024,
      };
    }
    return {
      jd_text: jdInputText.value.trim(),
      resume_text: resumeInputText.value.trim(),
      max_new_tokens: 1024,
    };
  }

  if (state.requestMode === "batch") {
    return {
      task: state.task,
      texts: splitBatchText(inputText.value.trim()),
      max_new_tokens: 1024,
    };
  }

  return {
    task: state.task,
    text: inputText.value.trim(),
    max_new_tokens: 1024,
  };
}

function hasValidPayload(payload) {
  if (state.task === "match") {
    if (state.requestMode === "batch") {
      return Array.isArray(payload.items) && payload.items.length > 0;
    }
    return payload.jd_text && payload.resume_text;
  }
  if (state.requestMode === "batch") {
    return Array.isArray(payload.texts) && payload.texts.length > 0;
  }
  return payload.text;
}

async function copyJson() {
  const content = rawView.textContent || "{}";
  try {
    await navigator.clipboard.writeText(content);
    setStatus("JSON 已复制", "ok");
  } catch {
    setStatus("复制失败", "error");
  }
}

function fillExample() {
  if (state.task === "match") {
    if (state.requestMode === "batch") {
      jdInputText.value = [examples.match.jd, examples.jd_parse].join(BATCH_DELIMITER);
      resumeInputText.value = [examples.match.resume, examples.resume_parse].join(BATCH_DELIMITER);
    } else {
      jdInputText.value = examples.match.jd;
      resumeInputText.value = examples.match.resume;
    }
  } else {
    inputText.value =
      state.requestMode === "batch"
        ? [examples[state.task], examples[state.task]].join(BATCH_DELIMITER)
        : examples[state.task];
  }
  updateInputStats();
}

jdMode.addEventListener("click", () => setMode("jd_parse"));
resumeMode.addEventListener("click", () => setMode("resume_parse"));
matchMode.addEventListener("click", () => setMode("match"));
singleRequestMode.addEventListener("click", () => setRequestMode("single"));
batchRequestMode.addEventListener("click", () => setRequestMode("batch"));
structuredTab.addEventListener("click", () => setView("structured"));
rawTab.addEventListener("click", () => setView("raw"));
exampleButton.addEventListener("click", fillExample);
parseButton.addEventListener("click", parseText);
warmupButton.addEventListener("click", warmupModel);
copyJsonButton.addEventListener("click", copyJson);
apiBase.addEventListener("change", () => {
  localStorage.setItem("jobmatch_api_base", apiBase.value);
  checkHealth();
});
inputText.addEventListener("input", updateInputStats);
jdInputText.addEventListener("input", updateInputStats);
resumeInputText.addEventListener("input", updateInputStats);

apiBase.value = localStorage.getItem("jobmatch_api_base") || apiBase.value;
inputText.value = examples.jd_parse;
renderStructured(null);
setView("structured");
updateInputStats();
checkHealth();
