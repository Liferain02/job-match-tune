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
};

const state = {
  task: "jd_parse",
  activeView: "structured",
  lastPayload: null,
};

const apiBase = document.querySelector("#apiBase");
const inputText = document.querySelector("#inputText");
const status = document.querySelector("#status");
const latency = document.querySelector("#latency");
const parseButton = document.querySelector("#parseButton");
const exampleButton = document.querySelector("#exampleButton");
const warmupButton = document.querySelector("#warmupButton");
const copyJsonButton = document.querySelector("#copyJsonButton");
const jdMode = document.querySelector("#jdMode");
const resumeMode = document.querySelector("#resumeMode");
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
const overviewDirection = document.querySelector("#overviewDirection");
const overviewRespCount = document.querySelector("#overviewRespCount");
const overviewSkillCount = document.querySelector("#overviewSkillCount");
const overviewBonusCount = document.querySelector("#overviewBonusCount");

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

function setMode(task) {
  state.task = task;
  jdMode.classList.toggle("active", task === "jd_parse");
  resumeMode.classList.toggle("active", task === "resume_parse");
  taskHint.textContent = task === "jd_parse" ? "当前任务：JD 结构化" : "当前任务：简历结构化";
}

function setView(view) {
  state.activeView = view;
  structuredTab.classList.toggle("active", view === "structured");
  rawTab.classList.toggle("active", view === "raw");
  structuredView.classList.toggle("active", view === "structured");
  rawView.classList.toggle("active", view === "raw");
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
  overviewDirection.textContent = "-";
  overviewRespCount.textContent = "0";
  overviewSkillCount.textContent = "0";
  overviewBonusCount.textContent = "0";
}

function renderStructured(data) {
  if (!data || typeof data !== "object") {
    resetOverview();
    structuredView.innerHTML = '<div class="empty-state">等待结构化结果</div>';
    return;
  }

  const responsibilities = data["核心职责"] || [];
  const skills = data["必备技能"] || [];
  const bonus = data["加分项"] || [];

  overviewDirection.textContent = data["岗位方向"] || "-";
  overviewRespCount.textContent = String(responsibilities.length);
  overviewSkillCount.textContent = String(skills.length);
  overviewBonusCount.textContent = String(bonus.length);

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

function updateServiceMeta(data) {
  backendName.textContent = data.backend || "-";
  modelPath.textContent = data.model_path || "-";
  adapterPath.textContent = data.adapter_path || "-";
  gpuState.textContent = data.cuda_available ? "CUDA 可用" : "CUDA 不可用";
  loadState.textContent = data.loaded ? "已加载" : "未加载";
}

function updateInputStats() {
  inputStats.textContent = `${inputText.value.trim().length} 字`;
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
  const text = inputText.value.trim();
  if (!text) {
    rawView.textContent = "请输入文本";
    setView("raw");
    return;
  }

  parseButton.disabled = true;
  parseButton.textContent = "处理中";
  latency.textContent = "-";
  setStatus("推理中", "ok");

  try {
    const response = await fetch(`${apiBase.value}/api/parse`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task: state.task,
        text,
        max_new_tokens: 1024,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`);
    }
    state.lastPayload = data;
    latency.textContent = data.latency_seconds ? `${data.latency_seconds}s` : "-";
    rawView.textContent = JSON.stringify(data, null, 2);
    renderStructured(data.data);
    setStatus(data.ok ? "解析完成" : "解析失败", data.ok ? "ok" : "error");
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

async function copyJson() {
  const content = rawView.textContent || "{}";
  try {
    await navigator.clipboard.writeText(content);
    setStatus("JSON 已复制", "ok");
  } catch {
    setStatus("复制失败", "error");
  }
}

jdMode.addEventListener("click", () => setMode("jd_parse"));
resumeMode.addEventListener("click", () => setMode("resume_parse"));
structuredTab.addEventListener("click", () => setView("structured"));
rawTab.addEventListener("click", () => setView("raw"));
exampleButton.addEventListener("click", () => {
  inputText.value = examples[state.task];
  updateInputStats();
});
parseButton.addEventListener("click", parseText);
warmupButton.addEventListener("click", warmupModel);
copyJsonButton.addEventListener("click", copyJson);
apiBase.addEventListener("change", () => {
  localStorage.setItem("jobmatch_api_base", apiBase.value);
  checkHealth();
});
inputText.addEventListener("input", updateInputStats);

apiBase.value = localStorage.getItem("jobmatch_api_base") || apiBase.value;
inputText.value = examples.jd_parse;
updateInputStats();
renderStructured(null);
setView("structured");
checkHealth();
