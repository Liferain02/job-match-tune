import { computed, onMounted, reactive } from "https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js";

import AppHeader from "./components/AppHeader.js";
import ControlPanel from "./components/ControlPanel.js";
import ResultPanel from "./components/ResultPanel.js";
import { examples } from "./config/examples.js";
import {
  getStatus,
  parseSingle,
  parseBatch,
  parseResumeFile,
  parseJdFile,
  matchSingle,
  matchFiles,
  matchBatch,
} from "./services/api.js";
import { buildMarkdownReport, prettyJson, splitBatchText } from "./utils/text.js";

const defaultApiBase = `${window.location.protocol}//${window.location.hostname || "localhost"}:8000`;

function createOverviewItems(state) {
  if (state.requestMode === "batch") {
    return [
      { label: "总样本", value: "0" },
      { label: "成功数", value: "0" },
      { label: "失败数", value: "0" },
      { label: state.task === "match" ? "批量任务" : "任务类型", value: state.task === "match" ? "批量匹配" : "批量解析" },
    ];
  }
  if (state.task === "match") {
    return [
      { label: "匹配等级", value: "-" },
      { label: "匹配分数", value: "0" },
      { label: "命中技能", value: "0" },
      { label: "缺失技能", value: "0" },
    ];
  }
  if (state.task === "resume_parse") {
    return [
      { label: "目标岗位", value: "-" },
      { label: "技能条目", value: "0" },
      { label: "项目经历", value: "0" },
      { label: "优势标签", value: "0" },
    ];
  }
  return [
    { label: "岗位方向", value: "-" },
    { label: "职责条目", value: "0" },
    { label: "技能条目", value: "0" },
    { label: "加分项", value: "0" },
  ];
}

export default {
  name: "App",
  components: { AppHeader, ControlPanel, ResultPanel },
  setup() {
    const state = reactive({
      apiBase: defaultApiBase,
      task: "match",
      requestMode: "single",
      activeView: "structured",
      singleText: "",
      jdText: "",
      resumeText: "",
      jdOcrText: "",
      resumeOcrText: "",
      selectedJdFile: null,
      selectedJdFileName: "",
      selectedResumeFile: null,
      selectedResumeFileName: "",
      busy: false,
      statusText: "未连接",
      statusMode: "",
      latencyText: "-",
      backendName: "-",
      matchParseMode: "-",
      gpuState: "-",
      loadState: "-",
      modelPath: "-",
      adapterPath: "-",
      rawText: "{}",
      lastPayload: null,
    });

    const overviewItems = computed(() => {
      if (!state.lastPayload) {
        return createOverviewItems(state);
      }
      if (state.requestMode === "batch") {
        const total = state.lastPayload.total ?? (state.lastPayload.items || []).length ?? 0;
        const success = state.lastPayload.success_count ?? 0;
        return [
          { label: "总样本", value: String(total) },
          { label: "成功数", value: String(success) },
          { label: "失败数", value: String(total - success) },
          { label: state.task === "match" ? "批量任务" : "任务类型", value: state.task === "match" ? "批量匹配" : "批量解析" },
        ];
      }
      if (state.task === "match") {
        const rule = state.lastPayload.rule_result || {};
        return [
          { label: "匹配等级", value: rule["匹配等级"] || "-" },
          { label: "匹配分数", value: String(rule["匹配分数"] ?? "-") },
          { label: "命中技能", value: String((rule["命中技能"] || []).length) },
          { label: "缺失技能", value: String((rule["缺失技能"] || []).length) },
        ];
      }
      const parsed = state.lastPayload.data || state.lastPayload;
      if (state.task === "resume_parse") {
        return [
          { label: "目标岗位", value: parsed["目标岗位"] || "-" },
          { label: "技能条目", value: String((parsed["核心技能"] || []).length) },
          { label: "项目经历", value: String((parsed["项目经历"] || []).length) },
          { label: "优势标签", value: String((parsed["优势标签"] || []).length) },
        ];
      }
      return [
        { label: "岗位方向", value: parsed["岗位方向"] || "-" },
        { label: "职责条目", value: String((parsed["核心职责"] || []).length) },
        { label: "技能条目", value: String((parsed["必备技能"] || []).length) },
        { label: "加分项", value: String((parsed["加分项"] || []).length) },
      ];
    });

    const inputStats = computed(() => {
      if (state.task === "match") {
        if (state.requestMode === "batch") {
          return `JD ${splitBatchText(state.jdText).length} 条 / 简历 ${splitBatchText(state.resumeText).length} 条`;
        }
        return `JD ${state.jdText.trim().length} 字 / 简历 ${state.resumeText.trim().length} 字`;
      }
      if (state.task === "resume_parse" && state.requestMode === "single" && state.selectedResumeFileName) {
        return `文件：${state.selectedResumeFileName}`;
      }
      if (state.task === "jd_parse" && state.requestMode === "single" && state.selectedJdFileName) {
        return `文件：${state.selectedJdFileName}`;
      }
      if (state.requestMode === "batch") {
        return `${splitBatchText(state.singleText).length} 条`;
      }
      return `${state.singleText.trim().length} 字`;
    });

    const taskHint = computed(() => {
      if (state.task === "jd_parse") return "当前任务：JD 结构化";
      if (state.task === "resume_parse") return "当前任务：简历结构化";
      return "当前任务：人岗匹配分析";
    });

    const batchTipVisible = computed(() => state.requestMode === "batch");

    function setStatus(text, mode = "") {
      state.statusText = text;
      state.statusMode = mode;
    }

    async function refreshStatus(updateUiStatus = true) {
      try {
        const data = await getStatus(state.apiBase);
        state.backendName = data.backend || "-";
        state.matchParseMode = data.match_parse_mode || "-";
        state.modelPath = data.model_path || "-";
        state.adapterPath = data.adapter_path || "-";
        state.gpuState = data.cuda_available ? "CUDA 可用" : "CUDA 不可用";
        state.loadState = data.loaded ? "已加载" : "未加载";
        if (updateUiStatus) {
          setStatus(data.loaded ? "模型已加载" : "服务可用", "ok");
        }
      } catch {
        state.backendName = "-";
        state.matchParseMode = "-";
        state.modelPath = "-";
        state.adapterPath = "-";
        state.gpuState = "-";
        state.loadState = "-";
        if (updateUiStatus) {
          setStatus("未连接", "error");
        }
      }
    }

    function fillExample() {
      state.selectedResumeFile = null;
      state.selectedResumeFileName = "";
      state.selectedJdFile = null;
      state.selectedJdFileName = "";
      state.jdOcrText = "";
      state.resumeOcrText = "";
      if (state.task === "match") {
        state.jdText = examples.match.jd;
        state.resumeText = examples.match.resume;
        return;
      }
      state.singleText = examples[state.task];
    }

    function setTask(task) {
      state.task = task;
      state.lastPayload = null;
      state.rawText = "{}";
      state.activeView = "structured";
    }

    function setRequestMode(mode) {
      state.requestMode = mode;
      state.lastPayload = null;
      state.rawText = "{}";
      if (mode === "batch") {
        state.selectedResumeFile = null;
        state.selectedResumeFileName = "";
        state.selectedJdFile = null;
        state.selectedJdFileName = "";
      }
    }

    function handleResumeFileChange(event) {
      const file = event.target.files?.[0] || null;
      state.selectedResumeFile = file;
      state.selectedResumeFileName = file ? file.name : "";
    }

    function handleJdFileChange(event) {
      const file = event.target.files?.[0] || null;
      state.selectedJdFile = file;
      state.selectedJdFileName = file ? file.name : "";
    }

    function hasValidPayload() {
      if (state.task === "match") {
        if (state.requestMode === "batch") {
          return splitBatchText(state.jdText).length > 0 && splitBatchText(state.resumeText).length > 0;
        }
        const hasJd = Boolean(state.jdText.trim() || state.selectedJdFile);
        const hasResume = Boolean(state.resumeText.trim() || state.selectedResumeFile);
        return hasJd && hasResume;
      }
      if (state.task === "resume_parse" && state.requestMode === "single" && state.selectedResumeFile) {
        return true;
      }
      if (state.task === "jd_parse" && state.requestMode === "single" && state.selectedJdFile) {
        return true;
      }
      if (state.requestMode === "batch") {
        return splitBatchText(state.singleText).length > 0;
      }
      return state.singleText.trim();
    }

    async function handleParse() {
      if (!hasValidPayload()) {
        let message = "请先提供需要解析的内容";
        if (state.task === "match") {
          const hasJd = Boolean(state.jdText.trim() || state.selectedJdFile);
          message = hasJd ? "请再提供候选人简历" : "请先提供目标岗位 JD";
        }
        state.rawText = prettyJson({ ok: false, error: message });
        setStatus(message, "error");
        return;
      }
      state.busy = true;
      state.latencyText = "-";
      const needsModelLoad = state.loadState !== "已加载";
      if (needsModelLoad) {
        setStatus("首次运行：正在加载模型并处理，约需 2–3 分钟", "ok");
      } else {
        setStatus(state.task === "match" ? "正在匹配，通常约需 1 分钟" : "正在解析", "ok");
      }
      try {
        let data;
        if (state.task === "match") {
          if (state.requestMode === "batch") {
            const jdItems = splitBatchText(state.jdText);
            const resumeItems = splitBatchText(state.resumeText);
            const size = Math.min(jdItems.length, resumeItems.length);
            const items = Array.from({ length: size }, (_, index) => ({
              jd_text: jdItems[index],
              resume_text: resumeItems[index],
            }));
            data = await matchBatch(state.apiBase, items);
          } else if (state.selectedJdFile || state.selectedResumeFile) {
            data = await matchFiles(state.apiBase, {
              jdText: state.jdText.trim(),
              resumeText: state.resumeText.trim(),
              jdFile: state.selectedJdFile,
              resumeFile: state.selectedResumeFile,
              jdOcrText: state.jdOcrText,
              resumeOcrText: state.resumeOcrText,
            });
          } else {
            data = await matchSingle(state.apiBase, state.jdText.trim(), state.resumeText.trim());
          }
        } else if (state.task === "resume_parse" && state.requestMode === "single" && state.selectedResumeFile) {
          data = await parseResumeFile(state.apiBase, state.selectedResumeFile, state.resumeOcrText);
        } else if (state.task === "jd_parse" && state.requestMode === "single" && state.selectedJdFile) {
          data = await parseJdFile(state.apiBase, state.selectedJdFile, state.jdOcrText);
        } else if (state.requestMode === "batch") {
          data = await parseBatch(state.apiBase, state.task, splitBatchText(state.singleText));
        } else {
          data = await parseSingle(state.apiBase, state.task, state.singleText.trim());
        }
        state.lastPayload = data;
        state.rawText = prettyJson(data);
        state.latencyText = data.latency_seconds ? `${data.latency_seconds}s` : "-";
        state.activeView = "structured";
        const completedText = state.task === "match" ? "匹配完成" : "解析完成";
        const elapsedText = data.latency_seconds ? `，用时 ${data.latency_seconds}s` : "";
        setStatus(data.ok === false ? "处理失败" : `${completedText}${elapsedText}`, data.ok === false ? "error" : "ok");
        await refreshStatus(false);
      } catch (error) {
        state.lastPayload = null;
        state.rawText = prettyJson({ ok: false, error: error.message });
        state.activeView = "raw";
        setStatus("请求失败", "error");
      } finally {
        state.busy = false;
      }
    }

    async function copyJson() {
      try {
        await navigator.clipboard.writeText(state.rawText || "{}");
        setStatus("JSON 已复制", "ok");
      } catch {
        setStatus("复制失败", "error");
      }
    }

    function downloadReport() {
      if (!state.lastPayload) {
        setStatus("暂无报告", "error");
        return;
      }
      const markdown = buildMarkdownReport(state.lastPayload, state.task, state.requestMode);
      const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const fileName = `jobmatch-${state.task}-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.md`;
      const link = document.createElement("a");
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setStatus("报告已下载", "ok");
    }

    onMounted(() => {
      refreshStatus();
    });

    return {
      state,
      overviewItems,
      inputStats,
      taskHint,
      batchTipVisible,
      refreshStatus,
      handleParse,
      fillExample,
      setTask,
      setRequestMode,
      handleJdFileChange,
      handleResumeFileChange,
      copyJson,
      downloadReport,
    };
  },
  template: `
    <main class="app-shell">
      <AppHeader
        :status-text="state.statusText"
        :status-mode="state.statusMode"
      />

      <section class="workspace-grid">
        <ControlPanel
          :api-base="state.apiBase"
          :task="state.task"
          :request-mode="state.requestMode"
          :single-text="state.singleText"
          :jd-text="state.jdText"
          :resume-text="state.resumeText"
          :jd-ocr-text="state.jdOcrText"
          :resume-ocr-text="state.resumeOcrText"
          :input-stats="inputStats"
          :task-hint="taskHint"
          :batch-tip-visible="batchTipVisible"
          :model-path="state.modelPath"
          :adapter-path="state.adapterPath"
          :backend-name="state.backendName"
          :match-parse-mode="state.matchParseMode"
          :selected-jd-file-name="state.selectedJdFileName"
          :selected-file-name="state.selectedResumeFileName"
          :busy="state.busy"
          :status-text="state.statusText"
          :status-mode="state.statusMode"
          :model-loaded="state.loadState === '已加载'"
          @update:api-base="state.apiBase = $event"
          @set-task="setTask"
          @set-request-mode="setRequestMode"
          @update:single-text="state.singleText = $event"
          @update:jd-text="state.jdText = $event"
          @update:resume-text="state.resumeText = $event"
          @update:jd-ocr-text="state.jdOcrText = $event"
          @update:resume-ocr-text="state.resumeOcrText = $event"
          @jd-file-change="handleJdFileChange"
          @resume-file-change="handleResumeFileChange"
          @fill-example="fillExample"
          @submit="handleParse"
        />

        <ResultPanel
          :active-view="state.activeView"
          :raw-text="state.rawText"
          :overview-items="overviewItems"
          :structured-payload="state.lastPayload"
          :task="state.task"
          :request-mode="state.requestMode"
          @set-view="state.activeView = $event"
          @copy-json="copyJson"
          @download-report="downloadReport"
        />
      </section>
    </main>
  `,
};
