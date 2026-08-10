export default {
  name: "ControlPanel",
  props: {
    apiBase: { type: String, required: true },
    task: { type: String, required: true },
    requestMode: { type: String, required: true },
    singleText: { type: String, required: true },
    jdText: { type: String, required: true },
    resumeText: { type: String, required: true },
    jdOcrText: { type: String, required: true },
    resumeOcrText: { type: String, required: true },
    inputStats: { type: String, required: true },
    taskHint: { type: String, required: true },
    batchTipVisible: { type: Boolean, required: true },
    modelPath: { type: String, required: true },
    adapterPath: { type: String, required: true },
    backendName: { type: String, required: true },
    matchParseMode: { type: String, required: true },
    selectedJdFileName: { type: String, default: "" },
    selectedFileName: { type: String, default: "" },
    busy: { type: Boolean, default: false },
    statusText: { type: String, required: true },
    statusMode: { type: String, default: "" },
    modelLoaded: { type: Boolean, default: false },
  },
  emits: [
    "update:apiBase",
    "set-task",
    "set-request-mode",
    "update:singleText",
    "update:jdText",
    "update:resumeText",
    "update:jdOcrText",
    "update:resumeOcrText",
    "jd-file-change",
    "resume-file-change",
    "fill-example",
    "submit",
  ],
  computed: {
    showJdFile() {
      return this.task === "jd_parse" && this.requestMode === "single";
    },
    showResumeFile() {
      return this.task === "resume_parse" && this.requestMode === "single";
    },
    showMatchFileInputs() {
      return this.task === "match" && this.requestMode === "single";
    },
    showMatchInputs() {
      return this.task === "match";
    },
    singleInputLabel() {
      return this.task === "resume_parse" ? "简历文本" : "JD 文本";
    },
    submitLabel() {
      if (this.busy) {
        if (!this.modelLoaded) return "首次运行中，约需 2–3 分钟";
        return this.task === "match" ? "正在匹配，约需 1 分钟" : "正在解析";
      }
      if (!this.modelLoaded) {
        return this.task === "match" ? "加载模型并开始匹配" : "加载模型并开始解析";
      }
      return this.task === "match" ? "开始匹配" : "开始解析";
    },
  },
  template: `
    <section class="panel control-panel">
      <div class="panel-head">
        <div>
          <p class="panel-kicker">{{ task === 'match' ? '核心流程' : '专项工具' }}</p>
          <h2>{{ task === 'match' ? '开始匹配' : '单项解析' }}</h2>
        </div>
      </div>

      <ol v-if="task === 'match'" class="flow-steps" aria-label="匹配步骤">
        <li><span>1</span><strong>提供岗位</strong></li>
        <li><span>2</span><strong>提供简历</strong></li>
        <li><span>3</span><strong>核对结果</strong></li>
      </ol>

      <div class="mode-row task-mode-row">
        <div class="segmented" role="tablist" aria-label="任务类型">
          <button class="segment" :class="{ active: task === 'match' }" type="button" @click="$emit('set-task', 'match')">人岗匹配</button>
          <button class="segment" :class="{ active: task === 'jd_parse' }" type="button" @click="$emit('set-task', 'jd_parse')">JD 解析</button>
          <button class="segment" :class="{ active: task === 'resume_parse' }" type="button" @click="$emit('set-task', 'resume_parse')">简历解析</button>
        </div>
        <button class="button secondary compact" type="button" @click="$emit('fill-example')">填入示例</button>
      </div>

      <template v-if="!showMatchInputs">
        <label class="field">
          <span>{{ singleInputLabel }}</span>
          <small class="field-note">可直接粘贴文本；单条模式也可上传文件，选择文件后以文件内容为准。</small>
          <textarea
            :value="singleText"
            spellcheck="false"
            :placeholder="task === 'resume_parse' ? '粘贴需要解析的简历内容' : '粘贴需要解析的岗位描述'"
            @input="$emit('update:singleText', $event.target.value)"
          ></textarea>
        </label>
      </template>

      <section v-if="showJdFile" class="file-upload-panel">
        <label class="field field-compact">
          <span>或上传 JD 文件</span>
          <span class="file-picker"><span>{{ selectedJdFileName || '选择文件' }}</span><input class="visually-hidden" type="file" accept=".txt,.md,.docx,.pdf,.png,.jpg,.jpeg,.webp,.bmp" @change="$emit('jd-file-change', $event)" /></span>
          <small v-if="selectedJdFileName" class="source-notice">已选择 {{ selectedJdFileName }}，本次将优先读取文件。</small>
        </label>
        <details v-if="selectedJdFileName" class="inline-details">
          <summary>扫描件 OCR 补充文本</summary>
          <label class="field field-compact">
            <span class="field-note">仅图片或扫描 PDF 无法提取正文时填写</span>
            <textarea class="compact-textarea small-textarea" spellcheck="false" :value="jdOcrText" @input="$emit('update:jdOcrText', $event.target.value)"></textarea>
          </label>
        </details>
      </section>

      <section v-if="showResumeFile" class="file-upload-panel">
        <label class="field field-compact">
          <span>或上传简历文件</span>
          <span class="file-picker"><span>{{ selectedFileName || '选择文件' }}</span><input class="visually-hidden" type="file" accept=".txt,.md,.docx,.pdf,.png,.jpg,.jpeg,.webp,.bmp" @change="$emit('resume-file-change', $event)" /></span>
          <small v-if="selectedFileName" class="source-notice">已选择 {{ selectedFileName }}，本次将优先读取文件。</small>
        </label>
        <details v-if="selectedFileName" class="inline-details">
          <summary>扫描件 OCR 补充文本</summary>
          <label class="field field-compact">
            <span class="field-note">仅图片或扫描 PDF 无法提取正文时填写</span>
            <textarea class="compact-textarea small-textarea" spellcheck="false" :value="resumeOcrText" @input="$emit('update:resumeOcrText', $event.target.value)"></textarea>
          </label>
        </details>
      </section>

      <section v-if="showMatchInputs" class="dual-input-grid">
        <article class="input-card">
          <div class="input-card-head">
            <span class="step-number">1</span>
            <div><strong>目标岗位 JD</strong><small>岗位职责、要求越完整，依据越可靠</small></div>
          </div>
          <textarea class="compact-textarea" spellcheck="false" placeholder="粘贴目标岗位描述" :value="jdText" @input="$emit('update:jdText', $event.target.value)"></textarea>
          <label v-if="showMatchFileInputs" class="file-choice">
            <span>或上传 JD 文件</span>
            <span class="file-picker"><span>{{ selectedJdFileName || '选择文件' }}</span><input class="visually-hidden" type="file" accept=".txt,.md,.docx,.pdf,.png,.jpg,.jpeg,.webp,.bmp" @change="$emit('jd-file-change', $event)" /></span>
            <small v-if="selectedJdFileName" class="source-notice">将优先使用 {{ selectedJdFileName }}</small>
          </label>
          <details v-if="showMatchFileInputs && selectedJdFileName" class="inline-details">
            <summary>扫描件 OCR 补充文本</summary>
            <textarea class="compact-textarea small-textarea" spellcheck="false" :value="jdOcrText" @input="$emit('update:jdOcrText', $event.target.value)"></textarea>
          </details>
        </article>

        <article class="input-card">
          <div class="input-card-head">
            <span class="step-number">2</span>
            <div><strong>候选人简历</strong><small>支持文本、DOCX、PDF 和图片</small></div>
          </div>
          <textarea class="compact-textarea" spellcheck="false" placeholder="粘贴候选人简历内容" :value="resumeText" @input="$emit('update:resumeText', $event.target.value)"></textarea>
          <label v-if="showMatchFileInputs" class="file-choice">
            <span>或上传简历文件</span>
            <span class="file-picker"><span>{{ selectedFileName || '选择文件' }}</span><input class="visually-hidden" type="file" accept=".txt,.md,.docx,.pdf,.png,.jpg,.jpeg,.webp,.bmp" @change="$emit('resume-file-change', $event)" /></span>
            <small v-if="selectedFileName" class="source-notice">将优先使用 {{ selectedFileName }}</small>
          </label>
          <details v-if="showMatchFileInputs && selectedFileName" class="inline-details">
            <summary>扫描件 OCR 补充文本</summary>
            <textarea class="compact-textarea small-textarea" spellcheck="false" :value="resumeOcrText" @input="$emit('update:resumeOcrText', $event.target.value)"></textarea>
          </details>
        </article>
      </section>

      <p v-if="task === 'match' && requestMode === 'single'" class="privacy-note">
        文件仅用于本次解析，当前服务不会主动保存上传原件；请勿上传未经授权的真实简历。
      </p>

      <div v-if="batchTipVisible" class="batch-tip">
        批量模式使用 <code>---</code> 分隔每条样本。匹配模式下，JD 与简历按顺序一一对应。
      </div>

      <details class="advanced-settings">
        <summary>高级设置</summary>
        <div class="mode-row advanced-mode-row">
          <span class="setting-label">处理方式</span>
          <div class="segmented" role="tablist" aria-label="请求方式">
            <button class="segment" :class="{ active: requestMode === 'single' }" type="button" @click="$emit('set-request-mode', 'single')">单条</button>
            <button class="segment" :class="{ active: requestMode === 'batch' }" type="button" @click="$emit('set-request-mode', 'batch')">批量</button>
          </div>
        </div>
        <label class="field">
          <span>API 地址</span>
          <input :value="apiBase" @input="$emit('update:apiBase', $event.target.value)" />
        </label>
        <div class="service-meta">
          <div class="meta-item"><span>推理后端</span><strong>{{ backendName || '-' }}</strong></div>
          <div class="meta-item"><span>匹配解析</span><strong>{{ matchParseMode === 'parallel' ? 'JD / 简历并行' : 'JD / 简历串行' }}</strong></div>
          <div class="meta-item"><span>模型路径</span><strong>{{ modelPath || '-' }}</strong></div>
          <div class="meta-item"><span>Adapter</span><strong>{{ adapterPath || '-' }}</strong></div>
        </div>
      </details>

      <div class="submit-row">
        <span class="form-status" :class="statusMode">{{ statusText }}</span>
        <button class="button primary submit-button" type="button" :disabled="busy" @click="$emit('submit')">
          {{ submitLabel }}
        </button>
      </div>

      <div class="editor-footer">
        <span>{{ inputStats }}</span>
        <span>{{ taskHint }}</span>
      </div>
    </section>
  `,
};
