export default {
  name: "ControlPanel",
  props: {
    apiBase: { type: String, required: true },
    task: { type: String, required: true },
    requestMode: { type: String, required: true },
    singleText: { type: String, required: true },
    jdText: { type: String, required: true },
    resumeText: { type: String, required: true },
    resumeOcrText: { type: String, required: true },
    inputStats: { type: String, required: true },
    taskHint: { type: String, required: true },
    batchTipVisible: { type: Boolean, required: true },
    modelPath: { type: String, required: true },
    adapterPath: { type: String, required: true },
    selectedFileName: { type: String, default: "" },
  },
  emits: [
    "update:apiBase",
    "set-task",
    "set-request-mode",
    "update:singleText",
    "update:jdText",
    "update:resumeText",
    "update:resumeOcrText",
    "resume-file-change",
    "fill-example",
  ],
  computed: {
    showResumeFile() {
      return this.task === "resume_parse" && this.requestMode === "single";
    },
    showMatchInputs() {
      return this.task === "match";
    },
    singleInputLabel() {
      return this.task === "resume_parse" ? "简历文本" : "输入文本";
    },
  },
  template: `
    <section class="panel control-panel">
      <div class="panel-head">
        <div>
          <p class="panel-kicker">输入配置</p>
          <h2>请求面板</h2>
        </div>
      </div>

      <label class="field">
        <span>API 地址</span>
        <input :value="apiBase" @input="$emit('update:apiBase', $event.target.value)" />
      </label>

      <div class="mode-row">
        <div class="segmented" role="tablist" aria-label="task">
          <button class="segment" :class="{ active: task === 'jd_parse' }" type="button" @click="$emit('set-task', 'jd_parse')">JD 解析</button>
          <button class="segment" :class="{ active: task === 'resume_parse' }" type="button" @click="$emit('set-task', 'resume_parse')">简历解析</button>
          <button class="segment" :class="{ active: task === 'match' }" type="button" @click="$emit('set-task', 'match')">人岗匹配</button>
        </div>
        <button class="button secondary compact" type="button" @click="$emit('fill-example')">填入示例</button>
      </div>

      <div class="mode-row">
        <div class="segmented" role="tablist" aria-label="request-mode">
          <button class="segment" :class="{ active: requestMode === 'single' }" type="button" @click="$emit('set-request-mode', 'single')">单条</button>
          <button class="segment" :class="{ active: requestMode === 'batch' }" type="button" @click="$emit('set-request-mode', 'batch')">批量</button>
        </div>
      </div>

      <div class="service-meta">
        <div class="meta-item">
          <span>模型路径</span>
          <strong>{{ modelPath || '-' }}</strong>
        </div>
        <div class="meta-item">
          <span>Adapter</span>
          <strong>{{ adapterPath || '-' }}</strong>
        </div>
      </div>

      <template v-if="!showMatchInputs">
        <label class="field">
          <span>{{ singleInputLabel }}</span>
          <textarea :value="singleText" spellcheck="false" @input="$emit('update:singleText', $event.target.value)"></textarea>
        </label>
      </template>

      <section v-if="showResumeFile" class="file-upload-panel">
        <label class="field field-compact">
          <span>简历文件</span>
          <input type="file" accept=".txt,.md,.docx,.pdf,.png,.jpg,.jpeg,.webp,.bmp" @change="$emit('resume-file-change', $event)" />
          <small v-if="selectedFileName" class="field-note">已选择：{{ selectedFileName }}</small>
        </label>
        <label class="field field-compact">
          <span>OCR 文本（可选）</span>
          <textarea
            class="compact-textarea small-textarea"
            spellcheck="false"
            placeholder="图片简历或扫描版 PDF 如果已经有 OCR 文本，可以直接粘贴到这里。"
            :value="resumeOcrText"
            @input="$emit('update:resumeOcrText', $event.target.value)"
          ></textarea>
        </label>
      </section>

      <section v-if="showMatchInputs" class="dual-input-grid">
        <label class="field field-compact">
          <span>JD 文本</span>
          <textarea class="compact-textarea" spellcheck="false" :value="jdText" @input="$emit('update:jdText', $event.target.value)"></textarea>
        </label>
        <label class="field field-compact">
          <span>简历文本</span>
          <textarea class="compact-textarea" spellcheck="false" :value="resumeText" @input="$emit('update:resumeText', $event.target.value)"></textarea>
        </label>
      </section>

      <div v-if="batchTipVisible" class="batch-tip">
        批量模式使用 <code>---</code> 分隔每条样本。匹配模式下，JD 与简历按顺序一一对应。
      </div>

      <div class="editor-footer">
        <span>{{ inputStats }}</span>
        <span>{{ taskHint }}</span>
      </div>
    </section>
  `,
};
