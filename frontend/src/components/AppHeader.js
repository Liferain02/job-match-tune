export default {
  name: "AppHeader",
  props: {
    statusText: { type: String, required: true },
    statusMode: { type: String, default: "" },
    busy: { type: Boolean, default: false },
    warmupBusy: { type: Boolean, default: false },
  },
  emits: ["warmup", "parse"],
  template: `
    <header class="hero">
      <div class="hero-copy">
        <p class="hero-kicker">JobMatchTune</p>
        <h1>招聘文本结构化与人岗匹配工作台</h1>
        <p class="hero-summary">
          Qwen3-14B + LoRA + 规则后处理，支持 JD 解析、简历解析、人岗匹配，以及批量请求与文件接入。
        </p>
      </div>
      <div class="hero-status">
        <span class="status-pill" :class="statusMode">{{ statusText }}</span>
        <div class="hero-actions">
          <button class="button secondary" type="button" :disabled="warmupBusy" @click="$emit('warmup')">
            {{ warmupBusy ? '预热中' : '预热模型' }}
          </button>
          <button class="button primary" type="button" :disabled="busy" @click="$emit('parse')">
            {{ busy ? '处理中' : '开始结构化' }}
          </button>
        </div>
      </div>
    </header>
  `,
};
