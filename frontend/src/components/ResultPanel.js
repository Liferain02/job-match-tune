export default {
  name: "ResultPanel",
  props: {
    activeView: { type: String, required: true },
    rawText: { type: String, required: true },
    overviewItems: { type: Array, required: true },
    structuredPayload: { type: Object, default: null },
    task: { type: String, required: true },
    requestMode: { type: String, required: true },
  },
  emits: ["set-view", "copy-json"],
  computed: {
    isBatch() {
      return this.requestMode === "batch";
    },
    isMatch() {
      return this.task === "match";
    },
    parseData() {
      if (!this.structuredPayload) return null;
      if (this.isBatch || this.isMatch) return this.structuredPayload;
      return this.structuredPayload.data || this.structuredPayload;
    },
  },
  template: `
    <section class="panel result-panel">
      <div class="panel-head result-head">
        <div>
          <p class="panel-kicker">输出结果</p>
          <h2>结果视图</h2>
        </div>
        <div class="result-actions">
          <div class="result-tabs">
            <button class="result-tab" :class="{ active: activeView === 'structured' }" type="button" @click="$emit('set-view', 'structured')">结构化</button>
            <button class="result-tab" :class="{ active: activeView === 'raw' }" type="button" @click="$emit('set-view', 'raw')">原始 JSON</button>
          </div>
          <button class="button secondary compact" type="button" @click="$emit('copy-json')">复制 JSON</button>
        </div>
      </div>

      <div class="result-overview">
        <div v-for="item in overviewItems" :key="item.label" class="overview-card">
          <span>{{ item.label }}</span>
          <strong>{{ item.value }}</strong>
        </div>
      </div>

      <section v-show="activeView === 'structured'" id="structuredView" class="result-body active">
        <template v-if="!parseData">
          <div class="empty-state">等待结构化结果</div>
        </template>

        <template v-else-if="isBatch && !isMatch">
          <div class="batch-result-list">
            <section v-for="item in parseData.items || []" :key="item.index" class="batch-result-card">
              <div class="batch-result-head">
                <strong>样本 {{ item.index + 1 }}</strong>
                <span class="status-pill" :class="item.ok ? 'ok' : 'error'">{{ item.ok ? '成功' : '失败' }}</span>
              </div>
              <div class="batch-result-meta">
                <span>{{ task === 'resume_parse' ? '目标岗位' : '岗位方向' }}：{{ task === 'resume_parse' ? (item.data?.['目标岗位'] || '-') : (item.data?.['岗位方向'] || '-') }}</span>
                <span>{{ task === 'resume_parse' ? '技能数' : '必备技能数' }}：{{ task === 'resume_parse' ? ((item.data?.['核心技能'] || []).length) : ((item.data?.['必备技能'] || []).length) }}</span>
              </div>
            </section>
          </div>
        </template>

        <template v-else-if="isMatch">
          <div v-if="isBatch" class="batch-result-list">
            <section v-for="item in parseData.items || []" :key="item.index" class="batch-result-card">
              <div class="batch-result-head">
                <strong>匹配样本 {{ item.index + 1 }}</strong>
                <span class="status-pill" :class="item.ok ? 'ok' : 'error'">{{ item.ok ? '成功' : '失败' }}</span>
              </div>
              <div class="batch-result-meta">
                <span>匹配等级：{{ item.rule_result?.['匹配等级'] || '-' }}</span>
                <span>匹配分数：{{ item.rule_result?.['匹配分数'] ?? '-' }}</span>
                <span>命中技能：{{ (item.rule_result?.['命中技能'] || []).length }}</span>
                <span>缺失技能：{{ (item.rule_result?.['缺失技能'] || []).length }}</span>
              </div>
            </section>
          </div>
          <div v-else class="result-grid">
            <section class="result-card compact"><span class="card-label">匹配等级</span><strong>{{ parseData.rule_result?.['匹配等级'] || '-' }}</strong></section>
            <section class="result-card compact"><span class="card-label">匹配分数</span><strong>{{ parseData.rule_result?.['匹配分数'] ?? '-' }}</strong></section>
            <section class="result-card compact"><span class="card-label">岗位方向匹配</span><strong>{{ parseData.rule_result?.['岗位方向匹配'] ? '是' : '否' }}</strong></section>
            <section class="result-card compact"><span class="card-label">学历 / 经验</span><strong>{{ parseData.rule_result?.['学历匹配'] ? '学历匹配' : '学历缺口' }} / {{ parseData.rule_result?.['经验匹配'] ? '经验匹配' : '经验缺口' }}</strong></section>
            <section class="result-card span-two">
              <div class="card-head"><span class="card-label">JD 结构化解析</span></div>
              <div class="nested-result-grid">
                <div class="nested-metric"><span>岗位方向</span><strong>{{ parseData.jd_parse?.['岗位方向'] || '-' }}</strong></div>
                <div class="nested-metric"><span>经验要求</span><strong>{{ parseData.jd_parse?.['经验要求'] || '-' }}</strong></div>
                <div class="nested-metric"><span>学历要求</span><strong>{{ parseData.jd_parse?.['学历要求'] || '-' }}</strong></div>
              </div>
              <div class="subsection">
                <span class="card-label">必备技能</span>
                <div class="chip-list"><span v-for="item in parseData.jd_parse?.['必备技能'] || []" :key="item" class="chip">{{ item }}</span></div>
              </div>
              <div class="subsection">
                <span class="card-label">核心职责</span>
                <ul class="bullet-list"><li v-for="item in parseData.jd_parse?.['核心职责'] || []" :key="item">{{ item }}</li></ul>
              </div>
            </section>
            <section class="result-card span-two">
              <div class="card-head"><span class="card-label">简历结构化解析</span></div>
              <div class="nested-result-grid">
                <div class="nested-metric"><span>目标岗位</span><strong>{{ parseData.resume_parse?.['目标岗位'] || '-' }}</strong></div>
                <div class="nested-metric"><span>教育背景</span><strong>{{ (parseData.resume_parse?.['教育背景'] || []).length }}</strong></div>
                <div class="nested-metric"><span>项目经历</span><strong>{{ (parseData.resume_parse?.['项目经历'] || []).length }}</strong></div>
              </div>
              <div class="subsection">
                <span class="card-label">核心技能</span>
                <div class="chip-list"><span v-for="item in parseData.resume_parse?.['核心技能'] || []" :key="item" class="chip">{{ item }}</span></div>
              </div>
              <div class="subsection">
                <span class="card-label">优势标签</span>
                <div class="chip-list"><span v-for="item in parseData.resume_parse?.['优势标签'] || []" :key="item" class="chip">{{ item }}</span></div>
              </div>
              <div class="subsection">
                <span class="card-label">项目经历</span>
                <ul class="bullet-list"><li v-for="item in parseData.resume_parse?.['项目经历'] || []" :key="item">{{ item }}</li></ul>
              </div>
            </section>
            <section class="result-card"><div class="card-head"><span class="card-label">命中技能</span></div><div class="chip-list"><span v-for="item in parseData.rule_result?.['命中技能'] || []" :key="item" class="chip">{{ item }}</span></div></section>
            <section class="result-card"><div class="card-head"><span class="card-label">缺失技能</span></div><div class="chip-list"><span v-for="item in parseData.rule_result?.['缺失技能'] || []" :key="item" class="chip">{{ item }}</span></div></section>
            <section class="result-card"><div class="card-head"><span class="card-label">匹配优势</span></div><ul class="bullet-list"><li v-for="item in parseData.analysis?.['匹配优势'] || []" :key="item">{{ item }}</li></ul></section>
            <section class="result-card"><div class="card-head"><span class="card-label">主要短板</span></div><ul class="bullet-list"><li v-for="item in parseData.analysis?.['主要短板'] || []" :key="item">{{ item }}</li></ul></section>
            <section class="result-card span-two"><div class="card-head"><span class="card-label">匹配结论</span></div><div class="paragraph-block">{{ parseData.analysis?.['匹配结论'] || '-' }}</div></section>
            <section class="result-card"><div class="card-head"><span class="card-label">命中项目</span></div><ul class="bullet-list"><li v-for="item in parseData.rule_result?.['命中项目'] || []" :key="item">{{ item }}</li></ul></section>
            <section class="result-card"><div class="card-head"><span class="card-label">推荐投递岗位方向</span></div><div class="chip-list"><span v-for="item in parseData.analysis?.['推荐投递岗位方向'] || []" :key="item" class="chip">{{ item }}</span></div></section>
            <section class="result-card span-two"><div class="card-head"><span class="card-label">简历优化建议</span></div><ul class="bullet-list"><li v-for="item in parseData.analysis?.['简历优化建议'] || []" :key="item">{{ item }}</li></ul></section>
          </div>
        </template>

        <template v-else-if="task === 'resume_parse'">
          <div class="result-grid">
            <section class="result-card compact"><span class="card-label">目标岗位</span><strong>{{ parseData['目标岗位'] || '-' }}</strong></section>
            <section class="result-card compact"><span class="card-label">教育背景条目</span><strong>{{ (parseData['教育背景'] || []).length }}</strong></section>
            <section class="result-card"><div class="card-head"><span class="card-label">核心技能</span></div><div class="chip-list"><span v-for="item in parseData['核心技能'] || []" :key="item" class="chip">{{ item }}</span></div></section>
            <section class="result-card"><div class="card-head"><span class="card-label">优势标签</span></div><div class="chip-list"><span v-for="item in parseData['优势标签'] || []" :key="item" class="chip">{{ item }}</span></div></section>
            <section class="result-card span-two"><div class="card-head"><span class="card-label">教育背景</span></div><ul class="bullet-list"><li v-for="item in parseData['教育背景'] || []" :key="item">{{ item }}</li></ul></section>
            <section class="result-card span-two"><div class="card-head"><span class="card-label">实习经历</span></div><ul class="bullet-list"><li v-for="item in parseData['实习经历'] || []" :key="item">{{ item }}</li></ul></section>
            <section class="result-card span-two"><div class="card-head"><span class="card-label">项目经历</span></div><ul class="bullet-list"><li v-for="item in parseData['项目经历'] || []" :key="item">{{ item }}</li></ul></section>
          </div>
        </template>

        <template v-else>
          <div class="result-grid">
            <section class="result-card compact"><span class="card-label">岗位方向</span><strong>{{ parseData['岗位方向'] || '-' }}</strong></section>
            <section class="result-card compact"><span class="card-label">经验要求</span><strong>{{ parseData['经验要求'] || '-' }}</strong></section>
            <section class="result-card compact"><span class="card-label">学历要求</span><strong>{{ parseData['学历要求'] || '-' }}</strong></section>
            <section class="result-card"><div class="card-head"><span class="card-label">必备技能</span></div><div class="chip-list"><span v-for="item in parseData['必备技能'] || []" :key="item" class="chip">{{ item }}</span></div></section>
            <section class="result-card span-two"><div class="card-head"><span class="card-label">核心职责</span></div><ul class="bullet-list"><li v-for="item in parseData['核心职责'] || []" :key="item">{{ item }}</li></ul></section>
            <section class="result-card span-two"><div class="card-head"><span class="card-label">加分项</span></div><ul class="bullet-list"><li v-for="item in parseData['加分项'] || []" :key="item">{{ item }}</li></ul></section>
          </div>
        </template>
      </section>

      <pre v-show="activeView === 'raw'" class="result-body active">{{ rawText }}</pre>
    </section>
  `,
};
