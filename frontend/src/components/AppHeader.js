export default {
  name: "AppHeader",
  props: {
    statusText: { type: String, required: true },
    statusMode: { type: String, default: "" },
  },
  template: `
    <header class="hero">
      <div class="hero-copy">
        <p class="hero-kicker">JobMatchTune</p>
        <h1>看清岗位与简历是否匹配</h1>
        <p class="hero-summary">
          输入 JD 和简历，先看匹配结论，再核对技能、方向、学历与经验依据，最后获取简历优化建议。
        </p>
      </div>
      <div class="hero-status">
        <span class="status-pill" :class="statusMode">{{ statusText }}</span>
      </div>
    </header>
  `,
};
