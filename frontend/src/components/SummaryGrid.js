export default {
  name: "SummaryGrid",
  props: {
    summaryItems: { type: Array, required: true },
  },
  template: `
    <section class="summary-grid">
      <article v-for="item in summaryItems" :key="item.label" class="summary-card">
        <span>{{ item.label }}</span>
        <strong>{{ item.value }}</strong>
      </article>
    </section>
  `,
};
