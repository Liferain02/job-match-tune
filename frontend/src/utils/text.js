export function splitBatchText(text) {
  return String(text || "")
    .split(/\n-{3,}\n/g)
    .map((item) => item.trim())
    .filter(Boolean);
}

export function prettyJson(value) {
  return JSON.stringify(value ?? {}, null, 2);
}
