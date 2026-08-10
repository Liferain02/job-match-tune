export async function fetchJson(url, init = {}) {
  const response = await fetch(url, init);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `HTTP ${response.status}`);
  }
  return data;
}

export function getStatus(baseUrl) {
  return fetchJson(`${baseUrl}/api/status`);
}

export function parseSingle(baseUrl, task, text, maxNewTokens = 1024) {
  return fetchJson(`${baseUrl}/api/parse`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task, text, max_new_tokens: maxNewTokens }),
  });
}

export function parseBatch(baseUrl, task, texts, maxNewTokens = 1024) {
  return fetchJson(`${baseUrl}/api/batch_parse`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task, texts, max_new_tokens: maxNewTokens }),
  });
}

export function parseResumeFile(baseUrl, file, ocrText = "", maxNewTokens = 1024) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("max_new_tokens", String(maxNewTokens));
  if (ocrText.trim()) {
    formData.append("ocr_text", ocrText.trim());
  }
  return fetchJson(`${baseUrl}/api/resume_file_parse`, {
    method: "POST",
    body: formData,
  });
}

export function parseJdFile(baseUrl, file, ocrText = "", maxNewTokens = 1024) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("max_new_tokens", String(maxNewTokens));
  if (ocrText.trim()) {
    formData.append("ocr_text", ocrText.trim());
  }
  return fetchJson(`${baseUrl}/api/jd_file_parse`, {
    method: "POST",
    body: formData,
  });
}

export function matchSingle(baseUrl, jdText, resumeText, maxNewTokens = 1024) {
  return fetchJson(`${baseUrl}/api/match`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jd_text: jdText, resume_text: resumeText, max_new_tokens: maxNewTokens }),
  });
}

export function matchFiles(baseUrl, payload, maxNewTokens = 1024) {
  const formData = new FormData();
  formData.append("jd_text", payload.jdText || "");
  formData.append("resume_text", payload.resumeText || "");
  formData.append("max_new_tokens", String(maxNewTokens));
  if (payload.jdFile) {
    formData.append("jd_file", payload.jdFile);
  }
  if (payload.resumeFile) {
    formData.append("resume_file", payload.resumeFile);
  }
  if ((payload.jdOcrText || "").trim()) {
    formData.append("jd_ocr_text", payload.jdOcrText.trim());
  }
  if ((payload.resumeOcrText || "").trim()) {
    formData.append("resume_ocr_text", payload.resumeOcrText.trim());
  }
  return fetchJson(`${baseUrl}/api/match_files`, {
    method: "POST",
    body: formData,
  });
}

export function matchBatch(baseUrl, items, maxNewTokens = 1024) {
  return fetchJson(`${baseUrl}/api/batch_match`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items, max_new_tokens: maxNewTokens }),
  });
}
