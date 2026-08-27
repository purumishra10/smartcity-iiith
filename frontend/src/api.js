const prefix = import.meta.env.VITE_API_URL || "";

async function parse(res) {
  const data = await res.json().catch(() => ({ detail: res.statusText }));
  if (!res.ok) {
    const detail = data.detail || "request_failed";
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

export function assetUrl(path) {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${prefix}${path}`;
}

export async function analyzeImage(file, context) {
  const body = new FormData();
  body.append("file", file);
  if (context) body.append("context", context);
  const res = await fetch(`${prefix}/api/analyze`, { method: "POST", body });
  return parse(res);
}

export async function listAnalyses() {
  const res = await fetch(`${prefix}/api/analyses`);
  return parse(res);
}

export async function getAnalysis(id) {
  const res = await fetch(`${prefix}/api/analyses/${id}`);
  return parse(res);
}

export async function health() {
  const res = await fetch(`${prefix}/health`);
  return parse(res);
}
