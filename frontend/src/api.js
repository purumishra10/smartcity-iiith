const prefix = import.meta.env.VITE_API_URL || "";

async function parse(res) {
  const data = await res.json().catch(() => ({ detail: res.statusText }));
  if (!res.ok) {
    const detail = data.detail || "request_failed";
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function opts(init = {}) {
  return { credentials: "include", ...init };
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
  const res = await fetch(`${prefix}/api/analyze`, opts({ method: "POST", body }));
  return parse(res);
}

export async function listAnalyses() {
  const res = await fetch(`${prefix}/api/analyses`, opts());
  return parse(res);
}

export async function getAnalysis(id) {
  const res = await fetch(`${prefix}/api/analyses/${id}`, opts());
  return parse(res);
}

export async function getMe() {
  const res = await fetch(`${prefix}/api/me`, opts());
  return parse(res);
}

export async function signup(email, password) {
  const res = await fetch(
    `${prefix}/api/auth/signup`,
    opts({
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    })
  );
  return parse(res);
}

export async function login(email, password) {
  const res = await fetch(
    `${prefix}/api/auth/login`,
    opts({
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    })
  );
  return parse(res);
}

export async function logout() {
  const res = await fetch(`${prefix}/api/auth/logout`, opts({ method: "POST" }));
  return parse(res);
}

export function reportUrl(id) {
  return assetUrl(`/api/analyses/${id}/report.pdf`);
}

export async function health() {
  const res = await fetch(`${prefix}/health`, opts());
  return parse(res);
}
