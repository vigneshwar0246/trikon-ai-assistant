import type { Metrics, ServiceState, SystemHealth } from "../types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");

type UnknownRecord = Record<string, unknown>;

function readPath(source: UnknownRecord, paths: string[]): unknown {
  for (const path of paths) {
    const result = path.split(".").reduce<unknown>((value, key) => {
      return value && typeof value === "object" ? (value as UnknownRecord)[key] : undefined;
    }, source);
    if (result !== undefined && result !== null) return result;
  }
}

function readNumber(source: UnknownRecord, paths: string[]) {
  const value = readPath(source, paths);
  return typeof value === "number" ? value : typeof value === "string" && value.trim() ? Number(value) : undefined;
}

function parseMetrics(data: UnknownRecord, measuredTotal: number): Metrics {
  return {
    sttMs: readNumber(data, ["metrics.stt_ms", "timings.stt_ms", "stt_ms"]),
    llmMs: readNumber(data, ["metrics.llm_ms", "timings.llm_ms", "llm_ms"]),
    ttsMs: readNumber(data, ["metrics.tts_ms", "timings.tts_ms", "tts_ms"]),
    totalMs: readNumber(data, ["metrics.total_ms", "timings.total_ms", "total_ms"]) ?? measuredTotal,
    firstTokenMs: readNumber(data, ["metrics.first_token_ms", "timings.first_token_ms", "first_token_ms"]),
    firstAudioMs: readNumber(data, ["metrics.first_audio_ms", "timings.first_audio_ms", "first_audio_ms"]),
  };
}

async function parseResponse(response: Response) {
  const data = await response.json().catch(() => ({})) as UnknownRecord;
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(typeof detail === "string" ? detail : `The local API returned ${response.status}.`);
  }
  return data;
}

function resolveAudioUrl(value: unknown): string | undefined {
  if (typeof value !== "string" || !value.trim()) return undefined;
  try { return new URL(value, API_BASE || window.location.origin).toString(); }
  catch { return undefined; }
}

export async function sendText(message: string, sessionId: string, language?: string) {
  const started = performance.now();
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId, language: language || undefined }),
  });
  const data = await parseResponse(response);
  const content = readPath(data, ["response", "message", "text", "assistant_text"]);
  if (typeof content !== "string" || !content.trim()) throw new Error("The API response did not contain assistant text.");
  const audio = readPath(data, ["audio_url", "audio.url", "tts_url"]);
  return { content, audioUrl: resolveAudioUrl(audio), sessionId: String(data.session_id ?? sessionId), metrics: parseMetrics(data, performance.now() - started) };
}

export interface TextStreamEvent {
  type: "text" | "done" | "error";
  session_id?: string;
  delta?: string;
  response?: string;
  detail?: string;
  action?: string;
  metrics?: UnknownRecord;
}

async function readNdjson<T extends { type: string; detail?: string }>(
  response: Response, signal: AbortSignal, onEvent: (event: T) => void,
): Promise<void> {
  if (!response.ok || !response.body) {
    await parseResponse(response);
    throw new Error("The streaming response was unavailable.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line) as T;
        if (event.type === "error") throw new Error(event.detail || "The streaming request failed.");
        onEvent(event);
      }
      if (done) break;
    }
    if (buffer.trim()) {
      const event = JSON.parse(buffer) as T;
      if (event.type === "error") throw new Error(event.detail || "The streaming request failed.");
      onEvent(event);
    }
  } finally {
    if (signal.aborted) await reader.cancel().catch(() => undefined);
  }
}

export async function streamText(
  message: string, sessionId: string, language: string | undefined, signal: AbortSignal,
  onEvent: (event: TextStreamEvent) => void,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST", signal, headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId, language: language || undefined }),
  });
  await readNdjson(response, signal, onEvent);
}

export async function sendVoice(blob: Blob, sessionId: string, responseLanguage?: string, signal?: AbortSignal) {
  const body = new FormData();
  const extension = blob.type.includes("ogg") ? "ogg" : blob.type.includes("mp4") ? "mp4" : "webm";
  body.append("audio", blob, `voice.${extension}`);
  body.append("session_id", sessionId);
  if (responseLanguage) body.append("response_language", responseLanguage);
  const started = performance.now();
  const data = await parseResponse(await fetch(`${API_BASE}/api/voice`, { method: "POST", body, signal }));
  const transcript = readPath(data, ["transcript", "user_text", "input"]);
  const content = readPath(data, ["response", "message", "text", "assistant_text"]);
  const audio = readPath(data, ["audio_url", "audio.url", "tts_url"]);
  const audioUrl = resolveAudioUrl(audio);
  if (typeof transcript !== "string" || !transcript.trim()) throw new Error("The API response did not contain a transcription.");
  if (typeof content !== "string" || !content.trim()) throw new Error("The API response did not contain assistant text.");
  return {
    transcript,
    content,
    audioUrl,
    action: typeof data.action === "string" ? data.action : undefined,
    language: typeof data.language === "string" ? data.language : undefined,
    languageConfidence: typeof data.language_confidence === "number" ? data.language_confidence : undefined,
    speechAvailable: data.speech_available !== false,
    sessionId: String(data.session_id ?? sessionId),
    metrics: parseMetrics(data, performance.now() - started),
  };
}

export interface VoiceStreamEvent {
  type: "transcript" | "text" | "audio" | "done" | "error";
  session_id?: string;
  transcript?: string;
  delta?: string;
  response?: string;
  url?: string;
  detail?: string;
  action?: string;
  language?: string;
  language_confidence?: number;
  speech_available?: boolean;
  speech_detail?: string;
  metrics?: UnknownRecord;
}

export async function streamVoice(
  blob: Blob,
  sessionId: string,
  responseLanguage: string | undefined,
  signal: AbortSignal,
  onEvent: (event: VoiceStreamEvent) => void,
): Promise<void> {
  const body = new FormData();
  const extension = blob.type.includes("ogg") ? "ogg" : blob.type.includes("mp4") ? "mp4" : "webm";
  body.append("audio", blob, `voice.${extension}`);
  body.append("session_id", sessionId);
  if (responseLanguage) body.append("response_language", responseLanguage);
  const response = await fetch(`${API_BASE}/api/voice/stream`, { method: "POST", body, signal });
  await readNdjson(response, signal, (event: VoiceStreamEvent) => {
    if (event.url) event.url = resolveAudioUrl(event.url);
    onEvent(event);
  });
}

export async function deleteSession(sessionId: string): Promise<void> {
  await parseResponse(await fetch(`${API_BASE}/api/chat/${encodeURIComponent(sessionId)}`, { method: "DELETE" }));
}

function toServiceState(value: unknown): ServiceState {
  if (value === true || ["ready", "online", "healthy", "ok"].includes(String(value).toLowerCase())) return "ready";
  if (value === undefined || value === null) return "unknown";
  return "unavailable";
}

export async function getHealth(signal?: AbortSignal): Promise<SystemHealth> {
  const data = await parseResponse(await fetch(`${API_BASE}/api/health`, { signal }));
  const services = (data.services ?? data.components ?? {}) as UnknownRecord;
  const backend = (data.backend ?? {}) as UnknownRecord;
  const models = (data.models ?? {}) as UnknownRecord;
  return {
    api: toServiceState(services.api ?? backend.available),
    stt: toServiceState(services.stt),
    llm: toServiceState(services.llm),
    tts: toServiceState(services.tts),
    models: {
      stt: typeof models.stt === "string" ? models.stt : undefined,
      llm: typeof models.llm === "string" ? models.llm : undefined,
      tts: typeof models.tts === "string" ? models.tts : undefined,
      language: typeof models.language === "string" ? models.language : undefined,
    },
    languageSupport: data.language_support && typeof data.language_support === "object"
      ? data.language_support as SystemHealth["languageSupport"] : undefined,
    checkedAt: new Date().toISOString(),
  };
}

export interface RuntimeDiagnostics {
  timestamp: string;
  checks: Record<string, { status: "PASS" | "FAIL" | "WARNING" | "UNKNOWN" | "NOT CONFIGURED"; detail: string; suggestion?: string }>;
  system: {
    cpu: string; logical_cpu_count: number | null; ram_total_bytes: number | null; ram_available_bytes: number | null;
    disk_total_bytes: number; disk_free_bytes: number; python: string; platform: string;
    gpu: { status: string; detail: string; devices: { name: string; memory_total_mb: number; memory_free_mb: number; driver: string }[] };
  };
  stt: { configured_device: string; active_device: string; last_successful_device: string | null };
  cuda: { device_count: number; cublas12_found: boolean; inference_verified: boolean; detail: string };
  languages: Record<string, { stt: boolean; llm: boolean; tts: boolean }>;
  ollama_loaded_models: { name: string; size_vram_bytes: number }[];
}

export async function getDiagnostics(verifyInference = false): Promise<RuntimeDiagnostics> {
  const suffix = verifyInference ? "?verify_inference=true" : "";
  return await parseResponse(await fetch(`${API_BASE}/api/diagnostics${suffix}`)) as unknown as RuntimeDiagnostics;
}
