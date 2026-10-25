import { afterEach, describe, expect, it, vi } from "vitest";
import { getHealth, sendText, sendVoice, streamVoice } from "./api";

afterEach(() => vi.unstubAllGlobals());

describe("local API client", () => {
  it("preserves Hindi stream text across split UTF-8 bytes and forwards the Hindi selection", async () => {
    const text = "भारत की राजधानी नई दिल्ली है।";
    const bytes = new TextEncoder().encode(JSON.stringify({ type: "done", response: text }) + "\n");
    const body = new ReadableStream({ start(controller) {
      for (let i = 0; i < bytes.length; i += 2) controller.enqueue(bytes.slice(i, i + 2));
      controller.close();
    } });
    const fetchMock = vi.fn().mockResolvedValue(new Response(body));
    vi.stubGlobal("fetch", fetchMock);
    const received: string[] = [];
    await streamVoice(new Blob(["audio"]), "session", "hi", new AbortController().signal,
      (event) => { if (event.response) received.push(event.response); });
    expect(received).toEqual([text]);
    expect((fetchMock.mock.calls[0][1].body as FormData).get("response_language")).toBe("hi");
  });

  it("keeps API availability separate from degraded model readiness", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      status: "degraded",
      backend: { available: true },
      services: { api: true, llm: false, stt: false, tts: false },
      models: { llm: "qwen2.5:3b" },
    }), { status: 200 })));

    await expect(getHealth()).resolves.toMatchObject({ api: "ready", llm: "unavailable", models: { llm: "qwen2.5:3b" } });
  });

  it("returns measured chat data and preserves server error details", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ session_id: "s1", response: "Actual local answer", metrics: { llm_ms: 24, total_ms: 26 } }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: "Ollama is offline." }), { status: 503 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(sendText("hello", "s1")).resolves.toMatchObject({ content: "Actual local answer", sessionId: "s1", metrics: { llmMs: 24, totalMs: 26 } });
    await expect(sendText("hello", "s1")).rejects.toThrow("Ollama is offline.");
  });

  it("uploads browser audio and resolves backend playback links", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      transcript: "hello", response: "spoken answer", session_id: "session-2", audio_url: "/api/audio/reply.wav",
      metrics: { stt_ms: 10, llm_ms: 20, tts_ms: 30, total_ms: 62 },
    }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await sendVoice(new Blob(["audio"], { type: "audio/webm;codecs=opus" }), "session-1", "hi");
    expect(result).toMatchObject({ transcript: "hello", content: "spoken answer", sessionId: "session-2", audioUrl: expect.stringContaining("/api/audio/reply.wav") });
    const form = fetchMock.mock.calls[0][1]?.body as FormData;
    expect((form.get("audio") as File).name).toBe("voice.webm");
    expect(form.get("session_id")).toBe("session-1");
    expect(form.get("response_language")).toBe("hi");
  });

  it("does not invent a transcript or audio when the voice API omits them", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ response: "answer" }), { status: 200 })));
    await expect(sendVoice(new Blob(["audio"], { type: "audio/webm" }), "session"))
      .rejects.toThrow("The API response did not contain a transcription.");
  });

  it("parses ordered incremental voice events and forwards cancellation", async () => {
    const encoder = new TextEncoder();
    const events = [
      { type: "transcript", transcript: "hello", language: "en" },
      { type: "text", delta: "Hello there." },
      { type: "audio", url: "/api/audio/chunk.wav", index: 0 },
      { type: "done", response: "Hello there.", metrics: { first_token_ms: 12 } },
    ];
    const body = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(events.map((event) => JSON.stringify(event)).join("\n") + "\n"));
        controller.close();
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, { status: 200 })));
    const received: string[] = [];
    const controller = new AbortController();

    await streamVoice(new Blob(["audio"], { type: "audio/webm" }), "session", "en", controller.signal,
      (event) => received.push(event.type));

    expect(received).toEqual(["transcript", "text", "audio", "done"]);
  });
});
