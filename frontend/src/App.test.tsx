import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import userEvent from "@testing-library/user-event";

describe("AI Voice Assistant", () => {
  beforeEach(() => { localStorage.clear(); vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline"))); });
  it("renders the dashboard without claiming services are online", async () => {
    render(<MemoryRouter><App /></MemoryRouter>);
    expect(screen.getByRole("heading", { name: "Hello!" })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Start listening" })).toHaveLength(2);
    expect(await screen.findByText("Unavailable")).toBeInTheDocument();
    expect(screen.queryByText("Online")).not.toBeInTheDocument();
  });

  it("renders the configured navigation", () => {
    render(<MemoryRouter><App /></MemoryRouter>);
    expect(screen.getByRole("link", { name: "Home" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Conversations" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Settings" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "About" })).toBeInTheDocument();
  });

  it("lets the user choose a transcription language for future voice turns", async () => {
    const user = userEvent.setup();
    render(<MemoryRouter initialEntries={["/settings"]}><App /></MemoryRouter>);
    const language = screen.getByRole("combobox", { name: "Transcription language" });
    await user.selectOptions(language, "hi");
    expect(language).toHaveValue("hi");
  });

  it("provides dedicated English and Hindi buttons and persists the choice", async () => {
    const user = userEvent.setup();
    render(<MemoryRouter><App /></MemoryRouter>);
    const english = screen.getByRole("button", { name: "Use English" });
    const hindi = screen.getByRole("button", { name: "Use Hindi" });
    expect(english).toHaveAttribute("aria-pressed", "true");
    expect(hindi).toHaveAttribute("aria-pressed", "false");
    expect(screen.queryByRole("button", { name: "Use Tamil" })).not.toBeInTheDocument();
    await user.click(hindi);
    expect(hindi).toHaveAttribute("aria-pressed", "true");
    expect(localStorage.getItem("trikon-language")).toBe("hi");
  });

  it("sends text to the backend and displays its actual response", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ status: "degraded", backend: { available: true }, services: { api: true, llm: true, stt: false, tts: false }, models: { llm: "qwen2.5:3b" } }), { status: 200 }))
      .mockResolvedValueOnce(new Response(
        '{"type":"text","delta":"A response "}\n{"type":"text","delta":"from Ollama"}\n{"type":"done","session_id":"test-session","response":"A response from Ollama","metrics":{"llm_ms":31,"total_ms":32}}\n',
        { status: 200, headers: { "Content-Type": "application/x-ndjson" } },
      ))
      .mockResolvedValueOnce(new Response(JSON.stringify({ session_id: "test-session", deleted: true }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<MemoryRouter><App /></MemoryRouter>);
    await user.type(screen.getByRole("textbox", { name: "Message" }), "Hello local model");
    await user.click(screen.getByRole("button", { name: "Send message" }));

    expect(await screen.findByText("A response from Ollama")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/api/chat/stream", expect.objectContaining({ method: "POST" }));
    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(await screen.findByText("No messages yet")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenLastCalledWith("/api/chat/test-session", expect.objectContaining({ method: "DELETE" }));
  });

  it("shows an Ollama error and never fabricates a reply", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ status: "degraded", backend: { available: true }, services: { api: true, llm: false, stt: false, tts: false } }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: "Ollama is offline." }), { status: 503 }));
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<MemoryRouter><App /></MemoryRouter>);
    await user.type(screen.getByRole("textbox", { name: "Message" }), "Hello");
    await user.click(screen.getByRole("button", { name: "Send message" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Ollama is offline.");
    expect(screen.queryByText("Hello! How can I help?")).not.toBeInTheDocument();
  });

  it("records audio, uploads it, and renders the returned transcript and audio control", async () => {
    const stopTrack = vi.fn();
    Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop: stopTrack }] }) } });
    class MockRecorder extends EventTarget {
      mimeType = "audio/webm";
      state: RecordingState = "inactive";
      start() { this.state = "recording"; }
      stop() {
        this.state = "inactive";
        const data = new Event("dataavailable");
        Object.defineProperty(data, "data", { value: new Blob(["recorded audio"], { type: this.mimeType }) });
        this.dispatchEvent(data);
        this.dispatchEvent(new Event("stop"));
      }
    }
    class MockAudio {
      paused = true;
      onended: (() => void) | null = null;
      onerror: (() => void) | null = null;
      async play() { this.paused = false; }
      pause() { this.paused = true; }
    }
    vi.stubGlobal("MediaRecorder", MockRecorder);
    vi.stubGlobal("Audio", MockAudio);
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ status: "degraded", backend: { available: true }, services: { api: true, llm: true, stt: true, tts: true } }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ session_id: "voice-session", transcript: "recognized words", response: "Spoken local response", audio_url: "/api/audio/reply.wav", metrics: { stt_ms: 10, llm_ms: 20, tts_ms: 30, total_ms: 62 } }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<MemoryRouter><App /></MemoryRouter>);
    await user.click(screen.getAllByRole("button", { name: "Start listening" })[0]);
    expect(await screen.findByText("Listening")).toBeInTheDocument();
    await user.click(screen.getAllByRole("button", { name: "Stop listening" })[0]);

    expect(await screen.findByText("recognized words")).toBeInTheDocument();
    expect(screen.getByText("Spoken local response")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Play response" })).toBeInTheDocument();
    expect(stopTrack).toHaveBeenCalled();
    expect(fetchMock).toHaveBeenLastCalledWith("/api/voice", expect.objectContaining({ method: "POST" }));
    await user.click(screen.getByRole("button", { name: "Play response" }));
    expect(await screen.findByText("Speaking")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Pause response" }));
    expect(await screen.findByText("Tap to speak")).toBeInTheDocument();
  });

  it("explains continuous hands-free listening and handles an unsupported microphone", async () => {
    vi.stubGlobal("MediaRecorder", undefined);
    const user = userEvent.setup();
    render(<MemoryRouter><App /></MemoryRouter>);
    expect(screen.getByText(/Enable hands-free once and start speaking naturally/)).toBeInTheDocument();
    expect(screen.queryByText(/Clap activation/)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Enable hands-free" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("does not support local microphone capture");
    expect(screen.getByRole("status", { name: /error/i })).toBeInTheDocument();
  });

  it("displays the cinematic intro on first visit and enters the dashboard on click", async () => {
    sessionStorage.removeItem("intro_seen");
    const user = userEvent.setup();
    render(<MemoryRouter initialEntries={["/"]}><App /></MemoryRouter>);

    expect(screen.getByText("AI VOICE ASSISTANT")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Meet Your AI Voice Assistant." })).toBeInTheDocument();
    expect(screen.getByText("A more natural way to interact with AI.")).toBeInTheDocument();

    const videoStage = screen.getByRole("region", { name: "Intro animation" });
    const video = videoStage.querySelector("video");
    expect(video).toHaveAttribute("src", "/videos/intro-sphere.mp4");
    expect(video).toHaveAttribute("autoplay");
    expect(video).toHaveProperty("muted", true);
    expect(video).toHaveAttribute("loop");
    expect(video).toHaveAttribute("playsinline");

    const enterBtn = screen.getByRole("button", { name: /enter the ai voice assistant/i });
    expect(enterBtn).toBeInTheDocument();
    expect(videoStage).toContainElement(enterBtn);
    await user.click(enterBtn);

    // After enter click and fade transition
    expect(sessionStorage.getItem("intro_seen")).toBe("1");
    expect(await screen.findByText("Tap to speak")).toBeInTheDocument();
  });
});

