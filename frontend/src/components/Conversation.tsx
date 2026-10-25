import { Bot, Mic, Pause, Play, Send, Sparkles, Trash2, UserRound } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useVoice } from "../context/VoiceContext";
import type { Message } from "../types";
import { Button } from "./ui/button";
import { Card } from "./ui/card";

function MessageAudio({ url, onSpeakingChange }: { url: string; onSpeakingChange: (speaking: boolean) => void }) {
  const audio = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [playbackError, setPlaybackError] = useState(false);
  useEffect(() => {
    const stop = () => { if (audio.current) { audio.current.pause(); audio.current.currentTime = 0; } setPlaying(false); onSpeakingChange(false); };
    window.addEventListener("ai-voice-stop-speech", stop);
    return () => { window.removeEventListener("ai-voice-stop-speech", stop); audio.current?.pause(); onSpeakingChange(false); };
  }, [onSpeakingChange]);
  const toggle = () => {
    audio.current ??= new Audio(url);
    const reset = () => { setPlaying(false); onSpeakingChange(false); };
    audio.current.onended = reset;
    audio.current.onerror = () => { reset(); setPlaybackError(true); };
    if (audio.current.paused) {
      setPlaybackError(false);
      void audio.current.play().then(() => { setPlaying(true); onSpeakingChange(true); }).catch(() => { reset(); setPlaybackError(true); });
    } else { audio.current.pause(); reset(); }
  };
  return <><button className="audio-control" onClick={toggle}>{playing ? <Pause /> : <Play />}<span>{playing ? "Pause response" : "Play response"}</span></button>{playbackError && <span className="playback-error" role="alert">Audio playback failed.</span>}</>;
}

function MessageBubble({ message, onSpeakingChange }: { message: Message; onSpeakingChange: (speaking: boolean) => void }) {
  const assistant = message.role === "assistant";
  return (
    <article className={`message ${message.role}`}>
      <span className="message-avatar">{assistant ? <Bot /> : <UserRound />}</span>
      <div className="message-content"><span className="message-label">{assistant ? "AI Voice Assistant" : "You"}</span><p>{message.content}</p>{assistant && message.audioUrl && <MessageAudio url={message.audioUrl} onSpeakingChange={onSpeakingChange} />}</div>
    </article>
  );
}

export function Conversation({ expanded = false }: { expanded?: boolean }) {
  const { messages, phase, error, language, submitText, toggleRecording, clearConversation, setSpeaking } = useVoice();
  const [draft, setDraft] = useState("");
  const feed = useRef<HTMLDivElement>(null);
  const busy = phase === "processing" || phase === "generating";
  const samples = language === "hi"
    ? ["भारत की राजधानी क्या है?", "Tell me the capital of India in Hindi", "पानी कैसे बचाएं?"]
    : ["What is the capital of India?", "Explain local AI simply", "How can I save water?"];
  useEffect(() => { feed.current?.scrollTo({ top: feed.current.scrollHeight, behavior: "smooth" }); }, [messages, phase]);
  const submit = (event: React.FormEvent) => { event.preventDefault(); if (draft.trim()) { void submitText(draft); setDraft(""); } };
  return (
    <Card className={`conversation ${expanded ? "conversation-expanded" : ""}`}>
      <div className="conversation-header"><div><p className="overline">Conversation</p><h2>{expanded ? "Conversation workspace" : "Continue in your own words"}</h2></div><Button variant="ghost" onClick={() => void clearConversation()} disabled={!messages.length || phase === "processing" || phase === "listening" || phase === "requesting-permission"}><Trash2 />Clear</Button></div>
      <div className="message-feed" ref={feed} aria-live="polite">
        {!messages.length && <div className="conversation-empty"><span><Sparkles /></span><strong>No messages yet</strong><p>Speak, type, or choose a sample question.</p><div className="conversation-suggestions">{samples.map((sample) => <button type="button" key={sample} disabled={busy} onClick={() => void submitText(sample)}>{sample}</button>)}</div></div>}
        {messages.map((message) => <MessageBubble key={message.id} message={message} onSpeakingChange={setSpeaking} />)}
        {phase === "processing" && <div className="thinking-row" role="status"><span /><span /><span /><p>Processing with local models</p></div>}
      </div>
      {error && <div className="inline-error" role="alert">{error}</div>}
      <form className="composer" onSubmit={submit}>
        <label htmlFor={expanded ? "message-expanded" : "message"}>Message</label>
        <div className="composer-field">
          <input id={expanded ? "message-expanded" : "message"} value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Ask me anything" disabled={busy} />
          <button type="button" className={`composer-icon ${phase === "listening" ? "recording" : ""}`} onClick={() => void toggleRecording()} aria-label={phase === "listening" ? "Stop listening" : "Start listening"}><Mic /></button>
          <button type="submit" className="composer-send" disabled={!draft.trim() || busy} aria-label="Send message"><Send /></button>
        </div>
      </form>
    </Card>
  );
}
