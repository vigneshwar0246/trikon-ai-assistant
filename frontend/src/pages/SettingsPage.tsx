import { Cpu, HardDrive, Headphones, Languages, MemoryStick, Mic2, RefreshCw, Server } from "lucide-react";
import { useEffect, useState } from "react";
import { Card } from "../components/ui/card";
import { useVoice } from "../context/VoiceContext";
import { getDiagnostics, type RuntimeDiagnostics } from "../lib/api";

export function SettingsPage() {
  const { health, language, setLanguage } = useVoice();
  const [diagnostics, setDiagnostics] = useState<RuntimeDiagnostics | null>(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => { void getDiagnostics().then(setDiagnostics).catch(() => setDiagnostics(null)); }, []);
  const runRequirements = async () => {
    setLoading(true);
    try { setDiagnostics(await getDiagnostics(true)); }
    finally { setLoading(false); }
  };
  const settings = [
    { icon: Mic2, title: "Speech recognition", value: health.models?.stt ?? "Not reported", note: health.stt === "ready" ? "Local model files are available" : "Configured by the local backend" },
    { icon: Server, title: "Language model", value: health.models?.llm ?? "Not reported", note: health.llm === "ready" ? "Available through Ollama" : "Model readiness requires the backend check" },
    { icon: Headphones, title: "Speech synthesis", value: health.models?.tts?.replaceAll("_", " ") ?? "Not reported", note: health.tts === "ready" ? "Local voice files are available" : "Selected by the local backend" },
  ];
  const voiceSuffix = (code: "en" | "hi") => health.languageSupport?.[code]?.tts === false ? " (voice missing)" : "";
  const bytes = (value: number | null | undefined) => value ? `${(value / 1024 ** 3).toFixed(1)} GB` : "Unknown";
  return (
    <div className="route-page">
      <header className="page-header"><p className="overline">Preferences</p><h1>Settings</h1><p>Model names and availability come from the backend. Language selection persists locally and applies to future text and voice turns.</p></header>
      <Card className="settings-list">
        {settings.map(({ icon: Icon, title, value, note }) => <div className="setting-row" key={title}><span className="setting-icon"><Icon /></span><div><strong>{title}</strong><p>{note}</p></div><span className="setting-value">{value}</span></div>)}
        <div className="setting-row"><span className="setting-icon"><Languages /></span><div><strong>Assistant language</strong><p>Choose English or Hindi before speaking. The microphone and assistant will use your selected language.</p></div><select aria-label="Transcription language" className="setting-select" value={language} onChange={(event) => setLanguage(event.target.value)}><option value="en">English{voiceSuffix("en")}</option><option value="hi">Hindi{voiceSuffix("hi")}</option></select></div>
        <div className="setting-row"><span className="setting-icon"><Server /></span><div><strong>CUDA diagnostics</strong><p>{diagnostics?.cuda.detail ?? "Diagnostics unavailable until the backend responds."}</p></div><span className="setting-value">{diagnostics?.cuda.inference_verified ? "Verified" : diagnostics ? "Not verified" : "Unknown"}</span></div>
        <div className="setting-row"><span className="setting-icon"><Mic2 /></span><div><strong>STT device</strong><p>Last successful inference: {diagnostics?.stt.last_successful_device ?? "Not measured in this process"}</p></div><span className="setting-value">{diagnostics?.stt.configured_device ?? "Unknown"}</span></div>
      </Card>
      <div className="requirements-heading"><div><p className="overline">Live diagnostics</p><h2>System requirements</h2><p>Detected hardware is reported separately from verified operational readiness.</p></div><button type="button" onClick={() => void runRequirements()} disabled={loading}><RefreshCw className={loading ? "spinning" : ""} />{loading ? "Checking…" : "Run system requirements"}</button></div>
      <Card className="settings-list requirements-list">
        <div className="setting-row"><span className="setting-icon"><Cpu /></span><div><strong>CPU</strong><p>{diagnostics?.system.cpu ?? "Not checked"}</p></div><span className="setting-value">{diagnostics?.system.logical_cpu_count ?? "?"} logical cores</span></div>
        <div className="setting-row"><span className="setting-icon"><MemoryStick /></span><div><strong>System memory</strong><p>Available {bytes(diagnostics?.system.ram_available_bytes)}</p></div><span className="setting-value">{bytes(diagnostics?.system.ram_total_bytes)}</span></div>
        <div className="setting-row"><span className="setting-icon"><HardDrive /></span><div><strong>System disk</strong><p>Free {bytes(diagnostics?.system.disk_free_bytes)}</p></div><span className="setting-value">{bytes(diagnostics?.system.disk_total_bytes)}</span></div>
        <div className="setting-row"><span className="setting-icon"><Cpu /></span><div><strong>Physical GPU</strong><p>{diagnostics?.system.gpu.detail ?? "Not checked"}</p></div><span className="setting-value">{diagnostics?.system.gpu.devices[0]?.name ?? diagnostics?.system.gpu.status ?? "Unknown"}</span></div>
        {diagnostics && Object.entries(diagnostics.checks).map(([name, check]) => <div className="setting-row diagnostic-row" key={name}><span className={`diagnostic-badge ${check.status.toLowerCase().replace(" ", "-")}`}>{check.status}</span><div><strong>{name.replaceAll("_", " ")}</strong><p>{check.detail}{check.suggestion ? ` ${check.suggestion}` : ""}</p></div><span className="setting-value">{diagnostics.timestamp.slice(11, 19)} UTC</span></div>)}
      </Card>
    </div>
  );
}
