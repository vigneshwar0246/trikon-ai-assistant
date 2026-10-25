import { Languages } from "lucide-react";
import { useVoice } from "../context/VoiceContext";
import { warmLanguageModel } from "../lib/api";

const languages = [
  { code: "en", label: "English", native: "English" },
  { code: "hi", label: "Hindi", native: "हिन्दी" },
] as const;

export function LanguageSwitcher() {
  const { language, setLanguage, phase } = useVoice();
  const busy = ["listening", "transcribing", "generating", "processing", "speaking"].includes(phase);
  return (
    <section className="language-switcher" aria-labelledby="assistant-language-label">
      <div className="language-switcher-label" id="assistant-language-label"><Languages /> Assistant language</div>
      <div className="language-switcher-options" role="group" aria-label="Choose assistant language">
        {languages.map(({ code, label, native }) => (
          <button type="button" key={code} className={language === code ? "active" : ""}
            aria-pressed={language === code} aria-label={`Use ${label}`} disabled={busy}
            onClick={() => { setLanguage(code); void warmLanguageModel(code); }}>
            <span>{native}</span><small>{label}</small>
          </button>
        ))}
      </div>
      <p>Speak in the selected language. The assistant will answer in the same language.</p>
    </section>
  );
}
