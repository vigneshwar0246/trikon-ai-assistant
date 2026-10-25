import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

/**
 * IntroPage — cinematic landing screen for the AI Voice Assistant.
 *
 * The existing animation fills the viewport; the enter action sits over its
 * centered artwork while the header and supporting copy remain over the video.
 */
export function IntroPage() {
  const navigate = useNavigate();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoFailed, setVideoFailed] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  const [exiting, setExiting] = useState(false);
  const prefersReducedMotion =
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ── autoplay ── */
  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    try {
      const playPromise = el.play();
      if (playPromise !== undefined && typeof playPromise.catch === "function") {
        playPromise.catch(() => {
          /* autoplay blocked — video stays paused, UI still works */
        });
      }
    } catch {
      /* media playback invocation error in non-supporting runtime */
    }
  }, []);

  /* ── enter handler ── */
  const handleEnter = () => {
    if (exiting) return;
    sessionStorage.setItem("intro_seen", "1");
    if (prefersReducedMotion) {
      navigate("/", { replace: true });
      return;
    }
    setExiting(true);
    /* fade-out duration matches CSS transition (600 ms) */
    setTimeout(() => {
      navigate("/", { replace: true });
    }, 600);
  };

  /* ── keyboard: Enter / Space on the wrapper ── */
  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      handleEnter();
    }
  };

  return (
    <div
      className={`intro-shell${exiting ? " intro-exit" : ""}`}
      onKeyDown={handleKey}
    >
      {/* ── 1. Header ── */}
      <header className="intro-header" aria-label="Intro header">
        <p className="intro-overline">AI VOICE ASSISTANT</p>
        <h1 className="intro-headline">Meet Your AI Voice Assistant.</h1>
      </header>

      {/* ── Full-screen video stage ── */}
      <section className="intro-video-stage" aria-label="Intro animation">
        {!videoReady && <div className="intro-video-fallback" aria-hidden="true" />}
        {!videoFailed ? (
          <video
            ref={videoRef}
            className={`intro-video${videoReady ? " intro-video-ready" : ""}`}
            src="/videos/intro-sphere.mp4"
            /* transparency: keep alpha channel if the codec supports it */
            autoPlay
            muted
            loop
            playsInline
            preload="auto"
            aria-hidden="true"
            onCanPlay={() => setVideoReady(true)}
            onError={() => setVideoFailed(true)}
          />
        ) : null}

        <button
          className="intro-enter-btn"
          onClick={handleEnter}
          aria-label="Enter the AI Voice Assistant"
          type="button"
        >
          Enter Assistant
        </button>
      </section>

      {/* ── Supporting copy ── */}
      <p className="intro-tagline">
        A more natural way to interact with AI.
      </p>
    </div>
  );
}

