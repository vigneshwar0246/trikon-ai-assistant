import { useCallback, useEffect, useRef, useState } from "react";
import { analyzeAudioFrame, commandCaptureDecision } from "../lib/audioDetectors";

export type HandsFreeState = "off" | "permission" | "armed" | "capturing" | "processing" | "muted" | "error";

interface HandsFreeOptions {
  silenceMs: number;
  suspended: boolean;
  onCommand: (audio: Blob, commandEndedAt: number, captureMs: number, vadEndMs: number) => Promise<void>;
  onError: (error: Error) => void;
  onStatusChange: (status: HandsFreeState) => void;
  monitorInterruption?: boolean;
  onInterruption?: () => void;
  speechThreshold?: number;
  minimumSpeechMs?: number;
  maxDurationMs?: number;
  interruptionThreshold?: number;
  interruptionDurationMs?: number;
  interruptionWarmupMs?: number;
}

export function useHandsFree(options: HandsFreeOptions) {
  const [status, setStatus] = useState<HandsFreeState>("off");
  const [muted, setMuted] = useState(false);
  const statusRef = useRef<HandsFreeState>("off");
  const mutedRef = useRef(false);
  const optionsRef = useRef(options);
  const streamRef = useRef<MediaStream | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const samplesRef = useRef<Float32Array<ArrayBuffer> | null>(null);
  const timerRef = useRef<number | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const startedAtRef = useRef(0);
  const lastSpeechRef = useRef<number | null>(null);
  const speechFramesRef = useRef(0);
  const noiseFloorRef = useRef(0.003);
  const discardRef = useRef(false);
  const captureStoppedAtRef = useRef(0);
  const generationRef = useRef(0);
  const startCaptureRef = useRef<() => void>(() => {});
  const speechStartedAtRef = useRef<number | null>(null);
  const interruptionFramesRef = useRef(0);
  const interruptionStartedAtRef = useRef<number | null>(null);

  useEffect(() => {
    optionsRef.current = options;
    if (!options.monitorInterruption) {
      interruptionStartedAtRef.current = null;
      interruptionFramesRef.current = 0;
    }
  }, [options]);

  const changeStatus = useCallback((next: HandsFreeState) => {
    statusRef.current = next;
    setStatus(next);
    optionsRef.current.onStatusChange(next);
  }, []);

  const stop = useCallback(() => {
    generationRef.current += 1;
    if (timerRef.current !== null) window.clearInterval(timerRef.current);
    timerRef.current = null;
    discardRef.current = true;
    interruptionFramesRef.current = 0;
    interruptionStartedAtRef.current = null;
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
    recorderRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    void contextRef.current?.close().catch(() => undefined);
    contextRef.current = null;
    analyserRef.current = null;
    samplesRef.current = null;
    changeStatus("off");
  }, [changeStatus]);

  const finishCapture = useCallback((discard: boolean) => {
    discardRef.current = discard;
    captureStoppedAtRef.current = performance.now();
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
  }, []);

  const startCapture = useCallback(() => {
    const stream = streamRef.current;
    if (!stream || statusRef.current !== "armed") return;
    try {
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      const captureGeneration = generationRef.current;
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
      recorder.onstop = () => {
        if (captureGeneration !== generationRef.current) return;
        if (discardRef.current || !chunks.length) {
          if (mutedRef.current) {
            changeStatus("muted");
          } else {
            changeStatus("armed");
            startCaptureRef.current();
          }
          return;
        }
        const generation = generationRef.current;
        changeStatus("processing");
        const audio = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
        const commandEndedAt = lastSpeechRef.current ?? captureStoppedAtRef.current;
        void optionsRef.current.onCommand(audio, commandEndedAt, captureStoppedAtRef.current - startedAtRef.current, captureStoppedAtRef.current - commandEndedAt).catch((error: unknown) => {
          if (!(error instanceof Error && error.name === "AbortError")) {
            optionsRef.current.onError(error instanceof Error ? error : new Error("Voice request failed."));
          }
        }).finally(() => {
          if (generation === generationRef.current) {
            if (mutedRef.current) {
              changeStatus("muted");
            } else {
              changeStatus("armed");
              startCaptureRef.current();
            }
          }
        });
      };
      recorder.start();
      startedAtRef.current = performance.now();
      lastSpeechRef.current = null;
      speechFramesRef.current = 0;
      speechStartedAtRef.current = null;
      discardRef.current = false;
      changeStatus("capturing");
    } catch (error) {
      optionsRef.current.onError(error instanceof Error ? error : new Error("Microphone recording failed."));
      changeStatus("armed");
    }
  }, [changeStatus]);

  const tick = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;
    const samples: Float32Array<ArrayBuffer> = samplesRef.current ??= new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(samples);
    const frame = analyzeAudioFrame(samples);
    const now = performance.now();
    const threshold = Math.max(optionsRef.current.speechThreshold ?? 0.009, noiseFloorRef.current * 2.4);
    if (statusRef.current === "processing" && optionsRef.current.monitorInterruption) {
      interruptionStartedAtRef.current ??= now;
      const warmupMs = optionsRef.current.interruptionWarmupMs ?? 0;
      if (warmupMs > 0 && now - interruptionStartedAtRef.current < warmupMs) {
        interruptionFramesRef.current = 0;
        return;
      }
      const interruptionThreshold = optionsRef.current.interruptionThreshold ?? threshold;
      const requiredFrames = optionsRef.current.interruptionDurationMs
        ? Math.max(1, Math.round(optionsRef.current.interruptionDurationMs / 10))
        : 5;
      interruptionFramesRef.current = frame.rms > interruptionThreshold ? interruptionFramesRef.current + 1 : 0;
      if (interruptionFramesRef.current >= requiredFrames) {
        interruptionFramesRef.current = 0;
        interruptionStartedAtRef.current = null;
        generationRef.current += 1;
        optionsRef.current.onInterruption?.();
        changeStatus("armed");
        startCaptureRef.current();
      }
      return;
    } else {
      interruptionStartedAtRef.current = null;
      interruptionFramesRef.current = 0;
    }
    if (optionsRef.current.suspended) return;
    if (statusRef.current === "capturing") {
      const speech = frame.rms > threshold;
      speechFramesRef.current = speech ? speechFramesRef.current + 1 : 0;
      if (speechFramesRef.current >= 2) {
        speechStartedAtRef.current ??= now;
        lastSpeechRef.current = now;
      }
      if (!speech) noiseFloorRef.current = Math.min(0.03, noiseFloorRef.current * 0.98 + frame.rms * 0.02);
      const decision = commandCaptureDecision(now, startedAtRef.current, lastSpeechRef.current, optionsRef.current.silenceMs, optionsRef.current.maxDurationMs);
      if (decision !== "continue") {
        const tooShort = speechStartedAtRef.current !== null && lastSpeechRef.current !== null
          && lastSpeechRef.current - speechStartedAtRef.current < (optionsRef.current.minimumSpeechMs ?? 180);
        finishCapture(decision === "discard" || tooShort);
      }
    }
  }, [changeStatus, finishCapture]);
  useEffect(() => { startCaptureRef.current = startCapture; }, [startCapture]);

  const start = useCallback(async () => {
    if (statusRef.current !== "off" && statusRef.current !== "error") return;
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder || !window.AudioContext) {
      optionsRef.current.onError(new Error("This browser does not support local microphone capture."));
      changeStatus("error");
      return;
    }
    const generation = generationRef.current;
    changeStatus("permission");
    try {
      // Create/resume AudioContext inside the click gesture, before the permission prompt
      // can consume browser user activation.
      const context = new AudioContext();
      contextRef.current = context;
      const audioReady = context.resume().then(() => null, (error: unknown) => error);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
      if (generation !== generationRef.current) { stream.getTracks().forEach((track) => track.stop()); return; }
      streamRef.current = stream;
      const resumeError = await audioReady;
      if (resumeError !== null) throw resumeError;
      if (generation !== generationRef.current) return;
      const analyser = context.createAnalyser();
      // Small overlapping frames keep speech endpoint detection responsive.
      analyser.fftSize = 512;
      context.createMediaStreamSource(stream).connect(analyser);
      // Keep the analyser in the active audio graph without monitoring the mic
      // through the speakers (which would create feedback and false activations).
      const silentOutput = context.createGain();
      silentOutput.gain.value = 0;
      analyser.connect(silentOutput);
      silentOutput.connect(context.destination);
      stream.getAudioTracks().forEach((track) => { track.onended = () => { optionsRef.current.onError(new Error("Microphone disconnected.")); stop(); }; });
      analyserRef.current = analyser;
      noiseFloorRef.current = 0.003;
      samplesRef.current = new Float32Array(analyser.fftSize);
      timerRef.current = window.setInterval(tick, 10);
      if (mutedRef.current) {
        changeStatus("muted");
      } else {
        changeStatus("armed");
        startCapture();
      }
    } catch (error) {
      if (generation !== generationRef.current) return;
      stop();
      optionsRef.current.onError(error instanceof Error ? error : new Error("Microphone permission failed."));
      changeStatus("error");
    }
  }, [changeStatus, startCapture, stop, tick]);

  const toggleMute = useCallback(() => {
    const next = !mutedRef.current;
    mutedRef.current = next;
    setMuted(next);
    if (next && statusRef.current === "capturing") finishCapture(true);
    if (statusRef.current === "armed" || statusRef.current === "muted") {
      if (next) {
        changeStatus("muted");
      } else {
        changeStatus("armed");
        startCapture();
      }
    }
  }, [changeStatus, finishCapture, startCapture]);
  useEffect(() => () => stop(), [stop]);

  return { status, muted, start, stop, toggleMute };
}
