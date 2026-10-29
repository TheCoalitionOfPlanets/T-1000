import { useEffect, useRef, useState } from "react"
import { ArrowDownToLine, ArrowRight, AudioLines, Check, CircleHelp, Cpu, Headphones, LoaderCircle, Mic, Plus, Radio, RotateCcw, ShieldCheck, SlidersHorizontal, Sparkles, Square, Upload, X } from "lucide-react"
import { AudioPlayerButton, AudioPlayerDuration, AudioPlayerProgress, AudioPlayerProvider, AudioPlayerSpeed, AudioPlayerTime, useAudioPlayer } from "@/components/elevenlabs/audio-player"
import { LiveWaveform } from "@/components/elevenlabs/live-waveform"
import { Button } from "@/components/ui/button"
import { downloadVerifiedAudio, generate, post, request, type Config, type Reference, type GenerationUpdate } from "@/lib/api"
import { VoiceRecorder } from "@/lib/recorder"

const phrases = [
  { name: "A warm welcome", text: "Welcome. Make yourself comfortable. I have a feeling we're going to create something wonderful." },
  { name: "Tell a story", text: "The city was still asleep when she found the letter. No name, no address. Just three words: it's time." },
  { name: "A little motivation", text: "Start small. Stay curious. The thing you've been imagining is closer than you think." },
]

function Player({ url, compact = false }: { url: string; compact?: boolean }) {
  const player = useAudioPlayer()
  useEffect(() => { void player.setActiveItem({ id: url, src: url }) }, [url, player.setActiveItem])
  return <>
    <div className={`player-controls ${compact ? "compact" : ""}`}>
      <AudioPlayerButton className="play-button" variant="default" size="icon" />
      <AudioPlayerTime />
      <AudioPlayerProgress aria-label="Audio playback position" className="player-progress" />
      <AudioPlayerDuration />
      {!compact && <AudioPlayerSpeed aria-label="Playback speed" variant="ghost" />}
    </div>
    {player.error && <p className="field-error">Audio could not be played. Try downloading the WAV file.</p>}
  </>
}

function AudioWaveform({ url }: { url: string }) {
  const [peaks, setPeaks] = useState<number[]>([])
  useEffect(() => {
    const controller = new AbortController()
    let context: AudioContext | undefined
    let disposed = false
    setPeaks([])
    void (async () => {
      const response = await fetch(url, { signal: controller.signal })
      if (!response.ok) return
      const bytes = await response.arrayBuffer()
      if (disposed) return
      context = new AudioContext()
      const audio = await context.decodeAudioData(bytes)
      const data = audio.getChannelData(0)
      const values = Array.from({ length: 96 }, (_, bin) => {
        const start = Math.floor(bin * data.length / 96)
        const end = Math.floor((bin + 1) * data.length / 96)
        let peak = 0
        for (let i = start; i < end; i++) peak = Math.max(peak, Math.abs(data[i]))
        return peak
      })
      const maximum = Math.max(...values, 0.01)
      if (!disposed) setPeaks(values.map(peak => peak / maximum))
    })().catch(() => {}).finally(() => { void context?.close() })
    return () => { disposed = true; controller.abort(); void context?.close().catch(() => {}) }
  }, [url])
  return <svg className="audio-waveform" viewBox="0 0 960 88" preserveAspectRatio="none" aria-hidden="true">
    {peaks.map((peak, index) => <line key={index} x1={index * 10 + 5} x2={index * 10 + 5} y1={44 - Math.max(2, peak * 36)} y2={44 + Math.max(2, peak * 36)} stroke="currentColor" strokeWidth="3" strokeLinecap="round" />)}
  </svg>
}

function Help({ close }: { close: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => { dialog.current?.showModal() }, [])
  return <dialog className="help-dialog" ref={dialog} onClose={close} onClick={event => { if (event.target === dialog.current) dialog.current.close() }}>
    <div className="dialog-heading"><h2>A few seconds. Your voice.</h2><Button variant="ghost" size="icon" aria-label="Close help" onClick={() => dialog.current?.close()}><X size={18} /></Button></div>
    <ol>
      <li><strong>Choose a voice.</strong> Start with the included example, upload a clip, or record one clear English sentence.</li>
      <li><strong>Check the transcript.</strong> The reference words should match the recording exactly.</li>
      <li><strong>Make it speak.</strong> Write a short line and generate. The first request also loads the local model.</li>
    </ol>
    <p>CPU generation can take several minutes. Stop generation finishes the current audio step and saves any partial audio.</p>
    <p>Audio and inference stay on this machine. No ElevenLabs account or API key is needed.</p>
    <a href="https://github.com/elevenlabs/ui" target="_blank" rel="noreferrer">Built with ElevenLabs UI <ArrowRight size={14} /></a>
  </dialog>
}

export default function App() {
  const [config, setConfig] = useState<Config | null>(null)
  const [connectionError, setConnectionError] = useState("")
  const [text, setText] = useState("Hello, this is a voice test.")
  const [normalizing, setNormalizing] = useState(false)
  const [textPreview, setTextPreview] = useState<{ original: string; expanded: string } | null>(null)
  const [normalizationError, setNormalizationError] = useState("")
  const [originalText, setOriginalText] = useState<string | null>(null)
  const [reference, setReference] = useState<Reference | null>(null)
  const [referenceName, setReferenceName] = useState("Example voice")
  const [transcript, setTranscript] = useState("")
  const [referenceStatus, setReferenceStatus] = useState("Choose an example, record your voice, or upload a clip.")
  const [referenceError, setReferenceError] = useState("")
  const [referenceBusy, setReferenceBusy] = useState(false)
  const [recording, setRecording] = useState(false)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [recordedSeconds, setRecordedSeconds] = useState(0)
  const [preview, setPreview] = useState("")
  const [recoveredRecording, setRecoveredRecording] = useState<File | null>(null)
  const [recoveryUrl, setRecoveryUrl] = useState<string | null>(null)
  const recoveryFile = useRef<File | null>(null)
  const [canResume, setCanResume] = useState(false)
  const pendingGeneration = useRef<{ text: string; reference_id: string; transcript: string; steps: number; seed: number; generation_id?: string } | null>(null)
  const [steps, setSteps] = useState(10)
  const [seed, setSeed] = useState("42")
  const [busy, setBusy] = useState(false)
  const [stopping, setStopping] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [resultLabel, setResultLabel] = useState("Included preview")
  const [status, setStatus] = useState("Ready when you are.")
  const [generationError, setGenerationError] = useState("")
  const [help, setHelp] = useState(false)
  const recorder = useRef<VoiceRecorder | null>(null)
  const generationController = useRef<AbortController | null>(null)
  const referenceController = useRef<AbortController | null>(null)
  const normalizationController = useRef<AbortController | null>(null)
  const normalizationVersion = useRef(0)
  const fileInput = useRef<HTMLInputElement>(null)
  const textInput = useRef<HTMLTextAreaElement>(null)

  async function connect() {
    setConnectionError("")
    try {
      const data = await request<Config>("/api/config")
      setConfig(data)
      if (data.example) { setReference(data.example); setTranscript(data.example.transcript); setReferenceStatus("Example voice ready. Make it your own, or start creating.") }
      setResult(data.sample_url)
    } catch { setConnectionError("Could not connect to the local server. Start the local API and try again.") }
  }

  useEffect(() => {
    void connect()
    const cancel = () => {
      void recorder.current?.cancel()
      const generationId = pendingGeneration.current?.generation_id
      if (generationId) void fetch("/api/stop", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ generation_id: generationId }), keepalive: true }).catch(() => {})
      generationController.current?.abort(); referenceController.current?.abort(); normalizationController.current?.abort()
    }
    window.addEventListener("pagehide", cancel)
    return () => { cancel(); window.removeEventListener("pagehide", cancel) }
  }, [])

  useEffect(() => {
    if (!recoveredRecording) { setRecoveryUrl(null); return }
    const url = URL.createObjectURL(recoveredRecording)
    setRecoveryUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [recoveredRecording])
  useEffect(() => () => { if (result?.startsWith("blob:")) URL.revokeObjectURL(result) }, [result])

  const referenceLocked = busy || referenceBusy || recording || canResume
  const validSeed = /^\d+$/.test(seed) && Number(seed) <= 4294967295
  const canGenerate = !!config && !!reference && !!text.trim() && !!transcript.trim() && text.length <= config.text_limit && validSeed && !referenceLocked && !normalizing && !textPreview

  function cancelExpansion() {
    normalizationVersion.current++
    normalizationController.current?.abort()
    normalizationController.current = null
    setNormalizing(false)
    setTextPreview(null)
    setNormalizationError("")
  }

  function changeText(next: string) {
    cancelExpansion()
    setOriginalText(null)
    setText(next)
  }

  async function expandNumbers() {
    if (!config || busy || normalizing || !text.trim() || text.length > config.text_limit) return
    cancelExpansion()
    const version = normalizationVersion.current
    const original = text
    const controller = new AbortController()
    normalizationController.current = controller
    setNormalizing(true)
    try {
      const result = await post<{ text: string }>("/api/text/normalize", { text: original }, controller.signal)
      if (version === normalizationVersion.current) setTextPreview({ original, expanded: result.text })
    } catch (error) {
      if (version === normalizationVersion.current && !controller.signal.aborted) setNormalizationError(error instanceof Error ? error.message : "Text expansion is unavailable. Your original text is still here.")
    } finally {
      if (version === normalizationVersion.current) { setNormalizing(false); normalizationController.current = null }
    }
  }

  function applyExpansion() {
    if (!textPreview || text !== textPreview.original) return
    const selected = textPreview
    changeText(selected.expanded)
    setOriginalText(selected.original)
    textInput.current?.focus()
  }

  function useExample() {
    recoveryFile.current = null; setRecoveredRecording(null)
    if (!config?.example) return
    setReference(config.example)
    setReferenceName("Example voice")
    setTranscript(config.example.transcript)
    setReferenceError("")
    setReferenceStatus("Example voice ready. Make it your own, or start creating.")
  }

  async function upload(file?: File) {
    if (!file || referenceLocked) return
    if (file !== recoveryFile.current) { recoveryFile.current = null; setRecoveredRecording(null) }
    setReference(null)
    setTranscript("")
    setReferenceError("")
    if (file.size > 32 * 1024 * 1024) { setReferenceError("Choose an audio file smaller than 32 MB."); return }
    setReferenceBusy(true)
    setReferenceStatus("Uploading your reference and transcribing its words…")
    referenceController.current = new AbortController()
    const data = new FormData()
    data.append("file", file)
    try {
      const selected = await request<Reference>("/api/reference", { method: "POST", body: data, signal: referenceController.current.signal })
      setReference(selected)
      setReferenceName(file.name)
      setTranscript(selected.transcript)
      setReferenceStatus(selected.status)
      recoveryFile.current = null; setRecoveredRecording(null)
    } catch (error) { setReferenceError(error instanceof Error ? error.message : "This audio could not be uploaded."); setReferenceStatus("Upload another recording or use the example voice.") }
    finally { setReferenceBusy(false); if (fileInput.current) fileInput.current.value = "" }
  }

  function record() {
    recoveryFile.current = null; setRecoveredRecording(null)
    setReference(null)
    setTranscript("")
    setPreview("")
    setReferenceError("")
    setRecordedSeconds(0)
    setRecording(true)
    recorder.current = new VoiceRecorder({
      onStream: setStream,
      onFinishing: () => setReferenceBusy(true),
      onProgress: (seconds, words) => { setRecordedSeconds(seconds); setPreview(words) },
      onStatus: setReferenceStatus,
      onComplete: selected => { setReference(selected); setReferenceName("My recorded voice"); setTranscript(selected.transcript); setReferenceStatus(selected.status); setRecording(false); setReferenceBusy(false); recorder.current = null },
      onRecovery: (file, message) => { recoveryFile.current = file; setRecoveredRecording(file); setReferenceStatus(message) },
      onError: message => { setReferenceError(message); setReferenceStatus(recoveryFile.current ? "Your complete recording is held in this tab. Download it before closing, or retry the upload." : "Record again, upload audio, or use the example voice."); setRecording(false); setReferenceBusy(false); recorder.current = null },
    })
    void recorder.current.start()
  }

  async function finishRecording() {
    setReferenceBusy(true)
    await recorder.current?.finish()
  }

  async function cancelRecording() {
    await recorder.current?.cancel()
    recorder.current = null
    setRecording(false)
    setReferenceBusy(false)
    setReferenceStatus("Recording discarded. Record again or choose the example voice.")
  }

  async function runGeneration(payload: NonNullable<typeof pendingGeneration.current>) {
    if (generationController.current) return
    setBusy(true)
    setStopping(false)
    setCanResume(false)
    setGenerationError("")
    setStatus("Preparing your reference recording…")
    const controller = new AbortController()
    generationController.current = controller
    let terminal: GenerationUpdate | undefined
    try {
      await generate(payload, controller.signal, update => {
        setStatus(update.status)
        if (update.done) terminal = update
        if (update.error) setGenerationError(update.error)
      })
      if (terminal?.audio_url) {
        let url = terminal.audio_url
        if (terminal.audio) {
          setStatus("Checking the received audio…")
          const blob = await downloadVerifiedAudio(url, terminal.audio, controller.signal)
          if (controller.signal.aborted) return
          url = URL.createObjectURL(blob)
        }
        setResult(url)
        setResultLabel(terminal.partial ? "Your partial speech" : "Your generated speech")
        setStatus(terminal.status)
      } else if (terminal) {
        setGenerationError(terminal.error || terminal.status)
      }
      pendingGeneration.current = null
      setCanResume(false)
    } catch (error) {
      if (!controller.signal.aborted) {
        setGenerationError(error instanceof Error ? error.message : "Generation connection was interrupted.")
        setStatus(payload.generation_id ? "Reconnect to the same generation to recover its audio." : "Generation ended. Review the message below and try again.")
        setCanResume(!!payload.generation_id)
      }
    } finally {
      setBusy(false); setStopping(false); generationController.current = null
    }
  }

  async function synthesize() {
    if (!canGenerate || !reference || generationController.current) return
    const payload = { text, reference_id: reference.id, transcript, steps, seed: Number(seed),
      ...(config?.resumable_generation ? { generation_id: crypto.randomUUID() } : {}) }
    pendingGeneration.current = payload
    await runGeneration(payload)
  }

  function resetGeneration() {
    const generationId = pendingGeneration.current?.generation_id
    pendingGeneration.current = null
    setCanResume(false); setStopping(false); setGenerationError("")
    setStatus("Your text and reference are ready. Generate again when the connection is available.")
    if (generationId) {
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), 4000)
      void post("/api/stop", { generation_id: generationId }, controller.signal).catch(() => {}).finally(() => clearTimeout(timer))
    }
  }

  async function stop() {
    setStopping(true)
    try {
      const generationId = pendingGeneration.current?.generation_id
      await post("/api/stop", generationId ? { generation_id: generationId } : {})
      setStatus("Stop requested. Finishing the current audio step…")
    } catch (error) { setStopping(false); setGenerationError(error instanceof Error ? error.message : "Could not stop generation. Try again.") }
  }

  return <div className="studio-shell">
    <aside className="sidebar">
      <a className="brand" href="#studio" aria-label="T-1000 home"><span className="brand-mark"><AudioLines size={22} /></span><span>T–1000<span className="brand-subtitle">VOICE STUDIO</span></span></a>
      <div className="workspace-label">YOUR WORKSPACE</div>
      <nav aria-label="Workspace"><a href="#studio" className="nav-item active"><AudioLines size={18} />Text to speech<span className="nav-dot" /></a><a href="#voice" className="nav-item"><Mic size={18} />Reference voice</a><a href="#settings" className="nav-item"><SlidersHorizontal size={18} />Voice settings</a></nav>
      <div className="sidebar-note"><div className="mini-wave"><span /><span /><span /><span /><span /><span /><span /></div><p>A little text.<br />A lot of possibility.</p><span>Give your ideas a voice.</span></div>
      <div className="sidebar-bottom"><div className="local-label"><span className="status-dot" />Runs on your machine</div><button className="help-button" onClick={() => setHelp(true)}><CircleHelp size={17} />Getting started<ArrowRight size={15} /></button><div className="sidebar-credit">Crafted with <a href="https://github.com/elevenlabs/ui" target="_blank" rel="noreferrer">ElevenLabs UI ↗</a></div></div>
    </aside>

    <main id="studio">
      <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> <strong>Text to speech</strong></div><div className="topbar-right"><span className="private-label"><ShieldCheck size={14} />Private by design</span><span className="avatar">T</span></div></header>
      <div className="main-content">
        <div className="page-heading"><div><div className="eyebrow"><span />THE LOCAL VOICE LAB</div><h1>Good words. <span>Great voice.</span></h1><p>Turn a thought into speech. Make it sound like you.</p></div><div className="model-badge"><span className="model-icon"><AudioLines size={19} /></span><div><strong>dots.tts SOAR</strong><span>{config ? config.device : "Connecting…"} <span>·</span> English</span></div></div></div>

        {connectionError && <div className="connection-error" role="alert"><span>{connectionError}</span><Button variant="outline" size="sm" onClick={() => void connect()}>Retry connection</Button></div>}

        <div className="studio-grid">
          <div className="creation-column">
            <section className="panel editor-panel" aria-labelledby="text-heading">
              <div className="panel-heading"><h2 id="text-heading"><span className="step-number">01</span>What would you like to say?</h2><span className="small-label">English</span></div>
              <div className="text-editor"><label className="sr-only" htmlFor="speech-text">Text to speak</label><textarea id="speech-text" ref={textInput} value={text} disabled={busy} placeholder="Every great voice starts with a few words…" onChange={event => changeText(event.target.value)} onKeyDown={event => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { event.preventDefault(); void synthesize() } }} /><div className="editor-meta"><span><span className="subtle-dot" />Let your words do the talking.</span><span className={config && text.length > config.text_limit ? "over-limit" : ""}>{text.length}<span> / {config?.text_limit ?? "—"}</span></span></div></div>
              <div className="text-expansion">
                <div className="text-tools"><Button variant="ghost" size="sm" disabled={!config || busy || normalizing || !text.trim() || text.length > config.text_limit} onClick={() => void expandNumbers()}>{normalizing ? <><LoaderCircle size={13} className="spin" />Expanding…</> : "Expand numbers"}</Button>{originalText !== null && <Button variant="ghost" size="sm" disabled={busy || normalizing} onClick={() => { changeText(originalText); textInput.current?.focus() }}>Undo expansion</Button>}{normalizing && <Button variant="ghost" size="sm" onClick={cancelExpansion}>Cancel</Button>}</div>
                {textPreview && <section className="spoken-text-preview" aria-labelledby="spoken-text-heading"><h3 id="spoken-text-heading">Spoken text preview</h3><p className="spoken-text-content">{textPreview.expanded}</p><p className="spoken-text-note">Review numbers, dates, phone numbers and abbreviations before applying. Edit the text if you want a different reading.</p><div className="text-tools"><Button variant="outline" size="sm" onClick={applyExpansion}>Apply</Button><Button variant="ghost" size="sm" onClick={cancelExpansion}>Cancel</Button></div></section>}
                {normalizationError && <p className="field-error" role="alert">{normalizationError}</p>}
              </div>
              <div className="inspiration"><span>NEED A FIRST LINE?</span><div>{phrases.map(phrase => <button key={phrase.name} disabled={busy} onClick={() => { changeText(phrase.text); textInput.current?.focus() }}><Plus size={12} />{phrase.name}</button>)}</div></div>
              <div className="generate-bar"><div className="generation-hint"><Sparkles size={16} /><span>Your words, a new dimension.</span></div>{busy ? <Button className="stop-button" disabled={stopping} onClick={() => void stop()}><Square size={14} fill="currentColor" />{stopping ? "Stopping…" : "Stop generation"}</Button> : <Button className="generate-button" disabled={!canGenerate} onClick={() => void synthesize()}><AudioLines size={17} />Generate speech<ArrowRight size={16} /></Button>}</div>
            </section>

            <section className="panel output-panel" aria-labelledby="output-heading" aria-busy={busy}>
              <div className="panel-heading"><h2 id="output-heading"><Headphones size={17} />The listening room</h2>{result && <a className="download-link" href={result} download="t-1000-speech.wav" aria-label="Download speech as WAV"><ArrowDownToLine size={15} />Download WAV</a>}</div>
              <div className="output-body">{busy && <div className="generating-state"><LoaderCircle size={19} className="spin" /><div><strong>{stopping ? "Finishing the current step" : "Bringing your words to life"}</strong><span>{config?.device === "CPU" ? "CPU synthesis can take a few minutes. You can stop at any time." : "Your local model is creating your audio."}</span></div></div>}
                {result ? <AudioPlayerProvider key={result}><div className="audio-result"><div className="audio-result-label"><span><span className="status-dot" />{resultLabel}</span><span>WAV · 48 kHz</span></div><AudioWaveform url={result} /><Player url={result} /></div></AudioPlayerProvider> : !busy && <div className="output-empty"><div className="empty-audio-lines"><AudioLines size={30} strokeWidth={1.3} /></div><strong>Something worth listening to.</strong><p>Your generated audio will be right here.</p></div>}
                {canResume && <div className="text-tools"><Button variant="outline" size="sm" onClick={() => { if (pendingGeneration.current) void runGeneration(pendingGeneration.current) }}>Reconnect to generation</Button><Button variant="ghost" size="sm" disabled={stopping} onClick={() => void stop()}>{stopping ? "Stop requested" : "Stop generation"}</Button><Button variant="ghost" size="sm" onClick={() => void resetGeneration()}>Stop and reset</Button></div>}
                <div className={`result-status ${generationError ? "has-error" : ""}`} role="status" aria-live="polite">{busy ? <span className="activity-dot" /> : <Check size={13} />}<span>{status}</span></div>
                {generationError && <p className="field-error" role="alert">{generationError}</p>}
              </div>
            </section>
            <div className="creation-footer"><ShieldCheck size={14} /><span>Made here. Stays here. Your voice is processed locally.</span><kbd>⌘ / Ctrl + Enter</kbd></div>
          </div>

          <div className="voice-column">
            <section className="panel reference-panel" id="voice" aria-labelledby="voice-heading">
              <div className="panel-heading"><h2 id="voice-heading"><span className="step-number">02</span>Make it your voice</h2><Mic size={16} className="muted-icon" /></div>
              <div className="reference-body"><p className="section-description">A short reference is all it takes. Record a clear sentence or bring an audio clip.</p>
                <div className={`selected-voice ${reference ? "is-ready" : ""}`}><span className="voice-avatar"><AudioLines size={22} /></span><div><strong>{reference ? referenceName : recording ? "Your microphone" : "Choose a reference"}</strong><span>{reference ? "English · Reference voice" : `Up to ${config?.reference_limit ?? "—"} seconds`}</span></div>{reference ? <span className="selected-check"><Check size={13} /></span> : referenceBusy ? <LoaderCircle size={17} className="spin" /> : <Mic size={16} />}</div>
                {reference && <AudioPlayerProvider key={reference.id}><Player url={reference.url} compact /></AudioPlayerProvider>}
                {recording && <div className="recording-view"><div className="recording-indicator"><span className="activity-dot" />{stream ? "Recording" : referenceBusy ? "Finishing" : "Connecting microphone"}<span>{recordedSeconds.toFixed(1)}s / {config?.reference_limit}s</span></div><LiveWaveform stream={stream} active={!!stream} processing={referenceBusy} height={56} barColor="#171717" barWidth={3} barGap={3} /><p className="live-preview">{preview || "Your words will appear here as you speak…"}</p></div>}
                <div className="reference-actions">{recording ? <><Button variant="outline" className="finish-recording" disabled={!stream || referenceBusy} onClick={() => void finishRecording()}><Square size={13} fill="currentColor" />Finish recording</Button><Button variant="ghost" size="icon" aria-label="Discard recording" onClick={() => void cancelRecording()}><X size={16} /></Button></> : <><Button variant="outline" disabled={!config || referenceLocked} onClick={record}><Mic size={15} />Record voice</Button><Button variant="outline" disabled={!config || referenceLocked} onClick={() => fileInput.current?.click()}><Upload size={15} />Upload audio</Button></>}</div>
                <input ref={fileInput} type="file" accept="audio/*,.wav,.flac,.mp3,.ogg,.m4a" className="sr-only" aria-label="Upload reference audio" onChange={event => void upload(event.target.files?.[0])} disabled={referenceLocked} />
                <div className="reference-note" role="status" aria-live="polite">{referenceBusy && <LoaderCircle size={12} className="spin" />}<span>{referenceStatus}</span></div>
                {referenceError && <p className="field-error" role="alert">{referenceError}</p>}
                {recoveryUrl && recoveredRecording && <div className="text-tools"><a className="download-link" href={recoveryUrl} download="my-voice-recording.wav">Download recording</a><Button variant="outline" size="sm" disabled={referenceBusy} onClick={() => void upload(recoveredRecording)}>Retry recording upload</Button></div>}
                <div className="transcript-heading"><label htmlFor="reference-transcript">Reference transcript</label><span>{recording ? "LIVE" : "EDITABLE"}</span></div><textarea id="reference-transcript" className="transcript-input" rows={3} value={transcript} disabled={referenceLocked || !reference} placeholder="The exact words spoken in your reference…" onChange={event => setTranscript(event.target.value)} /><p className="transcript-help">For the best match, these words should follow the recording exactly.</p>
                <button className="restore-example" disabled={!config?.example || referenceLocked} onClick={useExample}><RotateCcw size={12} />Use example voice</button>
              </div>
            </section>

            <section className="panel settings-panel" id="settings" aria-labelledby="settings-heading"><div className="panel-heading"><h2 id="settings-heading"><SlidersHorizontal size={16} />Fine-tune the delivery</h2></div><div className="settings-body"><div className="range-label"><label htmlFor="steps">Sampling steps</label><output htmlFor="steps">{steps}</output></div><input id="steps" type="range" min="4" max="32" step="1" value={steps} disabled={busy} onChange={event => setSteps(Number(event.target.value))} /><div className="range-captions"><span>Faster</span><span>More detail</span></div><div className="seed-field"><div><label htmlFor="seed">Generation seed</label><span>Keep it for a repeatable take.</span></div><input id="seed" type="number" min="0" max="4294967295" step="1" value={seed} disabled={busy} onChange={event => setSeed(event.target.value)} /></div>{!validSeed && <p className="field-error">Use a whole number from 0 to 4294967295.</p>}</div></section>
            <div className="device-note"><Cpu size={16} /><p><strong>{config?.device === "CPU" ? "Good things take a moment." : "Your machine. Your creativity."}</strong><span>{config?.device === "CPU" ? "Start with a short sentence. Your CPU handles everything locally." : "Audio is generated on your local hardware."}</span></p></div>
          </div>
        </div>
        <footer className="page-footer"><span>T–1000 <span> / </span> ENGLISH VOICE LAB</span><span><Radio size={12} />Local inference. Limitless ideas.</span><button onClick={() => setHelp(true)}>Quick guide<ArrowRight size={12} /></button></footer>
      </div>
    </main>
    {help && <Help close={() => setHelp(false)} />}
  </div>
}
