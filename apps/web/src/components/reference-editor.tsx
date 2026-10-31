import { useEffect, useRef, useState } from "react"
import { Pause, Play, RotateCcw, Scissors } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { Reference } from "@/lib/api"

export interface ReferenceEdits {
  start_seconds: number
  end_seconds: number | null
  noise_reduction: boolean
  auto_trim: boolean
}

interface Props {
  source: Reference
  disabled: boolean
  onApply: (edits: ReferenceEdits) => Promise<boolean>
  onPendingChange: (pending: boolean) => void
}

/** Every edit is measured against the immutable original, avoiding repeated cleanup. */
export function ReferenceEditor({ source, disabled, onApply, onPendingChange }: Props) {
  const [duration, setDuration] = useState(source.duration_seconds ?? 0)
  const [start, setStart] = useState("0")
  const [end, setEnd] = useState(source.duration_seconds?.toString() ?? "")
  const [noiseReduction, setNoiseReduction] = useState(source.noise_reduction ?? true)
  const [autoTrim, setAutoTrim] = useState(false)
  const [pending, setPending] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [previewError, setPreviewError] = useState("")
  const audio = useRef<HTMLAudioElement>(null)
  const from = Number(start), to = Number(end)
  const valid = start !== "" && end !== "" && Number.isFinite(from) && Number.isFinite(to)
    && from >= 0 && to <= duration + 0.000001 && to - from >= 1

  useEffect(() => {
    if (disabled && audio.current) audio.current.pause()
  }, [disabled])

  function changed() {
    audio.current?.pause()
    setPending(true)
    onPendingChange(true)
  }

  async function apply(restore = false) {
    if (disabled || (!restore && !valid)) return
    audio.current?.pause()
    const success = await onApply(restore
      ? { start_seconds: 0, end_seconds: null, noise_reduction: false, auto_trim: false }
      : { start_seconds: from, end_seconds: to, noise_reduction: noiseReduction, auto_trim: autoTrim })
    if (!success) return
    if (restore) { setStart("0"); setEnd(duration.toString()); setNoiseReduction(false); setAutoTrim(false) }
    setPending(false)
    onPendingChange(false)
  }

  async function previewSelection() {
    if (!audio.current || !valid || disabled) return
    if (playing) { audio.current.pause(); return }
    setPreviewError("")
    audio.current.currentTime = from
    try { await audio.current.play() }
    catch { setPreviewError("Audio preview is unavailable. You can still apply your edits.") }
  }

  return <details className="reference-editor">
    <summary><Scissors size={13} />Clean up and trim voice</summary>
    <section aria-label="Clean up and trim voice" className="reference-editor-body">
      <p className="trim-help">Keep the words you want, then apply. Edits always use the original recording.</p>
      <audio ref={audio} src={source.url} preload="metadata" onLoadedMetadata={event => {
        const seconds = event.currentTarget.duration
        if (Number.isFinite(seconds) && seconds > 0) {
          setDuration(seconds)
          setEnd(previous => previous || seconds.toString())
        }
      }} onTimeUpdate={event => { if (event.currentTarget.currentTime >= to) event.currentTarget.pause() }}
        onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} />
      <div className="trim-time-fields">
        <label>Start (seconds)<input type="number" min="0" max={Math.max(0, duration - 1)} step="0.01" value={start} disabled={disabled || !duration} onChange={event => { setStart(event.target.value); changed() }} /></label>
        <label>End (seconds)<input type="number" min="1" max={duration || undefined} step="0.01" value={end} disabled={disabled || !duration} onChange={event => { setEnd(event.target.value); changed() }} /></label>
      </div>
      <div className="trim-sliders">
        <input type="range" aria-label="Trim start" min="0" max={duration || 1} step="0.01" value={Number.isFinite(from) ? from : 0} disabled={disabled || !duration} onChange={event => { setStart(event.target.value); changed() }} />
        <input type="range" aria-label="Trim end" min="0" max={duration || 1} step="0.01" value={Number.isFinite(to) ? to : duration} disabled={disabled || !duration} onChange={event => { setEnd(event.target.value); changed() }} />
      </div>
      <div className="trim-selection"><span>{valid ? `${(to - from).toFixed(2)} seconds selected` : duration ? "Keep at least 1 second within the recording." : "Loading recording duration…"}</span><Button type="button" variant="ghost" size="sm" disabled={disabled || !valid} onClick={() => void previewSelection()}>{playing ? <Pause size={12} /> : <Play size={12} />}{playing ? "Pause preview" : "Preview selection"}</Button></div>
      <label className="cleanup-option"><input type="checkbox" checked={noiseReduction} disabled={disabled} onChange={event => { setNoiseReduction(event.target.checked); changed() }} /><span>Reduce noise and background hiss</span></label>
      <label className="cleanup-option"><input type="checkbox" checked={autoTrim} disabled={disabled} onChange={event => { setAutoTrim(event.target.checked); changed() }} /><span>Trim quiet edges automatically</span></label>
      <p className="trim-help">Gentle cleanup reduces steady noise. Speech or music in the background may remain. Listen before generating.</p>
      <div className="text-tools"><Button type="button" variant="outline" size="sm" disabled={disabled || !valid} onClick={() => void apply()}>Apply voice edits</Button><Button type="button" variant="ghost" size="sm" disabled={disabled} onClick={() => void apply(true)}><RotateCcw size={12} />Restore original</Button></div>
      {pending && <p className="trim-help" role="status">Apply your voice edits before generating speech.</p>}
      {previewError && <p className="field-error" role="alert">{previewError}</p>}
    </section>
  </details>
}
