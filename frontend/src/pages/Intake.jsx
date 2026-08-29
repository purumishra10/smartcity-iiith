import { useEffect, useId, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { analyzeImage } from "../api.js";

const CONTEXTS = [
  { id: "street", label: "Street still" },
  { id: "camera", label: "CCTV grab" },
  { id: "other", label: "Other" },
];

const EXAM_STEPS = [
  { at: 0, pct: 6, text: "Receiving the file and checking type and size…" },
  { at: 700, pct: 16, text: "Decoding pixels and building a working copy (max 512 px)…" },
  { at: 1500, pct: 28, text: "Measuring sharpness (Laplacian + Tenengrad) and luma…" },
  { at: 2300, pct: 40, text: "Measuring contrast, saturation, noise MAD, and entropy…" },
  { at: 3100, pct: 52, text: "Scoring JPEG blockiness, median residual, and local defects…" },
  { at: 4000, pct: 64, text: "Tiling the frame 16×16 for blur, exposure, noise, and defect maps…" },
  { at: 5000, pct: 76, text: "Computing FFT energy, MSCN variance, CLAHE, colour-cast, and glare…" },
  { at: 6000, pct: 88, text: "Running the hybrid CNN + feature model on this machine…" },
  { at: 7200, pct: 96, text: "Fusing score, label, diagnosis, and writing heatmap overlays…" },
];

export default function Intake() {
  const nav = useNavigate();
  const inputId = useId();
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [context, setContext] = useState("street");
  const [hot, setHot] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [pct, setPct] = useState(0);
  const [step, setStep] = useState("");
  const timers = useRef([]);

  useEffect(() => () => timers.current.forEach(clearTimeout), []);

  function take(f) {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setError("");
  }

  function clearTimers() {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  }

  async function submit() {
    if (!file || busy) return;
    setBusy(true);
    setError("");
    setPct(4);
    setStep(EXAM_STEPS[0].text);
    clearTimers();
    EXAM_STEPS.forEach((s) => {
      timers.current.push(
        setTimeout(() => {
          setPct(s.pct);
          setStep(s.text);
        }, s.at)
      );
    });

    const started = Date.now();
    try {
      const report = await analyzeImage(file, context);
      const wait = Math.max(0, 7800 - (Date.now() - started));
      await new Promise((r) => {
        const t = setTimeout(r, wait);
        timers.current.push(t);
      });
      clearTimers();
      setPct(100);
      setStep("Exam complete. Opening the report…");
      await new Promise((r) => setTimeout(r, 400));
      nav(`/exam/${report.id}`);
    } catch (err) {
      clearTimers();
      setBusy(false);
      setPct(0);
      setStep("");
      setError(err.message || "Analysis failed");
    }
  }

  return (
    <section className="page-card">
      <header className="page-head">
        <h1>Intake</h1>
        <p className="lede">
          Upload a civic still for review. Dr. Image checks sharpness, exposure, noise, and
          defects, then maps problem areas on the frame.
        </p>
      </header>

      <div className="intake-form">
        <div className="field">
          <span className="field-label">Image file</span>
          <div
            className="upload-wrap"
            onDragOver={(e) => {
              e.preventDefault();
              setHot(true);
            }}
            onDragLeave={() => setHot(false)}
            onDrop={(e) => {
              e.preventDefault();
              setHot(false);
              take(e.dataTransfer.files[0]);
            }}
          >
            <input
              id={inputId}
              className="file-input"
              type="file"
              accept="image/jpeg,image/png,image/webp,image/bmp"
              onChange={(e) => take(e.target.files[0])}
              disabled={busy}
            />
            <label htmlFor={inputId} className={`drop ${hot ? "hot" : ""}`}>
              {preview ? (
                <>
                  <img className="drop-preview" src={preview} alt="Selected preview" />
                  <p className="muted">{file?.name}</p>
                  <p className="field-hint">Click or drop to replace</p>
                </>
              ) : (
                <>
                  <span className="drop-icon" aria-hidden="true">
                    ↑
                  </span>
                  <p className="drop-title">Choose a photo or drag it here</p>
                  <p className="field-hint">JPEG, PNG, WebP, or BMP · max 10 MB</p>
                </>
              )}
            </label>
          </div>
        </div>

        <div className="field">
          <span className="field-label">Source type</span>
          <div className="radio-row" role="radiogroup" aria-label="Image source type">
            {CONTEXTS.map((c) => (
              <label key={c.id} className="radio-option">
                <input
                  type="radio"
                  name="context"
                  value={c.id}
                  checked={context === c.id}
                  onChange={() => setContext(c.id)}
                  disabled={busy}
                />
                {c.label}
              </label>
            ))}
          </div>
        </div>

        <div className="actions">
          <button className="btn" type="button" disabled={!file || busy} onClick={submit}>
            {busy ? "Exam in progress…" : "Run quality exam"}
          </button>
          {!file && <span className="muted">Select an image to continue</span>}
        </div>

        {busy && (
          <div className="exam-progress" role="status" aria-live="polite">
            <p className="exam-progress-label">{step}</p>
            <div className="exam-progress-track">
              <span style={{ width: `${pct}%` }} />
            </div>
            <p className="muted">{pct}% · typically 5–10 seconds on CPU</p>
          </div>
        )}

        {error && <div className="err">{error}</div>}
      </div>
    </section>
  );
}
