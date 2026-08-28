import { useId, useState } from "react";
import { useNavigate } from "react-router-dom";
import { analyzeImage } from "../api.js";

const CONTEXTS = [
  { id: "street", label: "Street still" },
  { id: "camera", label: "CCTV grab" },
  { id: "other", label: "Other" },
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

  function take(f) {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setError("");
  }

  async function submit() {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const report = await analyzeImage(file, context);
      nav(`/exam/${report.id}`);
    } catch (err) {
      setError(err.message || "Analysis failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="page-card">
      <header className="page-head">
        <h1>Intake</h1>
        <p className="lede">
          Upload a civic still for review. The clinic checks sharpness, exposure, noise, and
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
                />
                {c.label}
              </label>
            ))}
          </div>
        </div>

        <div className="actions">
          <button className="btn" type="button" disabled={!file || busy} onClick={submit}>
            {busy ? "Running exam…" : "Run quality exam"}
          </button>
          {!file && <span className="muted">Select an image to continue</span>}
        </div>

        {error && <div className="err">{error}</div>}
      </div>
    </section>
  );
}
