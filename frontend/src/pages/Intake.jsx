import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { analyzeImage } from "../api.js";

const CONTEXTS = ["street", "camera", "other"];

export default function Intake() {
  const nav = useNavigate();
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
    <section className="panel">
      <h1>Intake</h1>
      <p className="lede">
        Upload a civic still (street, CCTV grab, incident photo). The clinic measures vitals,
        runs a local quality model, and maps where problems sit.
      </p>
      <label
        className={`drop ${hot ? "hot" : ""}`}
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
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          hidden
          onChange={(e) => take(e.target.files[0])}
        />
        {preview ? <img src={preview} alt="preview" style={{ maxHeight: 220, maxWidth: "100%" }} /> : <p>Drop an image or click to choose</p>}
        {file && <p className="muted">{file.name}</p>}
      </label>
      <div className="row">
        {CONTEXTS.map((c) => (
          <button key={c} className={`chip ${context === c ? "on" : ""}`} type="button" onClick={() => setContext(c)}>
            {c}
          </button>
        ))}
      </div>
      <button className="btn" type="button" disabled={!file || busy} onClick={submit}>
        {busy ? "Examining…" : "Start exam"}
      </button>
      {error && <div className="err">{error}</div>}
    </section>
  );
}
