import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { assetUrl, getAnalysis, reportUrl } from "../api.js";
import { useAuth } from "../AuthContext.jsx";

const OVERLAYS = [
  { id: "blur", label: "Blur" },
  { id: "exposure", label: "Exposure" },
  { id: "noise", label: "Noise" },
  { id: "defect", label: "Defect" },
];

const ISSUE_OVERLAY = {
  blur: "blur",
  underexposure: "exposure",
  overexposure: "exposure",
  noise: "noise",
  defect: "defect",
  corruption: null,
};

export default function Exam() {
  const { id } = useParams();
  const { user } = useAuth();
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");
  const [overlay, setOverlay] = useState("blur");
  const [pickedIssue, setPickedIssue] = useState(null);
  const [local, setLocal] = useState(null);
  const [busyPdf, setBusyPdf] = useState(false);
  const [previewUrl, setPreviewUrl] = useState("");

  useEffect(() => {
    getAnalysis(id)
      .then(setReport)
      .catch((e) => setError(e.message));
  }, [id]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    getAnalysis(id)
      .then(setReport)
      .catch((e) => setError(e.message));
  }, [id]);

  const overlayUrl = useMemo(() => {
    if (!report?.heatmaps?.[overlay]) return "";
    return assetUrl(report.heatmaps[overlay]);
  }, [report, overlay]);

  function onClick(e) {
    if (!report?.grid) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    const col = Math.min(report.grid.cols - 1, Math.max(0, Math.floor(x * report.grid.cols)));
    const row = Math.min(report.grid.rows - 1, Math.max(0, Math.floor(y * report.grid.rows)));
    const tile = report.grid.tiles[row * report.grid.cols + col];
    setLocal({ row, col, tile });
    const order = ["blur", "defect", "noise", "underexposure", "overexposure"];
    const worst = order.reduce((a, k) => (tile[k] > (tile[a] || 0) ? k : a), order[0]);
    if (tile[worst] > 0.25) {
      setPickedIssue(worst);
      const mapped = ISSUE_OVERLAY[worst];
      if (mapped) setOverlay(mapped);
    }
  }

  async function loadPdfBlob() {
    const res = await fetch(reportUrl(id), { credentials: "include" });
    if (!res.ok) throw new Error("report_failed");
    return res.blob();
  }

  async function openPreview() {
    setBusyPdf(true);
    setError("");
    try {
      const blob = await loadPdfBlob();
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(URL.createObjectURL(blob));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusyPdf(false);
    }
  }

  function closePreview() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl("");
  }

  function savePdf(url) {
    const a = document.createElement("a");
    a.href = url;
    a.download = `dr-image-${id.slice(0, 8)}.pdf`;
    a.click();
  }

  async function downloadPdf() {
    if (previewUrl) {
      savePdf(previewUrl);
      return;
    }
    setBusyPdf(true);
    try {
      const blob = await loadPdfBlob();
      const url = URL.createObjectURL(blob);
      savePdf(url);
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusyPdf(false);
    }
  }

  if (error) return <div className="err">{error}</div>;
  if (!report) return <p className="busy">Loading exam report…</p>;

  const vitals = report.vital_explanations || [];
  const issueNotes = report.issue_explanations || [];
  const desc = report.frame_description;
  const descText = desc?.full || desc?.appearance || report.diagnosis;

  return (
    <section className="exam">
      <div className="viewer">
        <p className="viewer-label">Frame + heatmap</p>
        <div className="stage">
          <img className="base" src={assetUrl(report.image_url)} alt="Uploaded still" />
          {overlayUrl && <img className="overlay" src={overlayUrl} alt={`${overlay} overlay`} />}
          <div className="grid-hit" onClick={onClick} title="Click a region to inspect" />
        </div>
        <div className="toggles">
          {OVERLAYS.map((o) => (
            <button
              key={o.id}
              type="button"
              className={`toggle-btn ${overlay === o.id ? "on" : ""}`}
              onClick={() => setOverlay(o.id)}
            >
              {o.label}
            </button>
          ))}
        </div>
        {local && (
          <div className="local">
            Tile {local.row + 1}×{local.col + 1} — blur {local.tile.blur.toFixed(2)}, under{" "}
            {local.tile.underexposure.toFixed(2)}, over {local.tile.overexposure.toFixed(2)}, noise{" "}
            {local.tile.noise.toFixed(2)}, defect {local.tile.defect.toFixed(2)}
          </div>
        )}
        {descText && (
          <div className="frame-copy">
            <p className="vitals-title">What's in the frame</p>
            {desc?.appearance ? (
              <>
                <p>{desc.appearance}</p>
                {desc.usefulness && <p className="muted">{desc.usefulness}</p>}
              </>
            ) : (
              <p>{descText}</p>
            )}
          </div>
        )}
      </div>

      <aside className="panel report-panel">
        <h2>Exam report</h2>
        <div className="scoreline">
          <div className="score">{report.quality_score}</div>
          <span className={`badge ${report.quality_label}`}>{report.quality_label}</span>
        </div>
        <p className="diagnosis">{report.diagnosis}</p>

        {!user && (
          <div className="save-banner">
            This exam is listed under Past exams for this browser session.{" "}
            <Link to="/signup">Create an account</Link> to keep it after you close the browser.
          </div>
        )}

        <div className="report-actions">
          <button className="btn-ghost" type="button" onClick={openPreview} disabled={busyPdf}>
            {busyPdf && !previewUrl ? "Loading preview…" : "Preview report"}
          </button>
          <button className="btn" type="button" onClick={downloadPdf} disabled={busyPdf}>
            Download detailed report
          </button>
        </div>

        <div className="issues">
          {(report.issues || []).length === 0 && (
            <span className="muted">No issues above the confidence threshold.</span>
          )}
          {(report.issues || []).map((iss) => (
            <button
              key={iss.type}
              type="button"
              className={`issue ${pickedIssue === iss.type ? "on" : ""}`}
              onClick={() => {
                setPickedIssue(iss.type);
                const mapped = ISSUE_OVERLAY[iss.type];
                if (mapped) setOverlay(mapped);
              }}
            >
              <strong>{iss.type.replace("_", " ")}</strong>
              <div className="muted">
                {iss.severity} severity · {(iss.confidence * 100).toFixed(0)}% confidence
              </div>
            </button>
          ))}
        </div>

        {issueNotes.length > 0 && (
          <div className="explain-block">
            <p className="vitals-title">Why these issues</p>
            {issueNotes.map((note) => (
              <details key={note.type} className="explain" open={pickedIssue === note.type}>
                <summary>
                  {note.type.replace("_", " ")} · {(note.confidence * 100).toFixed(0)}%
                </summary>
                <p>{note.why}</p>
                <p className="muted">{note.evidence}</p>
                {note.operator_note && <p>{note.operator_note}</p>}
              </details>
            ))}
          </div>
        )}

        <div className="vitals">
          <p className="vitals-title">Measured vitals</p>
          {vitals.length
            ? vitals.map((v) => <Vital key={v.id} item={v} />)
            : null}
        </div>

        {(report.issue_heads || []).length > 0 && (
          <div className="explain-block">
            <p className="vitals-title">All six issue heads</p>
            {(report.issue_heads || []).map((h) => (
              <p key={h.type} className="ledger-line">
                <strong>{h.type.replace("_", " ")}</strong> {(h.confidence * 100).toFixed(1)}%
                {h.listed ? ` · listed (${h.severity})` : " · below 0.42 gate"}
                {h.heatmap ? ` · map: ${h.heatmap}` : " · no heatmap (global)"}
              </p>
            ))}
          </div>
        )}

        {(report.measurements || []).length > 0 && (
          <div className="explain-block">
            <p className="vitals-title">Every measurement used</p>
            {(report.measurements || []).map((m) => (
              <details key={m.id} className="explain">
                <summary>
                  {m.label} · {m.raw} {m.zscore != null ? `(z ${m.zscore})` : ""}
                </summary>
                <p>{m.meaning}</p>
                <p className="muted">Feeds: {m.feeds}</p>
              </details>
            ))}
          </div>
        )}

        {report.fusion?.rules && (
          <div className="explain-block">
            <p className="vitals-title">Fusion rules</p>
            <ul className="rule-list">
              {report.fusion.rules.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          </div>
        )}

        <p className="muted" style={{ marginTop: 16 }}>
          <Link to="/intake">Run another exam</Link>
          {" · "}
          <Link to="/history">View past exams</Link>
        </p>
      </aside>

      {previewUrl && (
        <div className="pdf-modal" role="dialog" aria-modal="true" aria-label="Report preview">
          <div className="pdf-modal-panel">
            <header className="pdf-modal-head">
              <h2>Report preview</h2>
              <div className="pdf-modal-actions">
                <button className="btn" type="button" onClick={() => savePdf(previewUrl)}>
                  Download
                </button>
                <button className="btn-ghost" type="button" onClick={closePreview}>
                  Close
                </button>
              </div>
            </header>
            <iframe className="pdf-frame" title="Dr. Image report preview" src={previewUrl} />
          </div>
        </div>
      )}
    </section>
  );
}

function Vital({ item }) {
  return (
    <div className="vital vital--deep">
      <div className="vital-row">
        <span>{item.label}</span>
        <div className="bar">
          <span style={{ width: `${Math.max(4, Math.min(100, item.display))}%` }} />
        </div>
        <span>{item.display}</span>
      </div>
      <p className="vital-why">{item.why}</p>
      <p className="muted vital-hint">{item.meaning}</p>
      <p className="vital-action">{item.action}</p>
    </div>
  );
}
