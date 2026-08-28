import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { assetUrl, getAnalysis } from "../api.js";

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

function vitalPct(stats, key) {
  const v = Number(stats?.[key] ?? 0);
  return Math.max(0, Math.min(100, Math.round(v * 100)));
}

export default function Exam() {
  const { id } = useParams();
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");
  const [overlay, setOverlay] = useState("blur");
  const [pickedIssue, setPickedIssue] = useState(null);
  const [local, setLocal] = useState(null);

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

  if (error) return <div className="err">{error}</div>;
  if (!report) return <p className="busy">Loading exam report…</p>;

  const stats = report.statistics || {};

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
      </div>

      <aside className="panel report-panel">
        <h2>Exam report</h2>
        <div className="scoreline">
          <div className="score">{report.quality_score}</div>
          <span className={`badge ${report.quality_label}`}>{report.quality_label}</span>
        </div>
        <p className="diagnosis">{report.diagnosis}</p>

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

        <div className="vitals">
          <p className="vitals-title">Measured vitals</p>
          <Vital label="Sharpness" pct={vitalPct(stats, "sharpness") * 4} raw={stats.sharpness} />
          <Vital label="Brightness" pct={vitalPct(stats, "brightness")} raw={stats.brightness} />
          <Vital label="Contrast" pct={vitalPct(stats, "contrast") * 2} raw={stats.contrast} />
          <Vital label="Noise" pct={vitalPct(stats, "noise_estimate") * 4} raw={stats.noise_estimate} />
          <Vital label="Saturation" pct={vitalPct(stats, "saturation")} raw={stats.saturation} />
        </div>

        <p className="muted" style={{ marginTop: 16 }}>
          <Link to="/intake">Run another exam</Link>
          {" · "}
          <Link to="/history">View past exams</Link>
        </p>
      </aside>
    </section>
  );
}

function Vital({ label, pct, raw }) {
  return (
    <div className="vital">
      <span>{label}</span>
      <div className="bar">
        <span style={{ width: `${Math.max(4, Math.min(100, pct))}%` }} />
      </div>
      <span>{Number(raw ?? 0).toFixed(2)}</span>
    </div>
  );
}
