import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { assetUrl, listAnalyses } from "../api.js";

export default function History() {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    listAnalyses()
      .then(setRows)
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="err">{error}</div>;
  if (!rows) return <p className="busy">Loading chart…</p>;
  if (!rows.length) return <p className="muted">No visits yet. Run an intake exam first.</p>;

  return (
    <section>
      <h1>Chart</h1>
      <p className="lede">Previous clinic visits, newest first.</p>
      <div className="history">
        {rows.map((row) => (
          <Link key={row.id} className="visit" to={`/exam/${row.id}`}>
            <img src={assetUrl(row.thumbnail_url)} alt="" />
            <div>
              <strong>{row.quality_label}</strong>
              <div className="muted">
                Score {row.quality_score}
                {row.context ? ` · ${row.context}` : ""}
              </div>
            </div>
            <span className="muted">{new Date(row.created_at).toLocaleString()}</span>
          </Link>
        ))}
      </div>
    </section>
  );
}
