import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { assetUrl, listAnalyses } from "../api.js";
import { useAuth } from "../AuthContext.jsx";

export default function History() {
  const { user, ready } = useAuth();
  const [rows, setRows] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!ready) return;
    listAnalyses()
      .then(setRows)
      .catch((e) => setError(e.message));
  }, [ready, user]);

  if (error) return <div className="err">{error}</div>;
  if (!ready || !rows) return <p className="busy">Loading past exams…</p>;

  return (
    <section className="page-card">
      <header className="page-head">
        <h1>Past exams</h1>
        <p className="lede">
          {user
            ? "Saved reviews for your account, most recent first."
            : "Exams from this browser session. Create an account to keep them after you close the browser."}
        </p>
      </header>

      {!user && (
        <div className="save-banner">
          These results stay on this device session. <Link to="/signup">Sign up</Link> or{" "}
          <Link to="/login">log in</Link> to keep Past exams on your account.
        </div>
      )}

      {!rows.length && (
        <p className="muted">
          Nothing here yet. <Link to="/intake">Upload a still</Link> to run an exam.
        </p>
      )}

      {rows.length > 0 && (
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
      )}
    </section>
  );
}
