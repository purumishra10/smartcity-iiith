import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext.jsx";

export default function Login() {
  const { login } = useAuth();
  const nav = useNavigate();
  const loc = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const next = loc.state?.next || "/history";

  async function onSubmit(e) {
    e.preventDefault();
    setError("");
    try {
      await login(email, password);
      nav(next);
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <section className="page-card auth-card">
      <header className="page-head">
        <h1>Log in</h1>
        <p className="lede">Exams stay free. An account is only required to keep Past exams.</p>
      </header>
      <form className="auth-form" onSubmit={onSubmit}>
        <label>
          Email
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </label>
        <label>
          Password
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8} />
        </label>
        {error && <div className="err">{error}</div>}
        <button className="btn" type="submit">
          Log in
        </button>
      </form>
      <p className="muted">
        No account? <Link to="/signup">Sign up</Link>
      </p>
    </section>
  );
}
