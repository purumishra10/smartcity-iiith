import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext.jsx";

export default function Signup() {
  const { signup } = useAuth();
  const nav = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  async function onSubmit(e) {
    e.preventDefault();
    setError("");
    try {
      await signup(email, password);
      nav("/history");
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <section className="page-card auth-card">
      <header className="page-head">
        <h1>Create an account</h1>
        <p className="lede">
          Free to use without signing up. Create an account to save this browser session’s exams into Past
          exams.
        </p>
      </header>
      <form className="auth-form" onSubmit={onSubmit}>
        <label>
          Email
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </label>
        <label>
          Password (8+ characters)
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8} />
        </label>
        {error && <div className="err">{error}</div>}
        <button className="btn" type="submit">
          Sign up
        </button>
      </form>
      <p className="muted">
        Already registered? <Link to="/login">Log in</Link>
      </p>
    </section>
  );
}
