import { Link, NavLink, Route, Routes } from "react-router-dom";
import Exam from "./pages/Exam.jsx";
import History from "./pages/History.jsx";
import Intake from "./pages/Intake.jsx";

export default function App() {
  return (
    <div className="shell">
      <header className="topbar">
        <Link to="/" className="brand">
          <span className="mark">IQ</span>
          <span>
            <strong>Image Quality Clinic</strong>
            <em>Civic still review · local inference</em>
          </span>
        </Link>
        <nav>
          <NavLink to="/" end>
            New exam
          </NavLink>
          <NavLink to="/history">Past exams</NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Intake />} />
          <Route path="/exam/:id" element={<Exam />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </main>
    </div>
  );
}
