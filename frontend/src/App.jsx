import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import Exam from "./pages/Exam.jsx";
import History from "./pages/History.jsx";
import Home from "./pages/Home.jsx";
import Intake from "./pages/Intake.jsx";

export default function App() {
  const { pathname } = useLocation();
  const isHome = pathname === "/";

  return (
    <div className={isHome ? "layout layout--home" : "layout layout--app"}>
      <header className="topbar">
        <Link to="/" className="brand" aria-label="Image Quality Clinic home">
          <span className="mark">IQ</span>
          <span>
            <strong>Image Quality Clinic</strong>
            <em>Civic still review · local inference</em>
          </span>
        </Link>
        <nav>
          <NavLink to="/" end>
            Home
          </NavLink>
          <NavLink to="/intake">New exam</NavLink>
          <NavLink to="/history">Past exams</NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/intake" element={<Intake />} />
          <Route path="/exam/:id" element={<Exam />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </main>
    </div>
  );
}
