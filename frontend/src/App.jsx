import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { AuthProvider, useAuth } from "./AuthContext.jsx";
import BrandMark from "./BrandMark.jsx";
import Exam from "./pages/Exam.jsx";
import History from "./pages/History.jsx";
import Home from "./pages/Home.jsx";
import Intake from "./pages/Intake.jsx";
import Login from "./pages/Login.jsx";
import Signup from "./pages/Signup.jsx";

function Shell() {
  const { pathname } = useLocation();
  const isHome = pathname === "/";
  const { user, logout } = useAuth();

  return (
    <div className={isHome ? "layout layout--home" : "layout layout--app"}>
      <header className="topbar">
        <Link to="/" className="brand" aria-label="Dr. Image home">
          <BrandMark />
          <span>
            <strong>Dr. Image</strong>
            <em>Image quality exams · local inference</em>
          </span>
        </Link>
        <nav>
          <NavLink to="/" end>
            Home
          </NavLink>
          <NavLink to="/intake">New exam</NavLink>
          <NavLink to="/history">Past exams</NavLink>
          {user ? (
            <>
              <span className="nav-user">{user.email}</span>
              <button type="button" className="nav-text" onClick={logout}>
                Log out
              </button>
            </>
          ) : (
            <>
              <NavLink to="/login">Log in</NavLink>
              <NavLink to="/signup">Sign up</NavLink>
            </>
          )}
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/intake" element={<Intake />} />
          <Route path="/exam/:id" element={<Exam />} />
          <Route path="/history" element={<History />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Shell />
    </AuthProvider>
  );
}
