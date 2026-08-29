import { createContext, useContext, useEffect, useState } from "react";
import { getMe, login as apiLogin, logout as apiLogout, signup as apiSignup } from "./api.js";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(false);

  async function refresh() {
    try {
      const me = await getMe();
      setUser(me.authenticated ? me : null);
    } catch {
      setUser(null);
    } finally {
      setReady(true);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function signup(email, password) {
    const res = await apiSignup(email, password);
    await refresh();
    return res;
  }

  async function login(email, password) {
    const res = await apiLogin(email, password);
    await refresh();
    return res;
  }

  async function logout() {
    await apiLogout();
    setUser(null);
  }

  return (
    <AuthContext.Provider value={{ user, ready, signup, login, logout, refresh }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
