import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';

import { AuthProvider } from './context/AuthContext';
import ProtectedRoute   from './components/ProtectedRoute';
import Navigation       from './components/Navigation';
import Footer           from './components/Footer';

// Pages
import LandingPage   from './pages/LandingPage';
import SearchPage    from './pages/SearchPage';
import LoginPage     from './pages/LoginPage';
import AssistantPage from './pages/AssistantPage';

// Inner layout — hides Footer on /login and /assistant
function AppLayout() {
  const location = useLocation();
  const isLoginPage = location.pathname === '/login';
  const isAssistant = location.pathname.startsWith('/assistant');
  const hideFooter = isLoginPage || isAssistant;
  const hideNav = isLoginPage;

  return (
    <div className="min-h-screen flex flex-col" style={{ backgroundColor: '#EAEAE4' }}>
      {!hideNav && <Navigation />}
      <main className="flex-1">
        <Routes>
          {/* ── Fully public ─────────────────────────────────── */}
          <Route path="/"      element={<LandingPage />} />
          <Route path="/login" element={<LoginPage />}   />

          {/* ── Auth-gated: Unified Assistant Workspace ───────── */}
          <Route path="/search" element={
            <ProtectedRoute><SearchPage /></ProtectedRoute>
          } />
          
          <Route path="/assistant" element={
            <ProtectedRoute><AssistantPage /></ProtectedRoute>
          } />
          
          <Route path="/assistant/:sessionId" element={
            <ProtectedRoute><AssistantPage /></ProtectedRoute>
          } />

          {/* ── Legacy Redirects (V1 -> V2) ───────────────────── */}
          <Route path="/analysis"            element={<Navigate to="/assistant" replace />} />
          <Route path="/analyze/:filename"   element={<Navigate to="/assistant" replace />} />
          <Route path="/legal-chat"          element={<Navigate to="/assistant" replace />} />
          <Route path="/confidential-upload" element={<Navigate to="/assistant" replace />} />
          <Route path="/documents"           element={<Navigate to="/assistant" replace />} />
          <Route path="/chat/:sessionId"     element={<Navigate to="/assistant" replace />} />
          
          {/* Catch-all */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      {!hideFooter && <Footer />}
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <Router>
        <AppLayout />
      </Router>
    </AuthProvider>
  );
}

export default App;
