import { BrowserRouter, Routes, Route } from "react-router-dom";
import User from "./routes/User";
import Admin from "./routes/Admin";
import ScrollToTop from "./components/ScrollToTop";

function App() {
  return (
    <div className="flex App">
      <BrowserRouter>
        <ScrollToTop /> {/* 경로 바뀔 때마다 스크롤 맨 위 */}
        <div className="flex-1">
          <Routes>
            <Route path="/*" element={<User />} />
            <Route path="/admin/*" element={<Admin />} />
          </Routes>
        </div>
      </BrowserRouter>
    </div>
  );
}

export default App;
