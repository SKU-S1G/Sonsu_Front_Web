import React from "react";
import { useNavigate } from "react-router-dom";

export default function AdminTitle() {
  const navigate = useNavigate();
  
  return (
    <div className="py-8 fontSB text-[20px] text-[#333333] flex justify-center" onClick={() => navigate("/admin")}>
      SONSU CLASS
    </div>
  );
}
