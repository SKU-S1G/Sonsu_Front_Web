import React from "react";
import { useNavigate } from "react-router-dom";

export default function UserTitle() {
  const navigate = useNavigate();
  
  return (
    <div className="py-8 fontSB text-[25px] text-[#000] flex justify-center cursor-pointer" onClick={() => navigate("/")} >
      손手잇다
    </div>
  );
}
