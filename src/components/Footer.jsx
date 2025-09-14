import React from "react";
import { useLocation } from "react-router-dom";

export default function Footer() {
  const location = useLocation();
  const isAdminPage = location.pathname.startsWith("/admin");

  return (
    <div className="h-[300px] overflow-hidden">
      <div className="relative">
        <div
          className={`text-[200px] text-[#F9E9B7] font-bold absolute top-[70px] left-[-30px] ${
            isAdminPage ? "opacity-10" : ""
          }`}
        >
          손手잇다
        </div>
      </div>
      <div className="flex flex-col h-full items-end justify-end p-4 text-[18px] font-semibold text-[#444]">
        <p>Copyright ⓒ 손수잇다 all rights reserved.</p>
        <p>경기도 안양시 만안구 성결대학로 53</p>
        <p>대표자 : 이호연 | 대표전화 : 010-1234-5678</p>
      </div>
    </div>
  );
}
