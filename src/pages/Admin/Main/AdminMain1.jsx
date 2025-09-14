import React from "react";
import AdminTitle from "../../../components/AdminTitle";
import { useNavigate } from "react-router-dom";

export default function AdminMain1() {
  const navigate = useNavigate();

  return (
    <div className="h-screen relative">
      <AdminTitle />
      <div className="flex flex-col items-center justify-center h-screen mt-[-150px]">
        <div className="text-[80px] font-semibold text-[#333]">Your Pace</div>
        <div className="text-[80px] font-semibold text-[#333] mt-[-30px]">
          Your Space
        </div>
        <button
          className="bg-[#e7e7e7] rounded-[15px] text-[#222] text-[24px] px-6 py-2 mt-10"
          onClick={() => navigate("/admin/ClassList")}
        >
          SONSU CLASS
        </button>
      </div>
      <img
        src="/assets/images/Admin/Main/AdminMain1_1.png"
        alt=""
        className="w-[400px] absolute top-24 left-32"
      />
      <img
        src="/assets/images/Admin/Main/AdminMain1_2.png"
        alt=""
        className="w-[400px] absolute top-48 right-[200px]"
      />
      <img
        src="/assets/images/Admin/Main/AdminMain1_3.png"
        alt=""
        className="w-[400px] absolute top-[500px] right-[100px]"
      />
    </div>
  );
}
