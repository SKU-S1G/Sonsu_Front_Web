import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Lottie from "react-lottie-player";
import MainHeader from "./MainHeader";

export default function Main1() {
  const [animationData, setAnimationData] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    fetch("/assets/animations/main.json")
      .then((res) => res.json())
      .then(setAnimationData);
  }, []);

  return (
    <div className="relative flex flex-col justify-center px-24 py-8">
      <MainHeader />

      {/* 상단 텍스트 */}
      <div className="flex flex-col items-center mt-6">
        <div className="text-[#3c3c3c] text-[25px]">
          손으로 이어지는 새로운 소통
        </div>
        <div className="text-[#121212] text-[80px] fontBold mb-12">
          손手잇다
        </div>
        <div className="flex">
          <button
            className="bg-[#FFE694] shadow-xl px-24 py-2 rounded-full text-[23px]"
            onClick={() => navigate("/home")}
          >
            시작하기
          </button>
        </div>
      </div>

      <div className="w-full mt-32">
        {animationData && (
          <Lottie
            loop
            animationData={animationData}
            play
            style={{ width: "100%", height: "100%" }}
          />
        )}
      </div>
    </div>
  );
}
