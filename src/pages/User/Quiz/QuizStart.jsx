import React from "react";
import { NavLink } from "react-router-dom";

export default function QuizStart() {
  return (
    <div>
      <div className="flex justify-center h-screen items-centers bg-gradient-to-b from-[#fffdef]">
        <div className="flex w-[100%]">
          <div className="flex justify-center items-center w-[100%]">
            <img src="/images/Vv.png" alt="report" className="flex w-[450px]" />
            <div className="flex flex-col items-center justify-center">
              <div className="flex w-fit text-center font-bold text-[50px]">
                <div className="">
                  오늘 배운 수어를
                  <div className="flex mt-[30px]">
                    <div className="animate-bounce text-[60px]">OX 퀴즈</div>를
                    통해 복습해 보세요
                  </div>
                </div>
              </div>
              <NavLink
                to="/QuizInfo"
                className="shadow-2xl mt-[70px] px-10 py-3 text-[30px] font-bold text-white transition-colors bg-yellow-500 rounded-lg hover:bg-yellow-400"
              >
                시작하기
              </NavLink>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
