import React from "react";
import { CgShapeCircle } from "react-icons/cg";
import UserNav from "../../../components/UserNav";
import UserTitle from "../../../components/UserTitle";
import { useNavigate } from "react-router-dom";

export default function QuizInfo() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-[#F28079]">
      <UserTitle />
      <div className="flex w-full">
        <UserNav />
        <div className="flex flex-col mr-10 w-full rounded-[40px] bg-[#fafafa] shadow-xl h-[850px] px-12 py-9 overflow-y-auto">
          <img
            src="/assets/images/review/OXTitle.png"
            alt="OX 타이틀"
            className="w-[150px] m-3"
          />
          <div className="flex justify-around w-full h-full">
            <div className="flex items-center justify-center">
              <div>
                <img
                  src="/assets/images/sonsu.png"
                  alt="sonsu"
                  className="w-[250px]"
                />
                <div className="flex mt-5 text-[20px] text-[#000000] fontMedium text-center justify-center">
                  캐릭터의 수어를 보고
                  <br /> 정답을 맞춰보세요!
                </div>
              </div>
            </div>

            <div className="flex flex-col items-center justify-center -mt-20">
              <div className="flex text-[30px] fontSB">📌 Tips 📌</div>
              <div className="flex text-[20px] mt-16 text-center">
                배움터에서 5개 이상의 학습을 진행해주세요! 📚
              </div>
              <div className="flex space-x-3">
                <div
                  className="flex text-[#fff] fontSB text-[20px] px-12 py-3 mt-12 rounded-full bg-[#F28079]"
                  onClick={() => navigate("/Classroom/easy")}
                >
                  학습하러 가기
                </div>{" "}
                <div
                  className="flex text-[#fff] fontSB text-[20px] px-12 py-3 mt-12 rounded-full bg-[#F28079]"
                  onClick={() => navigate("/quiz")}
                >
                  시작하기{" "}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
