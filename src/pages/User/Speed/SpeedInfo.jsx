import React from "react";
import UserNav from "../../../components/UserNav";
import UserTitle from '../../../components/UserTitle';
import { useNavigate } from 'react-router-dom';
// import { API_URL } from "../../../config";
// import { getToken } from "../../../utils/authStorage";

export default function Classroom() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-[#F28079]">
      <UserTitle />
      <div className="flex w-full">
        <UserNav />
        <div className="flex flex-col mr-10 w-full rounded-[40px] bg-[#fafafa] shadow-xl h-[850px] px-12 py-9 overflow-y-auto">
          <img src="/assets/images/review/speedTitle.png" alt="스피드 타이틀" className="w-[180px] m-3" />

          <div className="flex justify-around w-full h-full">
            <div className="flex items-center justify-center">
              <div>
                <img src="/assets/images/sonsu.png" alt="sonsu" className="w-[250px]" />
                <div className="flex mt-5 text-[20px] text-[#000000] fontMedium text-center justify-center">스피드 퀴즈를 위해<br/> 카메라를 준비해주세요</div>
              </div>
            </div>

            <div className="flex flex-col items-center justify-center">
              <div className="flex text-[30px] fontSB">📌  주의사항 📌</div>

              <div className="flex text-[20px] mt-16 text-center">☝🏻 정확도가 80% 이상일 때 <br />자동으로 다음 단어로 넘어가요</div>
              <div className="flex text-[20px] mt-16 text-center">✌🏻 배경이 너무 밝거나 어두우면 <br /> 인식이 잘 안 될 수 있어요</div>

              <div 
                className="flex text-[#fff] text-[20px] px-12 py-3 mt-12 rounded-full bg-[#F28079]"
                onClick={() => navigate("/speed")}
              >
                시작하기
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
