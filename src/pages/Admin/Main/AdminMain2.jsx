import React from "react";

export default function AdminMain2() {
  return (
    <div>
      {/* 흰배경 */}
      <div className="text-[#404040] text-[25px] leading-10 w-full bg-white h-[300px] flex flex-col items-center justify-center">
        <p>효율적이고 스마트한 수어 교육 관리,</p>
        <p>
          <span className="text-[#000] font-black text-[30px] mr-1">
            손수클래스
          </span>
          로 시작하세요.
        </p>
      </div>

      {/* 전체 화면 정렬 */}
      <div className="h-screen flex items-center justify-center">
        <div className="flex flex-col items-center gap-12">
          {/* 타이틀 */}
          <div className="text-center">
            <p className="text-[40px] text-[#333] font-semibold">SONSU CLASS</p>
            <p className="text-[20px] text-[#333] mx-auto mt-4">
              원하는 강의를 자유롭게 구성하고, 그룹별 수강생을 관리하며 맞춤형
              학습과 성적 확인까지 가능한 구독 기반 수어 클래스 서비스
            </p>
          </div>

          {/* 3개 카드 */}
          <div className="flex items-center justify-center gap-12">
            <div className="flex flex-col items-center">
              <img
                src="/assets/images/Admin/Main/AdminMain2_1.png"
                alt=""
                className="w-[400px]"
              />
              <p className="text-[24px] text-[#333] font-bold">강의 커리큘럼</p>
              <div className="text-[#333] text-[18px] flex flex-col items-center mt-2">
                <p>원하는 강의만 쏙쏙!</p>
                <p>뭐 어쩌구저쩌구</p>
              </div>
            </div>

            <div className="flex flex-col items-center">
              <img
                src="/assets/images/Admin/Main/AdminMain2_2.png"
                alt=""
                className="w-[400px]"
              />
              <p className="text-[24px] text-[#333] font-bold">수강그룹 관리</p>
              <div className="text-[#333] text-[18px] flex flex-col items-center mt-2">
                <p>그룹별 수강생 관리 OK!</p>
                <p>뭐 어쩌구저쩌구</p>
              </div>
            </div>

            <div className="flex flex-col items-center">
              <img
                src="/assets/images/Admin/Main/AdminMain2_3.png"
                alt=""
                className="w-[400px]"
              />
              <p className="text-[24px] text-[#333] font-bold">수강생 관리</p>
              <div className="text-[#333] text-[18px] flex flex-col items-center mt-2">
                <p>한눈에 성적/진도 확인!</p>
                <p>뭐 어쩌구저쩌구</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
