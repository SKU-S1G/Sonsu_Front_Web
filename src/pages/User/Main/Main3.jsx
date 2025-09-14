import React from "react";

export default function Main3() {
  return (
    <div className="h-screen px-32 mt-20">
      {/* title */}
      <div className="flex flex-col">
        <div className="text-[32px] font-light">수어, 이제 어렵지 않아요</div>
        <div className="text-[36px] font-semibold">
          필요한 기능만{" "}
          <span className="relative inline-block">
            <span className="absolute bottom-1.5 left-0 w-full h-[0.4em] bg-[#FFC400] rounded-full"></span>
            <span className="relative z-10 font-black">쏙쏙</span>
          </span>{" "}
          담은, 똑똑한{" "}
          <span className="relative inline-block">
            <span className="absolute bottom-1.5 left-0 w-full h-[0.4em] bg-[#FFC400] rounded-full"></span>
            <span className="relative z-10 font-black">메인화면</span>
          </span>
        </div>
      </div>

      {/* 설명 */}
      <div className="relative ml-20 mt-20">
        <img
          src="./assets/images/Mockup/Main.png"
          className="w-[280px] absolute top-32 z-10"
          alt=""
        />
        <img
          src="./assets/images/Mockup/MainScroll.png"
          className="w-[350px] absolute left-48 z-10"
          alt=""
        />
        <div className="relative">
          <div className="w-[300px] absolute h-[2px] bg-[#999] absolute top-[120px] left-[500px]"></div>
          <div className="w-[30px] h-[30px] rounded-full bg-[#FFE694] absolute top-[107px] left-[800px]"></div>
          <div className="absolute top-[100px] left-[870px]">
            <div className="text-[28px] font-bold">오늘의 출석</div>
            <div className="text-[23px] font-normal text-[#777]">
              하루 한 번, 수어 학습의 시작을 알리는 버튼!
            </div>
          </div>
        </div>
        <div className="relative">
          <div className="w-[440px] absolute h-[2px] bg-[#999] absolute top-[280px] left-[500px]"></div>
          <div className="w-[30px] h-[30px] rounded-full bg-[#FFE694] absolute top-[265px] left-[920px]"></div>
          <div className="absolute top-[260px] left-[980px]">
            <div className="text-[28px] font-bold">이어서 학습하기</div>
            <div className="text-[23px] font-normal text-[#777]">
              중단했던 학습, 한 번에 이어가기!
            </div>
          </div>
        </div>{" "}
        <div className="relative">
          <div className="w-[500px] absolute h-[2px] bg-[#999] absolute top-[530px] left-[500px]"></div>
          <div className="w-[30px] h-[30px] rounded-full bg-[#FFE694] absolute top-[515px] left-[970px]"></div>
          <div className="absolute top-[510px] left-[1040px]">
            <div className="text-[28px] font-bold">복습하기</div>
            <div className="text-[23px] font-normal text-[#777]">
              배웠던 수어, 게임으로 재밌게 마스터!
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
