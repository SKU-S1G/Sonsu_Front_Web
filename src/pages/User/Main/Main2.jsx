import React from "react";

export default function Main2() {
  return (
    <div className="">
      {/* 텍스트 */}
      <div className="text-[#404040] text-[25px] leading-10 w-full bg-[#FFE694] h-[300px] flex flex-col items-center justify-center mt-10">
        <p>이제껏 경험하지 못한 쉽고 즐거운 수어 교육 서비스,</p>
        <p>
          <span className="text-[#000] font-black text-[30px] mr-1">
            손수잇다
          </span>
          와 함께라면 당신의 배움이 달라질거예요.
        </p>
      </div>

      {/* 말풍선 */}
      <div className="relative mt-32 min-h-[800px] w-[80%] mx-auto">
        {/* #게임처럼_배우는_수어학습 */}
        <div className="absolute top-0 left-1/4 -translate-x-1/2">
          <div className="shadow-xl w-fit px-20 py-10 rounded-tl-[6rem] rounded-tr-[6rem] rounded-br-[6rem] bg-gradient-to-b from-white/70 to-white/100">
            <div className="text-[#DFAB00] font-bold text-[28px]">
              #게임처럼_배우는_수어학습
            </div>
            <div className="text-[18px] font-light whitespace-pre-line mt-3">
              수화를 어렵게 느낄 수 있는 사람들도 재밌는 퀴즈 형식으로
              자연스럽게{"\n"}
              접근할 수 있도록 구성한 점이 인상 깊었습니다.
            </div>
          </div>
        </div>

        {/* #누구나_쉽게_배울수있는_수어 */}
        <div className="absolute top-[230px] left-2/3 -translate-x-1/3 w-[68%]">
          <div className="shadow-xl w-fit px-20 py-10 rounded-tl-[6rem] rounded-tr-[6rem] rounded-bl-[6rem] bg-gradient-to-b from-white/70 to-white/100">
            <div className="text-[#DFAB00] font-bold text-[28px]">
              #누구나_쉽게_배울수있는_수어
            </div>
            <div className="text-[18px] font-light whitespace-pre-line mt-3">
              다양한 방식을 활용하여 배우는 사람에게 하여금 더 쉽게 기억에
              남도록 하는 점이{"\n"}
              정말 수어를 배우고자 하는 사람에게 도움을 주고 싶다는 의지가
              느껴지는 것 같습니다.
            </div>
          </div>
        </div>

        {/* #3D아바타로_직관적_학습 */}
        <div className="absolute top-[480px] left-[220px] z-12">
          <div className="shadow-xl w-fit px-20 py-10 rounded-tl-[6rem] rounded-tr-[6rem] rounded-br-[6rem] bg-gradient-to-b from-white/70 to-white/100">
            <div className="text-[#DFAB00] font-bold text-[28px]">
              #3D아바타로_직관적_학습
            </div>
            <div className="text-[18px] font-light whitespace-pre-line mt-3">
              3D 캐릭터와 게임 요소를 활용해 수어 학습을 흥미롭고 친근하게 만든
              점이 인상 깊었습니다.{"\n"}앱 디자인도 깔끔하고 직관적이어서
              실제로 출시된다면 많은 사람들이 즐겁게 활용할 수 있을 것 같습니다.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
