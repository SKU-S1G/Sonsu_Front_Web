import React from "react";

export default function Main4() {
  const steps = [
    {
      title: "STEP1",
      desc: "내가 현재 진행중인\n수어 학습을 선택합니다.",
      img: "./assets/images/Main/Step1.png",
      width: 400,
    },
    {
      title: "STEP2",
      desc: "3D아바타의 시연 동작을 보고,\n충분히 따라해보며 익힙니다.",
      img: "./assets/images/Main/Step2.png",
      width: 280,
    },
    {
      title: "STEP3",
      desc: "카메라에 비춘 나의 모습을 보며\n스스로 수어를 익힙니다.",
      img: "./assets/images/Main/Step3.png",
      width: 280,
    },
    {
      title: "STEP4",
      desc: "오늘 배운 수어를\nOX퀴즈, 스피드게임을 통해\n재미있게 복습!",
      img: "./assets/images/Main/Step4.png",
      width: 480,
    },
    {
      title: "STEP5",
      desc: "마이페이지에서 틀렸던 수어만\n골라서 다시 볼 수 있어요!",
      img: "./assets/images/Main/Step5.png",
      width: 280,
    },
    {
      title: "STEP6",
      desc: "주간 랭킹을 통해 나의 수준을 확인하고,\n주간 리포트를 통해 한 주간의 학습을 돌아봅니다.",
      img: "./assets/images/Main/Step6.png",
      width: 280,
    },
  ];

  return (
    <div className="h-screen px-32 mt-20">
      {/* title */}
      <div className="flex flex-col items-end">
        <div className="text-[32px] font-light">학습은 어떻게 진행될까?</div>
        <div className="text-[36px] font-semibold">
          누구나
          <span className="relative inline-block">
            <span className="absolute bottom-1.5 left-0 w-full h-[0.4em] bg-[#FFC400] rounded-full"></span>
            <span className="relative z-10 font-black">쉽고</span>
          </span>
          ,{" "}
          <span className="relative inline-block">
            <span className="absolute bottom-1.5 left-0 w-full h-[0.4em] bg-[#FFC400] rounded-full"></span>
            <span className="relative z-10 font-black">재미있게!</span>
          </span>{" "}
          수어학습, 시작해볼까요?
        </div>
      </div>

      {/* 설명 스크롤 */}
      <div className="relative py-20 overflow-x-auto">
        <div className="absolute top-24 bg-[#F3C11B] w-10 h-10 rounded-full"></div>
        {/* dashed line */}
        <div
          className="absolute h-2 top-28"
          style={{
            width: steps.reduce((acc, s) => acc + s.width + 208, 0), // 각 step 폭 + space-x-52
            backgroundImage:
              "repeating-linear-gradient(to right, #F3C11B, #F3C11B 10px, transparent 10px, transparent 20px)",
          }}
        ></div>

        <div className="flex space-x-52 flex-nowrap">
          {steps.map((step, idx) => (
            <div key={idx} className="flex flex-col items-center flex-shrink-0">
              <div className="bg-[#F3C11B] w-fit px-4 py-2 rounded-md text-white font-bold text-[24px] mt-3 z-10">
                {step.title}
              </div>
              <div className="text-[#838383] whitespace-pre-line text-center text-[18px] h-16 my-8">
                {step.desc}
              </div>
              <img
                src={step.img}
                style={{ width: step.width }}
                alt={step.title}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
