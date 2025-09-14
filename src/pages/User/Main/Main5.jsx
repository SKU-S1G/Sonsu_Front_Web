import React from "react";
import { useNavigate } from "react-router-dom";

export default function Main5() {
  const navigate = useNavigate();
    
  const menus = [
    {
      title: "강의 커리큘럼",
      img: "/assets/images/Admin/Main/AdminMain2_1.png",
    },
    {
      title: "그룹 관리",
      img: "/assets/images/Admin/Main/AdminMain2_2.png",
    },
    {
      title: "수강생 관리",
      img: "/assets/images/Admin/Main/AdminMain2_3.png",
    },
  ];

  return (
    <div className="h-screen mt-32">
      {/* 글자 영역 */}
      <div className="flex flex-col items-center justify-center">
        <div className="text-[36px] fontSB text-[#5A9CD0]">SONSU CLASS</div>

        <div className="flex flex-col text-center text-[40px] font-bold mt-10">
          <div>손수클래스로 더 똑똑하게, 자유롭게</div>
          <div>수어를 가르치세요.</div>
        </div>

        <div className="text-[20px] whitespace-pre-line text-center mt-4 text-[#888]">
          프리미엄 요금제에 가입하면{" "}
          <span className="text-[#5A9CD0] font-bold">
            강사와 관리자를 위한 전용 기능
          </span>
          이 열립니다.{`\n`}
        </div>

        <div className="flex items-center justify-center h-full gap-16">
          {menus.map((menu, idx) => (
            <div
              key={idx}
              className="flex flex-col items-center p-4 mb-20 transition-transform duration-300 ease-in-out cursor-pointer hover:scale-105 hover:brightness-105"
            >
              <img
                src={menu.img}
                alt={menu.title}
                className="w-[360px] rounded-xl"
              />
              <p className="text-[28px] text-[#333] font-semibold mt-4">
                {menu.title}
              </p>
            </div>
          ))}
        </div>

        <button
          onClick={() => navigate("/admin")}
          className="px-14 py-3 rounded-full text-[22px] font-semibold text-white
                bg-[#5A9CD0] shadow-lg hover:bg-[#4a8bbd]
                transition-all duration-300 ease-in-out
                hover:shadow-xl hover:scale-105"
        >
          바로가기 →
        </button>
      </div>
    </div>
  );
}
