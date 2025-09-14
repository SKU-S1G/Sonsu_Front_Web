import React from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import AdminTitle from "../../../components/AdminTitle";
import AdminNav from "../../../components/AdminNav";

export default function ClassMenu() {
  const { code } = useParams();
  const navigate = useNavigate();
  const { state } = useLocation(); // name, desc 가져오기

  const menus = [
    {
      title: "강의 커리큘럼",
      img: "/assets/images/Admin/Main/AdminMain2_1.png",
      path: `/admin/Curri_Part/${code}`,
    },
    {
      title: "수강그룹",
      img: "/assets/images/Admin/Main/AdminMain2_2.png",
      path: `/admin/group/${code}`,
    },
    {
      title: "수강생",
      img: "/assets/images/Admin/Main/AdminMain2_3.png",
      path: `/admin/student/${code}`,
    },
  ];

  return (
    <div className="min-h-screen bg-[#5A9CD0]">
      <AdminTitle />

      <div className="flex w-full">
        <AdminNav />

        <div className="flex flex-col mr-10 w-full rounded-3xl bg-[#fafafa] h-[850px]">
          <div className="flex items-center justify-center h-full">
            {menus.map((menu, idx) => (
              <div
                key={idx}
                className="flex flex-col items-center p-4 mb-20 transition-transform duration-300 ease-in-out cursor-pointer hover:scale-105 hover:brightness-105"
                onClick={() =>
                  navigate(menu.path, {
                    state,
                  })
                }
              >
                <img
                  src={menu.img}
                  alt={menu.title}
                  className="w-[370px] rounded-xl"
                />
                <p className="text-[20px] text-[#333] font-semibold mt-4">
                  {menu.title}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
