import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { FiSearch } from "react-icons/fi";
import AdminTitle from "../../../components/AdminTitle";
import AdminNav from "../../../components/AdminNav";
import { FaPlus } from "react-icons/fa6";
import axios from "axios";
import { API_URL } from "../../../config";

export default function ClassList() {
  const navigate = useNavigate();

  const [classes, setClasses] = useState([]);
  const [sortType, setSortType] = useState("latest");
  const [search, setSearch] = useState("");

  const colors = [
    "#DEE6F1",
    "#F2F3ED",
    "#F1E3D8",
    "#F4EBCE",
    "#DEE8D0",
    "#F0E4F8",
    "#D9D9D9",
  ];

  useEffect(() => {
    const fetchClasses = async () => {
      try {
        const token = localStorage.getItem("accessToken");

        const res = await axios.get(`${API_URL}/class/selectAll`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        console.log("API_URL:", API_URL);

        const data = res.data.data || [];
        setClasses(data);
      } catch (error) {
        console.error(error);
        alert("클래스 목록 불러오기에 실패했습니다.");
      }
    };

    fetchClasses();
  }, []);

  const filteredClasses = classes
    .filter((cls) =>
      cls.class_name.toLowerCase().includes(search.toLowerCase())
    )
    .sort((a, b) => {
      if (sortType === "name") return a.class_name.localeCompare(b.class_name);
      return 0;
    });

  return (
    <div className="min-h-screen bg-[#5A9CD0]">
      <AdminTitle />

      <div className="flex w-full">
        <AdminNav />

        <div className="flex flex-col mr-10 w-full rounded-3xl bg-[#fafafa] h-[850px]">
          {/* 상단 탭 + 검색 */}
          <div className="flex flex-col items-end mt-8 mb-6 mr-10">
            <div className="flex items-center justify-end w-full mb-6">
              {/* 검색창 */}
              <div className="relative flex">
                <input
                  type="text"
                  placeholder="클래스 검색..."
                  className="bg-[#E8E8E8] rounded-full px-10 py-4 w-full"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
                <FiSearch
                  className="absolute text-gray-600 -translate-y-1/2 right-8 top-1/2"
                  size={23}
                />
              </div>

              <div
                className="flex bg-[#E8E8E8] p-4 rounded-full ml-5 text-gray-600 cursor-pointer hover:bg-gray-300 transition"
                onClick={() => navigate("/admin/group/add")}
              >
                <FaPlus size={23} />
              </div>
            </div>

            {/* 정렬 버튼 */}
            <div className="flex mr-2">
              <button
                className={`px-2 py-2 rounded-lg ${
                  sortType === "latest" ? "font-black" : ""
                }`}
                onClick={() => setSortType("latest")}
              >
                최신순
              </button>
              <button
                className={`px-2 py-2 rounded-lg ${
                  sortType === "name" ? "font-black" : ""
                }`}
                onClick={() => setSortType("name")}
              >
                정렬순
              </button>
            </div>
          </div>

          {/* 클래스 카드 */}
          <div className="flex justify-center mb-10 overflow-auto">
            <div className="grid grid-cols-4 gap-10 mt-5">
              {filteredClasses.map((cls) => {
                // 백엔드에서 colorId가 1부터 시작한다고 가정
                const bg = colors[(cls.colorId || 1) - 1];

                return (
                  <div
                    key={cls.class_code}
                    className="rounded-2xl p-6 shadow-md h-[300px] w-[220px] cursor-pointer hover:scale-105 transition"
                    style={{ backgroundColor: bg }}
                    onClick={() =>
                      navigate(`/admin/ClassMenu/${cls.class_id}`, {
                        state: { name: cls.class_name, desc: cls.description },
                      })
                    }
                  >
                    <h2 className="mb-2 text-xl font-bold">{cls.class_name}</h2>
                    <p className="mb-1 text-sm text-gray-600">
                      # {cls.class_code}
                    </p>
                    <p className="text-gray-700">{cls.description}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
