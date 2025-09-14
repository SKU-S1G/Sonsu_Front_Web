import React, { useState } from "react";
import AdminTitle from "../../../components/AdminTitle";
import AdminNav from "../../../components/AdminNav";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { API_URL } from '../../../config';

const colors = [
  "#DEE6F1",
  "#F2F3ED",
  "#F1E3D8",
  "#F4EBCE",
  "#DEE8D0",
  "#F0E4F8",
  "#D9D9D9",
];

export default function AdminGroupAdd() {
  const [groupName, setGroupName] = useState("");
  const [description, setDescription] = useState("");
  const [selectedColor, setSelectedColor] = useState("");
  const navigate = useNavigate();

  // 색상 id 매핑 (백엔드에서 colorId 필요)
  const getColorId = (color) => colors.indexOf(color) + 1;

  const handleSubmit = async () => {
    if (!groupName || !selectedColor) {
      alert("필수 입력값을 채워주세요.");
      return;
    }

    try {
      const token = localStorage.getItem("accessToken"); 

      const body = {
        className: groupName,
        description: description,
        colorId: getColorId(selectedColor),
      };

      const res = await axios.post(
        `${API_URL}/class/generate`,
        body,
        {
          headers: {
            Authorization: `Bearer ${token}`, 
          },
        }
      );

      alert(res.data.message);
      navigate("/admin/classList");
    } catch (error) {
      console.error(error);
      alert("그룹 생성에 실패했습니다.");
    }
  };


  return (
    <div className="min-h-screen bg-[#5A9CD0]">
      <AdminTitle />

      <div className="flex w-full">
        <AdminNav />

        <div className="flex mr-10 w-full justify-center rounded-3xl bg-[#fafafa] h-[950px]">
          <div className="flex flex-col w-[80%] justify-center">
            <div className="flex text-[23px] text-[#333] fontSB">그룹 생성하기</div>

            <div className="flex justify-center">
              <div className="flex flex-col justify-center w-[55%]">
                <div className="flex justify-end text-red-500 text-[13px] fontMedium mt-12">
                  필수 *
                </div>

                <div className="flex text-[18px] text-[#333] fontMedium items-center">
                  그룹 이름
                  <span className="ml-1 text-red-500">*</span>
                </div>
                <input
                  type="text"
                  placeholder="손수잇다"
                  className="bg-transparent px-4 py-3 mt-5 border-[1.5px] border-[#555552] rounded-xl focus:outline-none focus:border-[#5A9CD0]"
                  value={groupName}
                  onChange={(e) => setGroupName(e.target.value)}
                />

                <div className="flex text-[18px] text-[#333] fontMedium mt-14 items-center">
                  부가 설명
                </div>
                <input
                  type="text"
                  placeholder="안녕하세요~"
                  className="bg-transparent px-4 py-3 mt-5 border-[1.5px] border-[#555552] rounded-xl focus:outline-none focus:border-[#5A9CD0]"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                />

                <div className="flex text-[18px] text-[#333] fontMedium mt-14 items-center">
                  그룹 색상
                  <span className="ml-1 text-red-500">*</span>
                </div>
                <div className="flex flex-wrap justify-center gap-8 mt-5">
                  {colors.map((color) => (
                    <div
                      key={color}
                      onClick={() => setSelectedColor(color)}
                      className="w-[40px] h-[40px] rounded-full cursor-pointer"
                      style={{
                        backgroundColor: color,
                        border:
                          selectedColor === color ? "3px solid #5A9CD0" : "none",
                      }}
                    />
                  ))}
                </div>

                <div className="flex items-center mt-14">
                  <div className="flex text-[18px] text-[#333] fontMedium">
                    그룹 코드
                  </div>
                  <div className="flex ml-8 text-[#666]">
                    그룹 코드는 그룹 생성 후 자동으로 생성됩니다.
                  </div>
                </div>
              </div>
            </div>

            <div className="flex justify-end mt-12">
              <div
                className="flex text-[20px] text-[#555552] fontSB cursor-pointer"
                onClick={() => navigate("/admin/classList")}
              >
                취소
              </div>
              <div
                className="flex text-[20px] ml-5 text-[#5A9CD0] fontSB cursor-pointer"
                onClick={handleSubmit}
              >
                만들기
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
