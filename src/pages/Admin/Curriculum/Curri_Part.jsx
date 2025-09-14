import { useParams, useLocation, useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";
import AdminTitle from "../../../components/AdminTitle";
import AdminNav from "../../../components/AdminNav";
import SonsuCard from "./SonsuCard";
import CustomCard from "./CustomCard";
import axios from "axios";
import { API_URL } from "../../../config";
import { getToken } from "../../../utils/authStorage";

export default function Curri_Part() {
  const { code: classId } = useParams();
  const { state } = useLocation();
  const name = state?.name;

  const [customLessons, setCustomLessons] = useState([]);

  // SonsuCard → lesson 추가 요청
  const handleAddLesson = async (lessonId) => {
    // 중복 체크
    if (customLessons.includes(lessonId)) {
      alert("이미 추가된 강의입니다.");
      return;
    }

    try {
      const token = getToken();

      const res = await axios.post(
        `${API_URL}/class/${classId}/add`,
        { lessonIds: [lessonId] },
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          withCredentials: true,
        }
      );

      console.log(res.data.message);

      // 성공 시 상태 업데이트
      setCustomLessons((prev) => [...prev, lessonId]);
    } catch (err) {
      console.error("레슨 추가 실패:", err);
      alert("이미 추가된 강의입니다.");
    }
  };

  const handleDeleteLesson = async (lessonId) => {
    if (!customLessons.includes(lessonId)) {
      alert("이미 삭제된 강의입니다.");
      return;
    }
    if (!window.confirm("정말 삭제하시겠습니까?")) return;

    try {
      const token = getToken();

      const res = await axios.delete(`${API_URL}/class/${classId}/delete`, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        withCredentials: true,
        data: { lessonIds: [lessonId] }, // DELETE 시 body는 data로
      });

      alert(res.data.message || "삭제 성공!");

      // 삭제된 강의는 화면에서도 제거
      setCustomLessons((prev) => prev.filter((id) => id !== lessonId));
    } catch (err) {
      console.error("레슨 삭제 실패:", err.response?.data || err);
      alert("레슨 삭제에 실패했습니다.");
    }
  };

  return (
    <div className="min-h-screen bg-[#5A9CD0]">
      <AdminTitle />
      <div className="flex w-full">
        <AdminNav />
        <div className="flex flex-col mr-10 w-full rounded-3xl bg-[#fafafa] h-[850px] px-10 py-8">
          <div>
            <p className="font-semibold text-[25px]">{name}</p>
            <div className="flex justify-between items-center">
              <div className="text-[20px] font-semibold text-[#777]">
                #{classId}
              </div>
              <button className="bg-[#5A9CD0] text-white text-[18px] rounded-full px-4 py-1 font-semibold">
                수정완료
              </button>
            </div>
          </div>

          <div className="flex w-full justify-evenly mt-8">
            {/* Sonsu → 추가 버튼 누르면 handleAddLesson 호출 */}
            <SonsuCard onAddLesson={handleAddLesson} />

            {/* CustomCard → 부모에서 관리하는 customLessons 전달 */}
            <CustomCard
              customLessons={customLessons}
              onDeleteLesson={handleDeleteLesson}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
