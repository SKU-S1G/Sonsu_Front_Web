import { useParams, useLocation } from "react-router-dom";
import { useState, useEffect } from "react";
import axios from "axios";
import { API_URL } from "../../../config";
import { getToken } from "../../../utils/authStorage";
import { FaRegTrashAlt } from "react-icons/fa";

export default function CustomCard({ onDeleteLesson }) {
  const [activeTab, setActiveTab] = useState("초급");
  const [lessons, setLessons] = useState([]);
  const { code: classId } = useParams();

  const { state } = useLocation();
  const name = state?.name;
  const desc = state?.desc;
  const [customLessons, setCustomLessons] = useState([]);
  const [loading, setLoading] = useState(true);

  // 클래스 강의 불러오기
  useEffect(() => {
    const fetchCustomLessons = async () => {
      try {
        const res = await axios.get(`${API_URL}/class/lesson`);
        setCustomLessons(res.data || []);
      } catch (err) {
        console.error("클래스 강의 불러오기 실패:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchCustomLessons();
  }, [activeTab]);

  const tabColors = {
    초급: "#39B360",
    중급: "#487BCD",
    고급: "#FF9381",
  };

  // 탭별 콘텐츠
  const renderContent = () => {
    if (!customLessons || customLessons.length === 0) {
      return <p className="text-gray-500 mt-4">강의가 없습니다.</p>;
    }

    return (
      <div className="h-[550px] overflow-y-auto space-y-4">
        {customLessons.map((lesson) => (
          <div
            key={lesson.lessonCategory_id}
            className="flex p-4 rounded-[20px] transition-all duration-200 hover:shadow-lg cursor-pointer"
          >
            {/* 이미지 */}
            <div className="relative p-4 rounded-[15px] shadow-lg bg-[#F2F2F2]">
              <img
                src="/assets/images/Sign.png"
                alt=""
                className="w-20 h-20 object-cover"
              />
            </div>

            {/* 강의 정보 */}
            <div className="flex flex-col justify-center ml-4 w-[70%]">
              <p className="font-bold text-lg">
                Part {lesson.part_number}. {lesson.category}
              </p>
              <p className="text-sm text-gray-600 truncate w-[150px]">
                {lesson.words?.join(", ")}
              </p>
            </div>

            {/* 삭제 버튼 */}
            <div className="flex items-center justify-end">
              <FaRegTrashAlt
                size={24}
                className="cursor-pointer"
                onClick={() => onDeleteLesson(lesson.lessonCategory_id)}
              />
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="w-[45%] h-[650px] rounded-[40px] p-8 bg-[#DEE6F1] shadow-xl">
      <div className="flex items-center justify-between">
        <p className="text-[#444] text-[18px] font-black">{name}</p>

        {/* 탭 버튼 */}
        <div className="flex items-center">
          {["초급", "중급", "고급"].map((tab, index, arr) => (
            <div key={tab} className="flex items-center">
              <button
                onClick={() => setActiveTab(tab)}
                className={`font-semibold text-[16px] transition-all px-1 py-1 rounded-full`}
                style={{
                  color: activeTab === tab ? tabColors[tab] : "#333",
                }}
              >
                {tab}
              </button>
              {index < arr.length - 1 && (
                <span className="mx-2 text-[#777]">|</span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* 탭 내용 */}
      <div className="mt-6">{renderContent()}</div>
    </div>
  );
}
