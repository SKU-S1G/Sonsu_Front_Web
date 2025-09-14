import { useParams, useLocation, useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";
import AdminTitle from "../../../components/AdminTitle";
import AdminNav from "../../../components/AdminNav";
import axios from "axios";
import { API_URL } from "../../../config";
import { getToken } from "../../../utils/authStorage";
import { RxPlus } from "react-icons/rx";

export default function SonsuCard({ onAddLesson }) {
  const [activeTab, setActiveTab] = useState("초급");
  const [lessons, setLessons] = useState([]);

  const { code } = useParams();
  const { state } = useLocation();
  const name = state?.name;
  const desc = state?.desc;

  const levelBgColor = {
    easy: "#D7EDD5",
    normal: "#CBD3DF",
    hard: "#ECD7D4",
  };

  useEffect(() => {
    const fetchLessons = async () => {
      try {
        // activeTab에 따른 levelId
        const tabLevelId = { 초급: 1, 중급: 2, 고급: 3 };
        const res = await axios.get(
          `${API_URL}/lessons/${tabLevelId[activeTab]}/categories`
        );
        setLessons(res.data.categoriesWithWord || []);
        console.log(activeTab, res.data);
      } catch (err) {
        console.error(`${activeTab} 강의 불러오기 실패:`, err);
      }
    };

    fetchLessons();
  }, [activeTab]);

  const tabToLevelKey = {
    초급: "easy",
    중급: "normal",
    고급: "hard",
  };

  // 탭별 콘텐츠
  const renderContent = () => {
    if (!lessons || lessons.length === 0) {
      return <p className="text-gray-500 mt-4">강의가 없습니다.</p>;
    }

    return (
      <div className="h-[550px] overflow-y-auto space-y-4">
        {lessons.map((lesson) => (
          <div
            key={lesson.lessonCategory_id}
            className="flex p-4 rounded-[20px] transition-all duration-200 hover:shadow-lg cursor-pointer"
            // onClick={() => handleLessonClick(lesson)} // 필요하면 클릭 이벤트 추가
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
            <div
              className="flex items-center justify-end"
              onClick={() => onAddLesson(lesson.lessonCategory_id)}
            >
              <RxPlus size={24} />
            </div>
          </div>
        ))}
      </div>
    );
  };

  // 각 탭별 색상 매핑
  const tabColors = {
    초급: "#39B360",
    중급: "#487BCD",
    고급: "#FF9381",
  };

  return (
    <div
      className="w-[45%] h-[650px] rounded-[40px] p-8 shadow-xl"
      style={{
        backgroundColor: levelBgColor[tabToLevelKey[activeTab]],
      }}
    >
      <div className="flex items-center justify-between">
        <p className="text-[#DBBF63] text-[18px] font-black">SONSU</p>

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
              {/* 구분자 | (마지막 요소 제외) */}
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
