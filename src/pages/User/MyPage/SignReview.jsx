import React, { useState, useEffect, useRef } from "react";
import { API_URL } from "../../../config";
import { getToken } from "../../../utils/authStorage";
import classData from "../../../utils/ClassData";
import { FaRegBookmark, FaBookmark } from "react-icons/fa";

export default function SignReview({ isActive, setIsActive }) {
  const [selectedLevel, setSelectedLevel] = useState("초급");
  const [savedSigns, setSavedSigns] = useState([]);
  const [bookmarkedTopics, setBookmarkedTopics] = useState([]);
  const [loading, setLoading] = useState(false);
  const ref = useRef(null);

  // 바깥 클릭 감지
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (ref.current && !ref.current.contains(event.target)) {
        setIsActive(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [setIsActive]);

  // 즐겨찾기 데이터 가져오기
  useEffect(() => {
    const fetchSavedSigns = async () => {
      if (!isActive) return;

      setLoading(true);
      try {
        const accessToken = getToken();
        if (!accessToken) {
          console.log("토큰이 없습니다");
          return;
        }

        const response = await fetch(`${API_URL}/review/lessons`, {
          headers: {
            Authorization: `Bearer ${accessToken}`,
          },
        });

        if (!response.ok) {
          throw new Error("데이터를 불러오지 못했습니다");
        }

        const data = await response.json();
        setSavedSigns(data);
        setBookmarkedTopics(data.map((item) => item.lesson_id));
      } catch (error) {
        console.error("즐겨찾기 데이터를 불러오지 못했습니다:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchSavedSigns();
  }, [isActive]);

  // 레벨별 필터링
  const filteredSigns = savedSigns.filter((item) => {
    if (selectedLevel === "초급") {
      return item.lesson_id <= 32;
    } else if (selectedLevel === "중급") {
      return item.lesson_id >= 33 && item.lesson_id <= 49;
    } else if (selectedLevel === "고급") {
      return item.lesson_id >= 50;
    }
    return false;
  });

  const levelColors = {
    초급: "#39B360",
    중급: "#487BCD",
    고급: "#FF9381",
  };

  // 북마크 토글
  const handleBookmark = async (topicId) => {
    try {
      const accessToken = getToken();
      if (!accessToken) {
        alert("로그인이 필요합니다.");
        return;
      }

      if (bookmarkedTopics.includes(topicId)) {
        // 북마크 삭제
        const response = await fetch(`${API_URL}/review/delete/${topicId}`, {
          method: "DELETE",
          headers: {
            Authorization: `Bearer ${accessToken}`,
          },
        });

        if (response.ok) {
          setBookmarkedTopics((prev) => prev.filter((id) => id !== topicId));
          setSavedSigns((prev) =>
            prev.filter((item) => item.lesson_id !== topicId)
          );
        }
      } else {
        // 북마크 추가
        const response = await fetch(`${API_URL}/review/save`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${accessToken}`,
          },
          body: JSON.stringify({ lessonId: topicId }),
        });

        if (response.ok) {
          setBookmarkedTopics((prev) => [...prev, topicId]);
          // 데이터 새로고침
          const updatedResponse = await fetch(`${API_URL}/review/lessons`, {
            headers: {
              Authorization: `Bearer ${accessToken}`,
            },
          });
          if (updatedResponse.ok) {
            const updatedData = await updatedResponse.json();
            setSavedSigns(updatedData);
          }
        }
      }
    } catch (error) {
      console.error("북마크 처리 중 오류:", error);
      alert("오류가 발생했습니다.");
    }
  };

  return (
    <div className="relative" ref={ref}>
      <div
        onClick={() => setIsActive(!isActive)}
        className={`bg-[#FFEEB8] rounded-[20px] pl-8 pt-6 w-full shadow-lg h-[200px]
             transform transition duration-300 ease-in-out
             cursor-pointer
             ${isActive ? "scale-110 shadow-2xl" : "scale-100 shadow-lg"}
             ${isActive ? "filter brightness-105" : ""}`}
      >
        <p className="fontSB text-[24px]">수어 즐겨찾기</p>
        <div className="fontSB text-[12px] text-[#555] mt-3">
          <p>저장한 수어로 언제든지 복습하고,</p>
          <p>학습한 내용을 되돌아보세요.</p>
        </div>

        <div className="flex justify-end -mt-4">
          <img
            src="/assets/images/MyPage/bookmark.png"
            alt=""
            className="w-[40%] opacity-80"
          />
        </div>
      </div>

      {isActive && (
        <div className="absolute -left-10 -top-[390px] mt-0 -translate-x-full w-[500px] p-6 bg-white rounded-xl shadow-2xl z-50 h-[600px] overflow-y-auto">
          <div className="space-y-6 h-full flex flex-col">
            {/* 헤더 */}
            <div className="text-center flex-shrink-0">
              <h3 className="text-xl font-bold text-gray-800">수어 즐겨찾기</h3>
              <p className="text-sm text-gray-600 mt-1">
                저장한 수어로 언제든지 복습하고, 학습한 내용을 되돌아보세요.
              </p>
            </div>

            {/* 레벨 선택 버튼 */}
            <div className="flex space-x-4">
              {["초급", "중급", "고급"].map((level) => (
                <button
                  key={level}
                  onClick={() => setSelectedLevel(level)}
                  className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                    selectedLevel === level
                      ? "text-white"
                      : "text-gray-600 bg-gray-100"
                  }`}
                  style={{
                    backgroundColor:
                      selectedLevel === level ? levelColors[level] : "",
                  }}
                >
                  {level}
                </button>
              ))}
            </div>

            {loading ? (
              <div className="flex-1 flex items-center justify-center">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-yellow-500 mx-auto"></div>
                  <span className="block mt-4 text-gray-600">
                    즐겨찾기를 불러오는 중...
                  </span>
                </div>
              </div>
            ) : (
              <div className="flex-1 space-y-4 overflow-y-auto">
                {filteredSigns.length === 0 ? (
                  <div className="flex-1 flex items-center justify-center">
                    <p className="text-gray-500 text-center">
                      저장된 수어가 없습니다.
                    </p>
                  </div>
                ) : (
                  filteredSigns.map((item) => {
                    const lesson = classData[selectedLevel]?.find(
                      (lesson) => lesson.id === item.lesson_id
                    );

                    return (
                      <div
                        key={item.userSaved_id}
                        className="bg-gray-50 rounded-lg p-4 border border-gray-200 hover:shadow-md transition-shadow relative"
                      >
                        {/* 북마크 버튼 */}
                        {/* 북마크 버튼 */}
                        <button
                          onClick={() => handleBookmark(item.lesson_id)}
                          className="absolute top-6 right-6 z-10 hover:scale-110 transition-transform"
                        >
                          {bookmarkedTopics.includes(item.lesson_id) ? (
                            <FaBookmark size={28} color="#FFCA1A" />
                          ) : (
                            <FaRegBookmark size={28} color="#FFCA1A" />
                          )}
                        </button>

                        {/* 비디오 영역 */}
                        <div className="w-full h-52 bg-gray-200 rounded-lg mb-3 flex items-center justify-center">
                          {item.animation_path ? (
                            <video
                              src={item.animation_path}
                              autoPlay
                              muted
                              loop
                              className="w-full h-full object-cover rounded-lg"
                            />
                          ) : (
                            <div className="text-gray-400">
                              <svg
                                className="w-12 h-12"
                                fill="currentColor"
                                viewBox="0 0 20 20"
                              >
                                <path d="M8 5v10l6-5-6-5z" />
                              </svg>
                            </div>
                          )}
                        </div>

                        {/* 텍스트 정보 */}
                        <div>
                          <p className="text-xs text-gray-500">
                            Part {item.lessonCategory_id}
                          </p>
                          <p className="text-lg font-semibold text-gray-800 mt-1">
                            {item.word}
                          </p>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            )}

            {/* 안내 메시지 */}
            <div className="text-center text-xs text-gray-500 pt-2 border-t border-gray-200 flex-shrink-0">
              ⭐ 별표를 클릭하여 즐겨찾기를 추가하거나 제거할 수 있습니다
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
