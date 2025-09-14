import React, { useState, useEffect, useCallback } from "react";
import { getToken } from "../../../utils/authStorage";
import axios from "axios";
import { API_URL } from "../../../config";

const ProgressBar = ({ progress }) => {
  return (
    <div className="w-full mt-2">
      <p className="text-sm font-medium mb-1">진도율 {progress}%</p>
      <div className="w-full h-2 bg-gray-300 rounded overflow-hidden">
        <div
          className="h-full bg-green-500 rounded"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
};

const ContinueLearning = () => {
  const currentLevel = "초급";
  const [nextLesson, setNextLesson] = useState(null);
  const [progress, setProgress] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchProgress = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const token = getToken();
      if (!token) {
        console.log("토큰이 없습니다.");
        return;
      }

      // 다음 수업 정보 가져오기
      const lessonResponse = await axios.get(`${API_URL}/progress/continue`, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        withCredentials: true,
      });

      setNextLesson(lessonResponse.data.nextLesson[0]);

      // 진도율 정보 가져오기
      const progressResponse = await axios.get(
        `${API_URL}/progress/percentage`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          withCredentials: true,
        }
      );

      const progressData = progressResponse.data;
      const progressValue = parseInt(progressData.progress.replace("%", ""));
      setProgress(progressValue);
    } catch (error) {
      console.error("진도율 또는 이어서 학습 정보 불러오기 실패:", error);
      setError("데이터를 불러오는데 실패했습니다.");

      // 오류 시 기본값 설정
      setNextLesson({
        lessonCategory_id: 1,
        word: "안녕하세요",
        animation_path: "",
      });
      setProgress(0);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchProgress();
  }, [fetchProgress]);

  const handleNavigateToClassroom = () => {
    console.log("배움터로 이동");
    window.location.href = "/classroom/easy";
  };

  if (loading) {
    return (
      <div className="my-12 w-[70%]">
        <div className="flex justify-center items-center h-48">
          <div className="text-gray-500">데이터를 불러오는 중...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="my-12 w-[70%]">
        <div className="flex justify-center items-center h-48">
          <div className="text-red-500">
            {error}
            <button
              onClick={fetchProgress}
              className="ml-2 px-3 py-1 bg-blue-500 text-white rounded text-sm"
            >
              다시 시도
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="my-12 w-[70%]">
      {/* 타이틀 */}
      <div className="flex justify-between items-center">
        <p className="text-[20px] font-semibold">이어서 학습하기</p>
      </div>

      {/* 컨텐츠 */}
      <div className="flex justify-around mt-6">
        {/* 왼쪽 (이미지/영상) */}
        <div className="w-[128px] h-[174px] bg-[#CEE9C4] rounded-2xl flex items-center justify-center overflow-hidden">
          {nextLesson?.animation_path ? (
            <video
              src={nextLesson.animation_path}
              autoPlay
              loop
              muted
              className="w-full h-full object-cover"
              onError={(e) => {
                console.error("Video loading failed:", e);
              }}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gray-200">
              <span className="text-gray-500 text-xs text-center px-2">
                {nextLesson?.word || "이미지"}
              </span>
            </div>
          )}
        </div>

        {/* 오른쪽 (정보) */}
        <div className="flex flex-col flex-1 pl-8 justify-between py-4">
          <div>
            <p className="text-[20px] font-bold">
              Part {nextLesson?.lessonCategory_id || 1}.{" "}
              {nextLesson?.word || "로딩 중..."}
            </p>
            <p className="text-[18px] font-light mt-1">{currentLevel}</p>
          </div>

          {/* 진도율 */}
          <ProgressBar progress={progress} />

          {/* 버튼 */}
          <div className="flex justify-center mt-3">
            <button
              onClick={handleNavigateToClassroom}
              className="bg-[#F7EABF] py-2 rounded-full shadow-md w-[60%] text-sm font-medium hover:bg-[#f5e1a3] transition-colors"
            >
              배움터 바로가기
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ContinueLearning;
