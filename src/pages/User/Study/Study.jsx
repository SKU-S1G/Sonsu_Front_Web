import React, { useEffect, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import UserTitle from "../../../components/UserTitle";
import UserNav from "../../../components/UserNav";
import { API_URL } from "../../../config";
import { serverIP } from "../../../config";
import { getToken } from "../../../utils/authStorage";
import { FaRegBookmark, FaBookmark } from "react-icons/fa";
import axios from "axios";

export default function Study() {
  const location = useLocation();
  const navigate = useNavigate();
  const { lessonId, level, partId } = useParams();
  const [lessons, setLessons] = useState([]);
  const [error, setError] = useState(null);
  const [videoSrc, setVideoSrc] = useState(`${serverIP}/video_feed`);
  const [isVideoPlaying, setIsVideoPlaying] = useState(true);
  const [bookmarkedSteps, setBookmarkedSteps] = useState([]);

  // location.state에서 topic, lesson, index 받기
  const { topic, lesson, index } = location.state || {};

  const [animation, setAnimation] = useState("");
  const [stepData, setStepData] = useState(null);
  const [allStepsInPart, setAllStepsInPart] = useState([]);

  console.log("레슨아이디", lesson);

  const levelId = { easy: 1, normal: 2, hard: 3 };

  // 현재 파트의 모든 스텝 가져오기
  const fetchAllStepsInPart = async () => {
    try {
      const currentPartId = lesson?.id || partId;
      if (!currentPartId) return;

      const response = await fetch(
        `${API_URL}/lessons/${currentPartId}/topics`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        setAllStepsInPart(data);
        console.log("현재 파트의 모든 스텝:", data);
      }
    } catch (error) {
      console.log("파트 스텝 불러오기 실패:", error.message);
    }
  };

  // 토픽 데이터 가져오기
  const fetchTopic = async () => {
    try {
      const response = await fetch(
        `${API_URL}/lessons/${lesson?.id || lessonId}/topics`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        console.log("토픽 데이터", data);

        // topic이 있으면 해당 topic 찾기, 없으면 lessonId로 찾기
        const topicData = topic
          ? data.find((t) => t.word === topic.word)
          : data.find((t) => t.lesson_id === parseInt(lessonId));

        if (topicData) {
          const animationPath = topicData.animation_path;
          console.log("서버에서 전달된 URL:", animationPath);
          setAnimation(animationPath);
          setStepData(topicData);
        }
      }
    } catch (error) {
      console.log("애니메이션 불러오기 실패:", error.message);
    }
  };

  // 강의 시작하기
  const startLesson = async () => {
    try {
      const targetLessonId = topic?.lesson_id || lessonId;
      const token = getToken();

      const response = await fetch(`${API_URL}/lessons/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token && { Authorization: `Bearer ${token}` }),
        },
        credentials: "include",
        body: JSON.stringify({ lessonId: targetLessonId }),
      });

      if (response.status === 200) {
        const data = await response.json();
        console.log("강의가 시작되었습니다", data);
      } else if (response.status === 401) {
        console.log("로그인을 해주세요.");
      } else if (response.status === 403) {
        console.log("접근이 거부되었습니다. 토큰이 유효하지 않습니다.");
      }
    } catch (error) {
      console.log("에러 발생:", error.message);
    }
  };

  useEffect(() => {
    fetchTopic();
    startLesson();
    fetchAllStepsInPart();
  }, [topic, lessonId]);

  // 북마크 토글 함수
  const handleBookmark = async (stepId) => {
    try {
      const token = getToken();
      if (!token) {
        alert("로그인이 필요합니다.");
        return;
      }

      if (bookmarkedSteps.includes(stepId)) {
        // 북마크 삭제
        const res = await fetch(`${API_URL}/review/delete/${stepId}`, {
          method: "DELETE",
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await res.json();
        setBookmarkedSteps((prev) => prev.filter((id) => id !== stepId));
        alert(data.message);
      } else {
        // 북마크 추가
        const res = await fetch(`${API_URL}/review/save`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ lessonId: stepId }),
        });
        const data = await res.json();
        setBookmarkedSteps((prev) => [...prev, stepId]);
        alert(data.message);
      }
    } catch (err) {
      console.error("북마크 처리 실패:", err);
      alert("북마크 처리 실패");
    }
  };

  // 북마크 상태 초기화
  useEffect(() => {
    const fetchBookmarks = async () => {
      try {
        const token = getToken();
        if (!token) return;

        const res = await fetch(`${API_URL}/review/lessons`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await res.json();
        setBookmarkedSteps(data.map((item) => item.lesson_id));
      } catch (err) {
        console.error("북마크 불러오기 실패:", err);
      }
    };

    fetchBookmarks();
  }, []);

  // 파트의 모든 스텝이 완료되었는지 확인
  const checkPartCompletion = async () => {
    try {
      const token = getToken();
      if (!token) return false;

      // 현재 파트의 완료된 스텝들 가져오기
      const res = await axios.post(
        `${API_URL}/progress/topics`,
        {},
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
          withCredentials: true,
        }
      );

      const completedTopics = res.data || [];

      // 현재 파트의 모든 스텝이 완료되었는지 확인
      const allStepsCompleted = allStepsInPart.every((step) =>
        completedTopics.some(
          (completed) =>
            completed.lesson_id === step.lesson_id &&
            completed.status === "completed"
        )
      );

      console.log("파트 완료 상태:", allStepsCompleted);
      return allStepsCompleted;
    } catch (error) {
      console.error("파트 완료 확인 실패:", error);
      return false;
    }
  };

  // 다음 스텝으로 이동
  const goToNextStep = () => {
    if (!allStepsInPart.length) return;

    const currentStepIndex = allStepsInPart.findIndex(
      (step) => step.lesson_id === currentTopic?.lesson_id
    );

    if (currentStepIndex < allStepsInPart.length - 1) {
      // 같은 파트 내의 다음 스텝으로 이동
      const nextStep = allStepsInPart[currentStepIndex + 1];
      navigate(`/study/${nextStep.lesson_id}`, {
        state: {
          topic: nextStep,
          lesson: lesson,
          index: currentStepIndex + 1,
        },
      });
    } else {
      // 파트의 마지막 스텝인 경우 - 파트 완료 후 classroom으로 이동
      alert("이 파트의 모든 스텝을 완료했습니다!");
      navigate(`/classroom/${level || "easy"}`);
    }
  };

  // 현재 topic 또는 stepData에서 표시할 데이터 결정
  const currentTopic = topic || stepData;
  const displayTitle = currentTopic
    ? `Step ${currentTopic.step_number || index + 1}. ${currentTopic.word}`
    : `Step ${lessonId}`;

  // 학습완료
  const completeLesson = async () => {
    try {
      const token = getToken();
      if (!token) {
        alert("로그인이 필요합니다.");
        return;
      }

      const response = await axios.put(
        `${API_URL}/lessons/complete`,
        { lessonId: currentTopic.lesson_id },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
          withCredentials: true,
        }
      );

      if (response.status === 200) {
        console.log("수강 완료:", response.data.message);

        // 파트 완료 상태 확인
        const isPartCompleted = await checkPartCompletion();

        if (isPartCompleted) {
          alert("파트를 모두 완료했습니다! 다음 파트가 해제되었습니다.");
          navigate(`/classroom/${level || "easy"}`);
        } else {
          alert("스텝 완료! 다음 스텝을 진행해주세요.");
          // 파트가 완료되지 않았으면 classroom으로 이동하여 다음 스텝 선택
          navigate(`/classroom/${level || "easy"}/${lesson?.id || partId}`);
        }
      }
    } catch (error) {
      console.error("완료 요청 중 에러 발생:", error.message);
      alert("수강 완료 처리에 실패했습니다.");
    }
  };

  return (
    <div className="min-h-screen bg-[#FFE694]">
      <UserTitle />
      <div className="flex w-full">
        <UserNav />
        <div className="flex flex-col mr-10 w-full rounded-[40px] bg-[#f5f5f5] shadow-xl h-[850px] px-12 py-9 overflow-y-auto">
          <div className="flex justify-center space-x-20">
            <div className="text-left ">
              <div className="flex items-center mt-[33px]">
                <h1 className="text-[30px] font-bold  text-[#333]">
                  {displayTitle}
                </h1>
                {currentTopic && (
                  <div
                    className="ml-6 cursor-pointer"
                    onClick={(e) => {
                      handleBookmark(currentTopic.lesson_id);
                    }}
                  >
                    {bookmarkedSteps.includes(currentTopic.lesson_id) ? (
                      <FaBookmark size={24} color="#333" />
                    ) : (
                      <FaRegBookmark size={24} />
                    )}
                  </div>
                )}
              </div>

              {/* 설명 텍스트 */}
              <div className="flex justify-center mt-12">
                <div className="">
                  <p className="text-[14px] py-[3px] m-0 text-[#333] leading-[1.5]">
                    1. 오른 손바닥으로 왼 팔등을 스쳐내리세요.
                  </p>
                  <p className="text-[14px] py-[3px] m-0 text-[#333] leading-[1.5]">
                    2. 두 주먹을 쥐고 바닥이 아래로 향하게 하여 가슴 앞에서
                    아래로 내려요
                  </p>
                </div>
              </div>
            </div>

            {/* 비디오 플레이어 */}
            {(currentTopic?.animation_path || animation) && (
              <div className="flex justify-end">
                <video
                  src={currentTopic?.animation_path || animation}
                  loop
                  autoPlay
                  muted
                  className="w-[55%] max-w-[600px] h-auto rounded-[10px] shadow-md"
                  style={{ aspectRatio: "16/9" }}
                  onError={(e) => console.log("Video error:", e)}
                >
                  브라우저가 비디오를 지원하지 않습니다.
                </video>
              </div>
            )}
          </div>

          {/* 카메라 피드 */}
          <div className="flex justify-center h-full">
            {error ? (
              <div className="mt-4 text-red-500">{error}</div>
            ) : (
              <div className="mt-4">
                {isVideoPlaying && (
                  <img
                    src={videoSrc}
                    alt="Video Stream"
                    className="w-[1100px] h-[500px] shadow-lg rounded-3xl"
                  />
                )}
              </div>
            )}
          </div>

          {/* 버튼들 */}
          <div className="flex justify-end space-x-5">
            <button
              onClick={goToNextStep}
              className="bg-[#FFE694] py-[7px] px-[30px] rounded-[8px] border-none cursor-pointer transition-all duration-200 shadow-[1px_3px_5px_rgba(0,0,0,0.3)] hover:transform hover:-translate-y-[2px] hover:shadow-[1px_5px_8px_rgba(0,0,0,0.4)]"
            >
              <span className="text-[21px] font-bold text-[#333] text-center">
                다음 스텝
              </span>
            </button>
            <button
              onClick={completeLesson}
              className="bg-[#FFE694] py-[7px] px-[30px] rounded-[8px] border-none cursor-pointer transition-all duration-200 shadow-[1px_3px_5px_rgba(0,0,0,0.3)] hover:transform hover:-translate-y-[2px] hover:shadow-[1px_5px_8px_rgba(0,0,0,0.4)]"
            >
              <span className="text-[21px] font-bold text-[#333] text-center">
                학습 완료
              </span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
