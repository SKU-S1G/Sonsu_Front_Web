import React, { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate, useParams } from "react-router-dom";
import UserNav from "../../../components/UserNav";
import UserTitle from "../../../components/UserTitle";
import { API_URL } from "../../../config";
import { getToken } from "../../../utils/authStorage";
import { FiLock } from "react-icons/fi";
import { FaRegBookmark, FaBookmark } from "react-icons/fa";

export default function ClassroomDetail() {
  const { level, partId } = useParams();
  const navigate = useNavigate();
  const [steps, setSteps] = useState([]);
  const [progress, setProgress] = useState([]);
  const [bookmarkedSteps, setBookmarkedSteps] = useState([]);

  // 레벨별 배경색
  const levelBgColor = {
    easy: "#D7EDD5",
    normal: "#CBD3DF",
    hard: "#ECD7D4",
  };

  useEffect(() => {
    // 해당 파트의 step 불러오기
    const fetchSteps = async () => {
      try {
        const res = await axios.get(`${API_URL}/lessons/${partId}/topics`, {
          withCredentials: true,
        });
        setSteps(res.data);
      } catch (err) {
        console.error("Step 불러오기 실패:", err);
      }
    };

    // 유저 진도 불러오기 (앱과 동일한 방식)
    const fetchProgress = async () => {
      try {
        const token = getToken();

        if (!token) {
          console.log("토큰이 없습니다.");
          return;
        }

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
        setProgress(res.data || []);
      } catch (err) {
        console.error("진도 불러오기 실패:", err);
      }
    };

    const fetchBookmarks = async () => {
      try {
        const token = await getToken();
        if (!token) return;

        const res = await axios.get(`${API_URL}/review/lessons`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        setBookmarkedSteps(res.data.map((item) => item.lesson_id));
      } catch (err) {
        console.error("북마크 불러오기 실패:", err);
      }
    };

    fetchBookmarks();
    fetchSteps();
    fetchProgress();
  }, [partId]);

  const handleBookmark = async (stepId) => {
    try {
      const token = await getToken();
      if (!token) {
        alert("로그인이 필요합니다.");
        return;
      }

      if (bookmarkedSteps.includes(stepId)) {
        // 삭제
        const res = await axios.delete(`${API_URL}/review/delete/${stepId}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        setBookmarkedSteps((prev) => prev.filter((id) => id !== stepId));
        alert(res.data.message);
      } else {
        // 추가
        const res = await axios.post(
          `${API_URL}/review/save`,
          { lessonId: stepId },
          { headers: { Authorization: `Bearer ${token}` } }
        );
        setBookmarkedSteps((prev) => [...prev, stepId]);
        alert(res.data.message);
      }
    } catch (err) {
      console.error("북마크 처리 실패:", err);
      alert("북마크 처리 실패");
    }
  };

  // 앱과 동일한 잠금 로직 (단순화 버전)
  const isStepLocked = (step, index) => {
    if (index === 0) return false; // 첫 번째 step은 항상 열림

    // 이전 step
    const prevStep = steps[index - 1];

    // 이전 step이 완료되었는지 확인
    const prevCompleted = progress.some(
      (p) => p.lesson_id === prevStep.lesson_id && p.status === "completed"
    );

    return !prevCompleted; // 이전 step 완료 안됐으면 잠금
  };

  // step이 완료되었는지 확인
  const isStepCompleted = (step) => {
    return progress.some(
      (p) => p.lesson_id === step.lesson_id && p.status === "completed"
    );
  };

  const handleStepClick = (step, index) => {
    if (!isStepLocked(step, index)) {
      navigate(`/study/${step.lesson_id}`, {
        state: {
          topic: step,
          lesson: { id: partId }, // 현재 파트 ID
          index: index,
        },
      });
    } else {
      alert("이 강의는 이전 강의를 완료한 후 이용할 수 있습니다.");
    }
  };

  return (
    <div className="min-h-screen bg-[#FFE694]">
      <UserTitle />
      <div className="flex w-full">
        <UserNav />
        <div className="flex flex-col mr-10 w-full rounded-[40px] bg-[#fafafa] shadow-xl h-[850px] px-12 py-9 overflow-y-auto">
          <div className="mb-6">
            <p className="text-[#222] font-bold text-[25px]">손수잇다</p>
            <p className="text-[#777] font-semibold text-[20px]">#12345</p>
          </div>

          <div
            className={`px-10 py-8 rounded-[40px] h-[85%] overflow-y-auto`}
            style={{ backgroundColor: levelBgColor[level] }}
          >
            <div className="grid grid-cols-2 gap-6">
              {steps.map((step, index) => {
                const isLocked = isStepLocked(step, index);
                const isCompleted = isStepCompleted(step);

                return (
                  <div
                    key={step.lesson_id}
                    className={`flex p-4 rounded-[20px] transition-all duration-200 ${
                      isLocked
                        ? "cursor-not-allowed"
                        : "cursor-pointer hover:shadow-lg transform hover:-translate-y-1"
                    }`}
                    onClick={() => handleStepClick(step, index)}
                  >
                    <div className="relative p-4 rounded-[15px] shadow-lg bg-[#F2F2F2]">
                      <img
                        src="/assets/images/Sign.png"
                        alt=""
                        className="w-20 h-20"
                      />

                      {/* 잠금 오버레이 */}
                      {isLocked && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 rounded-[15px]">
                          <FiLock className="text-white" size={32} />
                        </div>
                      )}

                      {/* 완료 표시 */}
                      {isCompleted && (
                        <div className="absolute -top-2 -right-2 bg-green-500 rounded-full p-1">
                          <svg
                            className="w-4 h-4 text-white"
                            fill="currentColor"
                            viewBox="0 0 20 20"
                          >
                            <path
                              fillRule="evenodd"
                              d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                              clipRule="evenodd"
                            />
                          </svg>
                        </div>
                      )}
                    </div>

                    <div className="flex items-center justify-center ml-4">
                      <p className={`font-bold text-lg`}>
                        Step {step.step_number}. {step.word}
                      </p>
                      {!isLocked && (
                        <div
                          onClick={(e) => {
                            e.stopPropagation(); // 이벤트 버블링 막기
                            handleBookmark(step.lesson_id);
                          }}
                          className="ml-8"
                        >
                          {bookmarkedSteps.includes(step.lesson_id) ? (
                            <FaBookmark size={20} color="#333" />
                          ) : (
                            <FaRegBookmark size={20} />
                          )}
                        </div>
                      )}
                    </div>
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
