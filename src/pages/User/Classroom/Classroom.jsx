import React, { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate, useParams } from "react-router-dom";
import UserNav from "../../../components/UserNav";
import UserTitle from "../../../components/UserTitle";
import { API_URL } from "../../../config";
import { getToken } from "../../../utils/authStorage";
import { FiLock } from "react-icons/fi";

export default function Classroom() {
  const { level } = useParams();
  const navigate = useNavigate();
  const [lessons, setLessons] = useState([]);
  const [completedTopics, setCompletedTopics] = useState([]);
  const [nextLesson, setNextLesson] = useState(null);

  const levelBgColor = {
    easy: "#D7EDD5",
    normal: "#CBD3DF",
    hard: "#ECD7D4",
  };

  const levelId = { easy: 1, normal: 2, hard: 3 };

  useEffect(() => {
    const fetchLessons = async () => {
      try {
        const res = await axios.get(
          `${API_URL}/lessons/${levelId[level]}/categories`,
          { withCredentials: true }
        );
        setLessons(res.data.categoriesWithWord || []);
        console.log("Lessons:", res.data);
      } catch (err) {
        console.error(`${level} 강의 불러오기 실패:`, err);
      }
    };

    const fetchProgress = async () => {
      try {
        const token = getToken();

        if (!token) {
          console.log("토큰이 없습니다.");
          return;
        }

        // 다음 강의 정보 가져오기
        const continueRes = await axios.get(`${API_URL}/progress/continue`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
          withCredentials: true,
        });
        setNextLesson(continueRes.data.nextLesson?.[0] || null);

        // 완료된 토픽(step) 정보 가져오기
        const topicsRes = await axios.post(
          `${API_URL}/progress/topics`,
          {},
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
            withCredentials: true,
          }
        );
        setCompletedTopics(topicsRes.data || []);
        console.log("Completed topics:", topicsRes.data);
      } catch (err) {
        console.error("진도 불러오기 실패:", err);
      }
    };

    fetchLessons();
    fetchProgress();
  }, [level]);

  // 특정 파트의 모든 스텝이 완료되었는지 확인
  const isPartCompleted = async (partId) => {
    try {
      // 해당 파트의 모든 스텝을 가져옴
      const res = await axios.get(`${API_URL}/lessons/${partId}/topics`, {
        withCredentials: true,
      });
      const allSteps = res.data;

      // 모든 스텝이 완료되었는지 확인
      const allCompleted = allSteps.every((step) =>
        completedTopics.some(
          (completed) =>
            completed.lesson_id === step.lesson_id &&
            completed.status === "completed"
        )
      );

      return allCompleted;
    } catch (err) {
      console.error("파트 완료 상태 확인 실패:", err);
      return false;
    }
  };

  // 현재 레벨에서 완료된 파트 수 계산
  const getCompletedPartsCount = async () => {
    let completedCount = 0;

    // part_number 순으로 정렬
    const sortedLessons = [...lessons].sort(
      (a, b) => a.part_number - b.part_number
    );

    for (const lesson of sortedLessons) {
      const isCompleted = await isPartCompleted(lesson.lessonCategory_id);
      if (isCompleted) {
        completedCount++;
      } else {
        // 순서대로 확인하다가 완료되지 않은 파트를 만나면 중단
        break;
      }
    }

    return completedCount;
  };

  // 파트가 잠겨있는지 확인
  const isPartLocked = (lesson) => {
    // 첫 번째 파트(Part 1)는 항상 열림
    if (lesson.part_number === 1) return false;

    // 이전 파트들이 모두 완료되었는지 확인해야 하므로 비동기 처리 필요
    // 실제 구현에서는 useEffect에서 미리 계산된 값을 사용
    return false; // 임시로 false 반환, 실제로는 아래 로직 사용
  };

  // 파트별 완료 상태를 미리 계산
  const [partCompletionStatus, setPartCompletionStatus] = useState({});

  useEffect(() => {
    const calculatePartCompletion = async () => {
      if (lessons.length === 0 || completedTopics.length === 0) return;

      const completionStatus = {};

      for (const lesson of lessons) {
        try {
          const res = await axios.get(
            `${API_URL}/lessons/${lesson.lessonCategory_id}/topics`,
            {
              withCredentials: true,
            }
          );
          const allSteps = res.data;

          const allCompleted = allSteps.every((step) =>
            completedTopics.some(
              (completed) =>
                completed.lesson_id === step.lesson_id &&
                completed.status === "completed"
            )
          );

          completionStatus[lesson.lessonCategory_id] = allCompleted;
        } catch (err) {
          console.error("파트 완료 상태 확인 실패:", err);
          completionStatus[lesson.lessonCategory_id] = false;
        }
      }

      setPartCompletionStatus(completionStatus);
    };

    calculatePartCompletion();
  }, [lessons, completedTopics]);

  // 개선된 파트 잠금 로직
  const isPartLockedImproved = (lesson) => {
    // 첫 번째 파트는 항상 열림
    if (lesson.part_number === 1) return false;

    // 이전 파트가 완료되었는지 확인
    const sortedLessons = [...lessons].sort(
      (a, b) => a.part_number - b.part_number
    );
    const currentIndex = sortedLessons.findIndex(
      (l) => l.lessonCategory_id === lesson.lessonCategory_id
    );

    if (currentIndex <= 0) return false;

    // 이전 파트가 완료되었는지 확인
    const prevLesson = sortedLessons[currentIndex - 1];
    return !partCompletionStatus[prevLesson.lessonCategory_id];
  };

  const handleLessonClick = (lesson) => {
    const isLocked = isPartLockedImproved(lesson);

    if (!isLocked) {
      navigate(`/classroom/${level}/${lesson.lessonCategory_id}`, {
        state: { lesson },
      });
    } else {
      alert("이 강의는 이전 강의의 모든 스텝을 완료한 후 이용할 수 있습니다.");
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
              {lessons
                .sort((a, b) => a.part_number - b.part_number) // part_number 순으로 정렬
                .map((lesson) => {
                  const isLocked = isPartLockedImproved(lesson);
                  const isCompleted =
                    partCompletionStatus[lesson.lessonCategory_id] || false;

                  return (
                    <div
                      key={lesson.lessonCategory_id}
                      className={`flex p-4 rounded-[20px] transition-all duration-200 ${
                        isLocked
                          ? "cursor-not-allowed"
                          : "cursor-pointer hover:shadow-lg transform hover:-translate-y-1"
                      }`}
                      onClick={() => handleLessonClick(lesson)}
                    >
                      <div className="relative p-4 rounded-[15px] shadow-lg bg-[#F2F2F2]">
                        <img
                          src="/assets/images/Sign.png"
                          alt=""
                          className="w-20 h-20"
                        />
                        {isLocked && (
                          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 rounded-[15px]">
                            <FiLock className="text-white" size={32} />
                          </div>
                        )}

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

                      <div className="flex flex-col justify-center ml-4">
                        <p className="font-bold text-lg">
                          Part {lesson.part_number}. {lesson.category}
                        </p>
                        <p className="text-sm text-gray-600 truncate w-[150px]">
                          {lesson.words?.join(", ")}
                        </p>
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
