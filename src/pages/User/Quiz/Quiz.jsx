import React, { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { getToken } from "../../../utils/authStorage";
import { API_URL } from "../../../config";
import UserNav from "../../../components/UserNav";
import UserTitle from "../../../components/UserTitle";

export default function Quiz() {
  const [quizzes, setQuizzes] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [userAnswers, setUserAnswers] = useState([]);
  const [animation, setAnimation] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [quizResult, setQuizResult] = useState({ score: 0, total: 0 });

  const navigate = useNavigate();

  useEffect(() => {
    const fetchQuizzes = async () => {
      try {
        const token = getToken();
        console.log("토큰 확인:", token); // 디버깅용

        if (!token) {
          console.log("토큰이 없습니다.");
          alert("로그인이 필요합니다.");
          navigate("/login");
          return;
        }

        const res = await axios.get(`${API_URL}/quiz/generate`, {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          withCredentials: true,
        });

        setQuizzes(res.data.quizzes);
        setSessionId(res.data.sessionId);
        setUserAnswers(new Array(res.data.quizzes.length).fill(null));
        setAnimation(res.data.quizzes[0]?.animation_path || null);

        console.log("퀴즈 데이터:", res.data);
      } catch (err) {
        console.error("퀴즈 불러오기 실패:", err);
        if (err.response?.status === 401) {
          alert("로그인이 만료되었습니다. 다시 로그인해주세요.");
          navigate("/login");
        }
      }
    };

    fetchQuizzes();
  }, [navigate]);

  const handleAnswer = (answer) => {
    const updatedAnswers = [...userAnswers];
    updatedAnswers[currentQuestion] = {
      quiz_id: quizzes[currentQuestion].quiz_id,
      answer,
    };
    setUserAnswers(updatedAnswers);

    if (currentQuestion < quizzes.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
      setAnimation(quizzes[currentQuestion + 1].animation_path);
    }
  };

  useEffect(() => {
    if (userAnswers.every((ans) => ans !== null) && quizzes.length > 0) {
      submitQuiz();
    }
  }, [userAnswers, quizzes.length]);

  const submitQuiz = async () => {
    if (isSubmitting) return;
    setIsSubmitting(true);

    try {
      const token = getToken();
      console.log("제출 시 토큰:", token); // 디버깅용

      if (!token) {
        alert("로그인이 필요합니다.");
        navigate("/login");
        return;
      }

      console.log("제출할 데이터:", {
        sessionId,
        answers: userAnswers,
      });

      // 올바른 axios 요청 형식
      const res = await axios.post(
        `${API_URL}/quiz/check`,
        {
          sessionId,
          answers: userAnswers,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          withCredentials: true,
        }
      );

      if (res.status === 200) {
        setQuizResult({ score: res.data.score, total: res.data.total });
        console.log("퀴즈 제출 성공:", res.data);

        // 결과에 따른 추가 처리
        if (res.data.passed) {
          setTimeout(() => {
            alert(
              `축하합니다! ${res.data.total}문제 중 ${res.data.score}문제를 맞췄습니다!`
            );
          }, 1000);
        } else {
          setTimeout(() => {
            alert(
              `${res.data.total}문제 중 ${res.data.score}문제를 맞췄습니다.`
            );
          }, 1000);
        }
      }
    } catch (err) {
      console.error("퀴즈 제출 실패:", err);

      if (err.response?.status === 401) {
        alert("로그인이 만료되었습니다. 다시 로그인해주세요.");
        navigate("/login");
      } else if (err.response?.status === 400) {
        alert("잘못된 요청입니다. 다시 시도해주세요.");
      } else {
        alert("퀴즈 제출에 실패했습니다. 다시 시도해주세요.");
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  // 로딩 중일 때
  if (quizzes.length === 0 && !isSubmitting) {
    return (
      <div className="min-h-screen bg-[#F28079]">
        <UserTitle />
        <div className="flex w-full">
          <UserNav />
          <div className="flex flex-col mr-10 w-full rounded-[40px] bg-[#fafafa] shadow-xl h-[850px] px-12 py-9 overflow-y-auto">
            <div className="flex items-center justify-center h-full">
              <div className="text-2xl font-bold text-gray-600">
                퀴즈 로딩 중...
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#F28079]">
      <UserTitle />
      <div className="flex w-full">
        <UserNav />
        <div className="flex flex-col mr-10 w-full rounded-[40px] bg-[#fafafa] shadow-xl h-[850px] px-12 py-9 overflow-y-auto">
          <img
            src="/assets/images/review/OXTitle.png"
            alt="OX 타이틀"
            className="w-[150px] m-3"
          />

          {/* 문제별 하트 표시 */}
          <div className="text-4xl mb-4 flex justify-center gap-1">
            {quizzes.map((quiz, index) => {
              const answer = userAnswers[index];
              if (answer === null) return <span key={index}>🩶</span>;
              if (answer.answer === quiz.check_answer)
                return <span key={index}>❤️</span>;
              return <span key={index}>💔</span>;
            })}
          </div>

          {/* 현재 문제가 있을 때만 표시 */}
          {quizzes[currentQuestion] && (
            <>
              {/* 비디오 */}
              {animation && (
                <div className="mb-6">
                  <video
                    src={animation}
                    autoPlay
                    loop
                    muted
                    className="w-[50%] rounded-lg mx-auto shadow-lg"
                    onError={(e) => console.log("Video error:", e)}
                  />
                </div>
              )}

              {/* 질문 */}
              <h2 className="text-3xl font-bold my-6 text-center">
                {quizzes[currentQuestion].question}
              </h2>

              {/* 버튼 - 모든 문제에 답하지 않았을 때만 표시 */}
              {!userAnswers.every((ans) => ans !== null) && (
                <div className="flex justify-center gap-6">
                  <button
                    className="bg-white px-8 py-6 h-[150px] rounded-xl text-[70px] shadow-md hover:scale-105 transition-transform disabled:opacity-50"
                    onClick={() => handleAnswer(true)}
                    disabled={isSubmitting}
                  >
                    ⭕️
                  </button>
                  <button
                    className="bg-white px-10 py-6 rounded-xl text-[70px] shadow-md hover:scale-105 transition-transform disabled:opacity-50"
                    onClick={() => handleAnswer(false)}
                    disabled={isSubmitting}
                  >
                    ❌
                  </button>
                </div>
              )}
            </>
          )}

          {/* 제출 중 표시 */}
          {isSubmitting && (
            <div className="mt-6 text-center text-2xl font-bold text-blue-600">
              퀴즈 제출 중...
            </div>
          )}

          {/* 결과 */}
          {userAnswers.every((ans) => ans !== null) && quizResult.total > 0 && (
            <div className="mt-6 text-center">
              <div className="text-2xl font-bold mb-4">
                {quizResult.total}문제 중 {quizResult.score}문제 맞췄습니다!
              </div>
              <button
                onClick={() => navigate("/classroom/easy")}
                className="bg-[#FFE694] px-6 py-3 rounded-lg font-bold hover:shadow-lg transition-all"
              >
                학습으로 돌아가기
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
