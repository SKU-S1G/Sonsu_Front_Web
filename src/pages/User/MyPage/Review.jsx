import { useEffect, useState } from "react";
import axios from "axios";
import { API_URL } from "../../../config";
import { useNavigate } from "react-router-dom";
import { getToken } from "../../../utils/authStorage";

export default function Review() {
  const [cards, setCards] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchWrongAnswers = async () => {
      try {
        const token = getToken();

        if (!token) {
          console.error("토큰이 없습니다. 로그인이 필요합니다.");
          navigate("/login"); // 로그인 페이지로 리다이렉트
          return;
        }

        const response = await axios.get(`${API_URL}/quiz/wrong`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        const data = response.data.rows;
        console.log(data);

        const sortedData = data.sort(
          (a, b) => new Date(b.recorded_at) - new Date(a.recorded_at)
        );
        setCards(sortedData.slice(0, 10));
      } catch (error) {
        console.error("오답 수어 데이터를 불러오지 못했습니다.", error);

        // 401 에러인 경우 로그인 페이지로 리다이렉트
        if (error.response?.status === 401) {
          console.error("인증 실패: 로그인이 필요합니다.");
          navigate("/login");
        }
      } finally {
        setLoading(false);
      }
    };

    fetchWrongAnswers();
  }, [navigate]);

  if (loading) {
    return <p className="text-center mt-5">Loading...</p>;
  }

  return (
    <div className="w-fit">
      <div className="flex justify-between mt-3 mb-3">
        <h2 className="text-lg font-bold">오답 수어 다시보기</h2>
      </div>
      <div className="w-[75%] max-w-4xl bg-white mt-5 mb-10 p-5 rounded-2xl shadow-lg">
        <div className="flex gap-4 overflow-x-auto pb-3">
          {cards.map((lesson, index) => (
            <Card
              key={index}
              lesson={lesson}
              index={index}
              onClick={() =>
                navigate(`/study/${lesson.lesson_id}`, {
                  state: {
                    topic: {
                      word: lesson.word,
                      lesson_id: lesson.lesson_id,
                      animation_path: lesson.animation_path,
                      step_number: index + 1,
                    },
                    lesson,
                    index,
                  },
                })
              }
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function Card({ lesson, index, onClick }) {
  return (
    <div className="flex flex-col items-center min-w-[180px] bg-gray-50 rounded-xl shadow-md p-3">
      <div className="w-[160px] h-[90px] bg-black rounded-lg overflow-hidden flex items-center justify-center">
        <video
          src={lesson.animation_path}
          autoPlay
          muted
          loop
          playsInline
          className="w-full h-full object-contain"
        />
      </div>
      <p className="text-sm font-semibold mt-2 text-center truncate w-full">
        Part {lesson.lesson_id}. {lesson.word}
      </p>
      <button
        onClick={onClick}
        className="bg-yellow-300 hover:bg-yellow-400 text-xs font-medium mt-2 px-3 py-1.5 rounded-lg"
      >
        다시 학습하기
      </button>
    </div>
  );
}
