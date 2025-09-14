import { useEffect, useState } from "react";
import { FiAlertCircle } from "react-icons/fi";
import axios from "axios";
import { jwtDecode } from "jwt-decode";
import { API_URL } from "../../../config";
import InfoModal from "../../../components/InfoModal";
import { getToken } from "../../../utils/authStorage";

const WeeklyRank = () => {
  const [modalVisible, setModalVisible] = useState(false);
  const [userRankData, setUserRankData] = useState([]);
  const [myUsername, setMyUsername] = useState("");

  useEffect(() => {
    const token = getToken();
    if (token) {
      try {
        const decoded = jwtDecode(token);
        setMyUsername(decoded.iss);
      } catch (err) {
        console.error("❌ jwtDecode 실패:", err);
      }
    }
  }, []);

  useEffect(() => {
    const fetchRanking = async () => {
      try {
        const response = await axios.get(`${API_URL}/mypage/ranking`);
        console.log("API 응답:", response.data);

        const data = response.data;
        if (!data || data.length === 0) return;

        // 최대 점수 계산 (프로그레스바용)
        const maxPoints = Math.max(...data.map((user) => user.week_points));

        // 상위 3명 + 현재 사용자 처리
        const processedData = [];

        // 상위 3명 추가
        const top3 = data.slice(0, 3);
        top3.forEach((user, index) => {
          processedData.push({
            rank: index + 1,
            name: user.username,
            progress: user.week_points,
            progressRate:
              maxPoints > 0 ? (user.week_points / maxPoints) * 100 : 0,
            isCurrentUser: user.username === myUsername,
          });
        });

        // 현재 사용자가 top3에 없으면 추가
        if (myUsername && !top3.some((user) => user.username === myUsername)) {
          const currentUserIndex = data.findIndex(
            (user) => user.username === myUsername
          );
          if (currentUserIndex !== -1) {
            const currentUser = data[currentUserIndex];
            processedData.push({
              rank: currentUserIndex + 1,
              name: currentUser.username,
              progress: currentUser.week_points,
              progressRate:
                maxPoints > 0 ? (currentUser.week_points / maxPoints) * 100 : 0,
              isCurrentUser: true,
            });
          }
        }

        setUserRankData(processedData);
      } catch (error) {
        console.error("❌ 랭킹 API 에러:", error);
      }
    };

    if (myUsername) {
      fetchRanking();
    }
  }, [myUsername]);

  return (
    <div className="w-[80%]">
      <div className="flex justify-between mt-3 mb-3">
        <h2 className="text-lg font-bold">주간 랭킹</h2>
        <button onClick={() => setModalVisible(true)}>
          <FiAlertCircle size={21} className="text-gray-500" />
        </button>
      </div>

      {userRankData.length === 0 ? (
        <div className="text-center py-4 text-gray-500">
          랭킹 데이터가 없습니다.
        </div>
      ) : (
        userRankData.map((user, index) => (
          <div
            key={index}
            className={`flex items-center px-6 py-2.5 border-b border-gray-300 transition ${
              user.isCurrentUser
                ? "bg-[#FFEEB8] font-bold text-black rounded-lg shadow-md"
                : ""
            }`}
          >
            <div className="flex items-baseline w-1/2 space-x-2">
              <span
                className={`text-2xl font-bold drop-shadow ${
                  user.rank === 1 ? "text-yellow-500" : "text-black"
                }`}
              >
                {user.rank}
              </span>
              <span className="text-lg text-gray-800">{user.name}</span>
              <span className="text-xs">{user.progress}점</span>
            </div>
            <div className="mt-2 w-full">
              <div className="w-full h-2 bg-gray-200 rounded overflow-hidden shadow">
                <div
                  className={`h-full ${
                    user.isCurrentUser ? "bg-yellow-500" : "bg-yellow-400"
                  } rounded`}
                  style={{ width: `${user.progressRate}%` }}
                ></div>
              </div>
            </div>
          </div>
        ))
      )}

      <InfoModal
        visible={modalVisible}
        onClose={() => setModalVisible(false)}
        title="주간랭킹"
        content={
          "사용자의 학습 점수를 통해 랭킹을 결정합니다.\n점수를 높여 더 좋은 랭킹을 달성하세요!"
        }
      />
    </div>
  );
};

export default WeeklyRank;
