import React, { useState, useEffect } from "react";
import UserNav from "../../../components/UserNav";
import UserTitle from "../../../components/UserTitle";
import DailyCheckIn from "./DailyCheckIn";
import ContinueLearning from "./ContinueLearning";
import axios from "axios";
import { getToken } from "../../../utils/authStorage";
import { useNavigate } from "react-router-dom";
import { API_URL } from "../../../config";

export default function Home() {
  const [userInfo, setUserInfo] = useState({});
  const navigate = useNavigate();

  useEffect(() => {
    const fetchUserInfo = async () => {
      try {
        const token = getToken();

        if (!token) {
          console.log("토큰이 없습니다.");
          return;
        }

        console.log("사용할 토큰:", token.substring(0, 20) + "...");

        const response = await axios.get(`${API_URL}/login/success`, {
          headers: {
            Authorization: `Bearer ${token}`, // Bearer 접두사 추가
            "Content-Type": "application/json",
          },
          withCredentials: true, // 쿠키도 함께 전송
        });

        if (response.data) {
          console.log("받아온 사용자 정보:", response.data);
          setUserInfo(response.data);
        }
      } catch (error) {
        console.error("사용자 정보 가져오기 실패:", error);
        console.error("에러 상세:", error.response?.data);

        // 토큰이 만료되었거나 유효하지 않은 경우
        if (error.response?.status === 401) {
          console.log("토큰이 만료되었거나 유효하지 않습니다.");
          // 필요시 로그인 페이지로 리다이렉트
          // navigate('/login');
        }
      }
    };

    fetchUserInfo();
  }, []);

  return (
    <div className="min-h-screen bg-[#D2D2D2]">
      <UserTitle />

      <div className="flex w-full">
        <UserNav />

        <div className="flex flex-col mr-10 w-full rounded-3xl bg-[#fafafa] h-[850px]">
          <div className="flex h-full flex items-center">
            {/* 왼 */}
            <div className="w-[50%] flex flex-col items-center justify-center">
              <img
                src="/assets/images/sonsu.png"
                alt=""
                className="w-[220px]"
              />
              <p className="text-[32px] fontMedium mt-12">
                안녕하세요, {userInfo?.username ? userInfo.username : "사용자"}
                님!
              </p>
            </div>
            <div className="w-[1px] h-[500px] bg-[#D9D9D9]"></div>
            {/* 오 */}
            <div className="w-[50%] h-[500px] flex flex-col ml-32 justify-between">
              {/* 출석 */}
              <DailyCheckIn />
              {/* 이어서학습하기 */}
              <ContinueLearning />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
