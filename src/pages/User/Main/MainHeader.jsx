import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { getToken, removeToken } from "../../../utils/authStorage";
import { API_URL } from "../../../config";

export default function MainHeader() {
  const [userInfo, setUserInfo] = useState(null);
  const navigate = useNavigate();

  // 사용자 정보 가져오기
  const fetchUserInfo = async () => {
    try {
      const token = getToken();
      const config = token
        ? {
            headers: { Authorization: `Bearer ${token}` },
            withCredentials: true,
          }
        : { withCredentials: true };

      const response = await axios.get(`${API_URL}/login/success`, config);

      if (response.data?.username) setUserInfo(response.data);
      else setUserInfo(null);
    } catch (error) {
      console.error("사용자 정보 가져오기 실패:", error);
      setUserInfo(null);
    }
  };

  useEffect(() => {
    fetchUserInfo();
  }, []);

  // 로그아웃
  const handleLogout = async () => {
    const confirmLogout = window.confirm("정말 로그아웃 하시겠습니까?");
    if (!confirmLogout) return;

    try {
      const token = getToken();
      const config = token
        ? {
            headers: { Authorization: `Bearer ${token}` },
            withCredentials: true,
          }
        : { withCredentials: true };

      // 로그아웃 요청
      await axios.post(`${API_URL}/logout`, {}, config);
    } catch (error) {
      console.error("로그아웃 중 오류 발생:", error);
    } finally {
      // 토큰 삭제 + 상태 초기화 + 홈 이동
      removeToken();
      setUserInfo(null);
      navigate("/");
    }
  };

  return (
    <div className="flex items-center justify-between px-8 py-4">
      {/* 로고 */}
      <div className="flex items-center">
        <img src="assets/images/logo.png" className="w-20" alt="Logo" />
        <div className="text-[25px] fontBold text-[#222]">SONSU</div>
      </div>

      {/* 로그인 상태에 따라 버튼/인사 표시 */}
      <div className="flex items-center space-x-4">
        {userInfo?.username ? (
          <div className="flex items-center space-x-2">
            <div className="text-[#222] fontSB">
              안녕하세요, {userInfo.username}님!
            </div>
            <div className="w-[1px] bg-[#222] self-center h-4"></div>
            <button
              className="fontSB text-[#222] rounded-lg hover:text-[#DBBF63] transition-colors"
              onClick={handleLogout}
            >
              Logout
            </button>
          </div>
        ) : (
          <>
            <button
              className="fontSB text-[#222] hover:text-[#DBBF63] transition-colors"
              onClick={() => navigate("/login")}
            >
              Login
            </button>
            <div className="w-[1px] bg-[#222] self-center h-4"></div>
            <button
              className="fontSB text-[#222] rounded-lg hover:text-[#DBBF63] transition-colors"
              // onClick={() => navigate("/signup")}
            >
              Sign Up
            </button>
          </>
        )}
      </div>
    </div>
  );
}
