import React, { useState, useEffect, useCallback } from "react";
import { API_URL } from "../../../config";
import { getToken } from "../../../utils/authStorage";
import axios from "axios";

const DailyCheckIn = () => {
  const [attendedDates, setAttendedDates] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const weekDays = ["일", "월", "화", "수", "목", "금", "토"];

  const getCurrentWeek = () => {
    const today = new Date();
    const startOfWeek = new Date(
      today.setDate(today.getDate() - today.getDay())
    );

    return Array.from({ length: 7 }).map((_, i) => {
      const date = new Date(startOfWeek);
      date.setDate(startOfWeek.getDate() + i);
      return {
        date: date.getDate(),
        day: weekDays[date.getDay()],
        fullDate: date.toISOString().split("T")[0],
      };
    });
  };

  const [week] = useState(getCurrentWeek);

  const fetchAttendanceData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const token = getToken();
      if (!token) {
        console.log("토큰이 없습니다.");
        return;
      }

      const response = await axios.get(`${API_URL}/attend`, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        withCredentials: true,
      });

      console.log("출석 API 응답:", response.data); // 디버깅용

      const data = response.data;
      const attendedMap = {};

      // 데이터 타입 확인 및 안전한 처리
      let attendanceArray = [];

      if (Array.isArray(data)) {
        attendanceArray = data;
      } else if (data && Array.isArray(data.data)) {
        attendanceArray = data.data;
      } else if (data && Array.isArray(data.attendance)) {
        attendanceArray = data.attendance;
      } else {
        console.warn("예상치 못한 데이터 형태:", data);
        setAttendedDates({});
        return;
      }

      attendanceArray.forEach((item) => {
        try {
          const dateObj = new Date(item.attend_date);
          // KST 기준 날짜 추출
          const kstOffset = 9 * 60 * 60 * 1000;
          const localDate = new Date(dateObj.getTime() + kstOffset);
          const dateString = localDate.toISOString().split("T")[0];
          attendedMap[dateString] = true;
        } catch (dateError) {
          console.error("날짜 처리 오류:", dateError, item);
        }
      });

      setAttendedDates(attendedMap);
    } catch (error) {
      console.error("출석 데이터를 불러오는 중 오류 발생:", error);
      console.error("에러 응답:", error.response?.data);
      setError("출석 데이터를 불러오는데 실패했습니다.");
      setAttendedDates({});
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAttendanceData();
  }, [fetchAttendanceData]);

  const handleNavigateToAttendance = () => {
    console.log("출석 체크 페이지로 이동");
    window.location.href = "/mypage";
  };

  if (loading) {
    return (
      <div className="w-[400px]">
        <div className="flex justify-center items-center h-32">
          <div className="text-gray-500">출석 데이터를 불러오는 중...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-[400px]">
      {/* 상단 타이틀 */}
      <div className="flex justify-between items-center">
        <p className="text-[20px] font-semibold">오늘의 출석</p>
      </div>

      {/* 에러 상태 */}
      {error && (
        <div className="mt-4 p-3 bg-red-100 border border-red-300 rounded text-red-700 text-sm">
          {error}
          <button
            onClick={fetchAttendanceData}
            className="ml-2 px-2 py-1 bg-red-500 text-white rounded text-xs hover:bg-red-600"
          >
            다시 시도
          </button>
        </div>
      )}

      {/* 캘린더 */}
      <div className="flex justify-around items-center h-[100px] bg-white rounded-2xl mt-6 px-3 py-4 shadow-md ml-[-8px]">
        {week.map(({ date, day, fullDate }, index) => (
          <div key={index} className="flex flex-col items-center">
            <p className="text-[14px] font-bold text-black mb-2">{day}</p>
            <div
              className={`w-[35px] h-[35px] flex items-center justify-center rounded-full ${
                attendedDates[fullDate] ? "bg-[#FFEB99]" : ""
              }`}
            >
              {attendedDates[fullDate] ? (
                <div className="w-[30px] h-[30px] rounded-full flex items-center justify-center">
                  <img src="/assets/images/logo.png" alt="" />
                </div>
              ) : (
                <p className="text-[14px] font-bold text-[#333]">{date}</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DailyCheckIn;
