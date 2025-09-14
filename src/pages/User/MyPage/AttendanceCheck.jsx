import React, { useState, useEffect, useRef } from "react";
import { API_URL } from "../../../config";
import { getToken } from "../../../utils/authStorage";

export default function AttendanceCheck({ isActive, setIsActive }) {
  const [daysInMonth, setDaysInMonth] = useState([]);
  const [currentDate, setCurrentDate] = useState(new Date());
  const [attendanceData, setAttendanceData] = useState([]);
  const containerRef = useRef(null);
  const bubbleRef = useRef(null);

  //   바깥 클릭 감지 - 컨테이너와 말풍선 모두 제외
  useEffect(() => {
    const handleClickOutside = (event) => {
      // containerRef와 bubbleRef 둘 다 체크
      const isClickInsideContainer =
        containerRef.current && containerRef.current.contains(event.target);
      const isClickInsideBubble =
        bubbleRef.current && bubbleRef.current.contains(event.target);

      if (!isClickInsideContainer && !isClickInsideBubble) {
        setIsActive(false);
      }
    };

    if (isActive) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isActive, setIsActive]);

  // 출석 데이터 Fetch
  useEffect(() => {
    const fetchAttendanceData = async () => {
      try {
        const token = getToken();

        if (!token) {
          console.log("토큰이 없습니다.");
          return;
        }

        const response = await fetch(`${API_URL}/attend`, {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          withCredentials: true,
        });
        const data = await response.json();
        setAttendanceData(Array.isArray(data.data) ? data.data : []);
      } catch (error) {
        console.error("출석 데이터 가져오기 실패", error);
      }
    };

    if (isActive) {
      fetchAttendanceData();
    }
  }, [isActive]);

  // 현재 달의 출석한 날 수 계산
  const getAttendanceCount = () => {
    const currentMonth = currentDate.getMonth();
    const currentYear = currentDate.getFullYear();

    const attendedDays = attendanceData.filter((attendance) => {
      const attendDate = new Date(attendance.attend_date);
      return (
        attendDate.getMonth() === currentMonth &&
        attendDate.getFullYear() === currentYear &&
        attendance.status === 1
      );
    });

    return attendedDays.length;
  };

  // 현재 달의 일 수 계산
  const getDaysInMonth = () => {
    const year = currentDate.getFullYear();
    const month = currentDate.getMonth();
    const lastDay = new Date(year, month + 1, 0);
    return lastDay.getDate();
  };

  // 출석률 계산
  const attendanceCount = getAttendanceCount();
  const totalDays = getDaysInMonth();
  const attendancePercentage =
    totalDays > 0 ? (attendanceCount / totalDays) * 100 : 0;

  // 현재 날짜에 맞는 달력 데이터 생성
  useEffect(() => {
    const now = currentDate;
    const year = now.getFullYear();
    const month = now.getMonth();
    const firstDayOfMonth = new Date(year, month, 1);
    const lastDayOfMonth = new Date(year, month + 1, 0);

    const days = [];
    const startDay = firstDayOfMonth.getDay();
    const lastDate = lastDayOfMonth.getDate();

    // 이전 달의 날짜 추가
    const prevMonth = new Date(year, month, 0);
    const prevMonthLastDate = prevMonth.getDate();
    for (let i = startDay; i > 0; i--) {
      days.push({
        date: prevMonthLastDate - i + 1,
        isCurrentMonth: false,
      });
    }

    // 현재 월의 날짜들 추가
    for (let i = 1; i <= lastDate; i++) {
      days.push({ date: i, isCurrentMonth: true });
    }

    // 다음 달 날짜 추가 (42개 맞추기)
    const totalCells = 42;
    const remainingCells = totalCells - days.length;
    for (let i = 1; i <= remainingCells; i++) {
      days.push({
        date: i,
        isCurrentMonth: false,
      });
    }

    setDaysInMonth(days);
  }, [currentDate]);

  // 오늘 날짜와 비교하여 강조하는 함수
  const isToday = (day) => {
    const today = new Date();
    return (
      today.getDate() === day.date &&
      today.getMonth() === currentDate.getMonth() &&
      today.getFullYear() === currentDate.getFullYear() &&
      day.isCurrentMonth
    );
  };

  // 출석 상태가 있는지 확인하는 함수
  const isAttended = (date) => {
    const formattedDate = new Date(
      currentDate.getFullYear(),
      currentDate.getMonth(),
      date
    );
    return attendanceData.some((attendance) => {
      const attendDate = new Date(attendance.attend_date);
      return (
        attendDate.getDate() === formattedDate.getDate() &&
        attendDate.getMonth() === formattedDate.getMonth() &&
        attendDate.getFullYear() === formattedDate.getFullYear() &&
        attendance.status === 1
      );
    });
  };

  // 이전 달로 이동
  const goToPreviousMonth = () => {
    const newDate = new Date(
      currentDate.getFullYear(),
      currentDate.getMonth() - 1,
      1
    );
    setCurrentDate(newDate);
  };

  // 이후 달로 이동
  const goToNextMonth = () => {
    const newDate = new Date(
      currentDate.getFullYear(),
      currentDate.getMonth() + 1,
      1
    );
    setCurrentDate(newDate);
  };

  // 요일 배열
  const weekdays = ["일", "월", "화", "수", "목", "금", "토"];

  return (
    <div className="relative">
      <div
        ref={containerRef}
        onClick={() => setIsActive(!isActive)}
        className={`bg-[#FFEEB8] rounded-[20px] pl-8 pt-6 w-full shadow-lg h-[200px]
             transform transition duration-300 ease-in-out
             cursor-pointer
             ${isActive ? "scale-110 shadow-2xl" : "scale-100 shadow-lg"}
             ${isActive ? "filter brightness-105" : ""}`}
      >
        <p className="fontSB text-[24px]">출석체크</p>
        <div className="fontSB text-[13px] text-[#555] mt-3">
          <p>내 수어 학습 출석률은?</p>
          <p>오늘도 한 걸음 더 나아가요!</p>
        </div>

        <div className="flex justify-end -mt-6">
          <img
            src="/assets/images/MyPage/calendar.png"
            alt=""
            className="w-[45%] opacity-80"
          />
        </div>
      </div>

      {/* 클릭 시 나오는 말풍선 */}
      {isActive && (
        <div
          ref={bubbleRef}
          className="absolute -left-10 top-0 mt-0 -translate-x-full w-[500px] p-6 bg-white rounded-xl shadow-2xl z-50 transform transition duration-300 ease-in-out max-h-[600px] overflow-y-auto"
        >
          {/* 헤더 */}
          <div className="flex justify-center items-center mb-4">
            <div className="flex items-center gap-4">
              <button
                onClick={goToPreviousMonth}
                className="text-gray-600 hover:text-gray-800 text-xl font-bold"
              >
                ←
              </button>
              <h3 className="fontSB text-[18px] ">
                {currentDate.getFullYear()}년 {currentDate.getMonth() + 1}월
              </h3>
              <button
                onClick={goToNextMonth}
                className="text-gray-600 hover:text-gray-800 text-xl font-bold"
              >
                →
              </button>
            </div>
          </div>

          {/* 달력 */}
          <div className="flex justify-center">
            <div className="bg-white rounded-lg w-[80%]">
              {/* 요일 헤더 */}
              <div className="grid grid-cols-7 gap-1 mb-2">
                {weekdays.map((weekday, index) => (
                  <div
                    key={index}
                    className="text-center py-2 font-semibold text-gray-600"
                  >
                    {weekday}
                  </div>
                ))}
              </div>

              {/* 날짜 */}
              <div className="grid grid-cols-7 gap-1">
                {daysInMonth.map((day, index) => (
                  <div
                    key={`${day.date}-${day.isCurrentMonth}-${index}`}
                    className={`relative aspect-square flex items-center justify-center text-sm rounded-lg
                    ${!day.isCurrentMonth ? "text-gray-300" : "text-gray-700"}
                    ${isToday(day) ? "text-orange-500 font-bold" : ""}
                    ${isAttended(day.date) && day.isCurrentMonth ? "" : ""}
                  `}
                  >
                    <span>{day.date}</span>
                    {isAttended(day.date) && day.isCurrentMonth && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-10 h-10 rounded-full bg-yellow-200 flex items-center justify-center">
                          <img
                            src="/assets/images/logo.png"
                            alt="출석"
                            className="w-8 h-8"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* 출석률 정보 */}
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">이번 달 출석률</span>
              <span className="text-lg font-bold text-yellow-600">
                {attendancePercentage.toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between items-center mt-1">
              <span className="text-xs text-gray-500">
                {attendanceCount}일 / {totalDays}일 출석
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
              <div
                className="bg-yellow-400 h-2 rounded-full transition-all duration-500"
                style={{ width: `${attendancePercentage}%` }}
              ></div>
            </div>
          </div>

          {/* 안내 메시지 */}
          <div className="mt-3 text-xs text-gray-500 text-center">
            💡 출석은 하루에 최소 1개 이상의 학습을 완료해야만 인정됩니다.
          </div>
        </div>
      )}
    </div>
  );
}
