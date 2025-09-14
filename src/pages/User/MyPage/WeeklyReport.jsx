import React, { useState, useEffect, useRef } from "react";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer } from "recharts";
import { API_URL } from "../../../config";
import { getToken } from "../../../utils/authStorage";

export default function WeeklyReport({ isActive, setIsActive }) {
  const [reportData, setReportData] = useState(null);
  const [barData, setBarData] = useState([]);
  const [loading, setLoading] = useState(false);

  // 주간 리포트 데이터 Fetch
  useEffect(() => {
    const fetchReport = async () => {
      if (!isActive) return; // 활성화되지 않았으면 API 호출하지 않음

      setLoading(true);
      const accessToken = getToken();
      let retryCount = 0;
      const maxRetries = 3;

      while (retryCount < maxRetries) {
        try {
          const response = await fetch(`${API_URL}/mypage/report`, {
            headers: {
              Authorization: `Bearer ${accessToken}`,
            },
          });

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const data = await response.json();
          setReportData(data);

          const dayMap = {
            Sunday: "일",
            Monday: "월",
            Tuesday: "화",
            Wednesday: "수",
            Thursday: "목",
            Friday: "금",
            Saturday: "토",
          };

          const fullWeek = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
          ];

          const chartData = fullWeek.map((day) => {
            const found = data.lessonCount?.find((d) => d.day === day);
            return {
              name: dayMap[day],
              value: found ? found.lesson_count : 0,
            };
          });

          setBarData(chartData);
          break; // 성공했으므로 루프 종료
        } catch (error) {
          if (error.message.includes("429")) {
            const waitTime = Math.pow(2, retryCount) * 1000; // 1초, 2초, 4초
            console.warn(
              `Rate limit! 재시도 중... ${
                retryCount + 1
              }회 (대기: ${waitTime}ms)`
            );
            await new Promise((res) => setTimeout(res, waitTime));
            retryCount++;
          } else {
            console.error("리포트 데이터를 불러오는 중 오류 발생:", error);
            break; // 다른 오류는 반복하지 않음
          }
        }
      }
      setLoading(false);
    };

    fetchReport();
  }, [isActive]);

  const maxValue = Math.max(...barData.map((item) => item.value), 5); // 최소값 5로 설정

  return (
    <div className="relative z-50">
      <div
        onClick={() => setIsActive(!isActive)}
        className={`bg-[#FFEEB8] rounded-[20px] pl-8 pt-6 w-full shadow-lg h-[200px]
             transform transition duration-300 ease-in-out
             cursor-pointer
             ${isActive ? "scale-110 shadow-2xl" : "scale-100 shadow-lg"}
             ${isActive ? "filter brightness-105" : ""}`}
      >
        <p className="fontSB text-[24px]">주간 리포트</p>
        <div className="fontSB text-[13px] text-[#555] mt-3">
          <p>AI와 함께</p>
          <p>한 주 간의 학습 상황을 분석해요</p>
        </div>

        <div className="flex justify-end -mt-6">
          <img
            src="/assets/images/MyPage/report.png"
            alt=""
            className="w-[45%] opacity-80"
          />
        </div>
      </div>

      {isActive && (
        <div className="absolute -left-10 top-[-200px] mt-0 -translate-x-full w-[500px] p-6 bg-white rounded-xl shadow-2xl z-50 transform transition duration-300 ease-in-out h-[600px] overflow-y-auto">
          <div className="space-y-6">
            {/* 헤더 */}
            <div className="text-center">
              <h3 className="text-xl font-bold text-gray-800">주간 리포트</h3>
              <p className="text-sm text-gray-600 mt-1">
                한 주 간의 학습 상황을 분석합니다
              </p>
            </div>

            {loading ? (
              <div className="flex flex-col items-center justify-center h-[430px]">
                <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-yellow-500"></div>
                <span className="mt-4 text-gray-600 text-lg">
                  데이터를 분석하는 중...
                </span>
              </div>
            ) : (
              <>
                {/* 바차트 */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="text-md font-semibold text-gray-700 mb-4 text-center">
                    요일별 학습 단어 수
                  </h4>
                  <div className="h-32">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={barData}
                        margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
                      >
                        <XAxis
                          dataKey="name"
                          axisLine={false}
                          tickLine={false}
                          tick={{ fontSize: 12, fill: "#666" }}
                        />
                        <YAxis hide />
                        <Bar
                          dataKey="value"
                          fill="#FFE694"
                          radius={[4, 4, 0, 0]}
                          maxBarSize={40}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* 총 학습 단어 수 표시 */}
                  <div className="text-center mt-3">
                    <span className="text-sm text-gray-600">
                      이번 주 총 학습 단어:
                    </span>
                    <span className="font-bold text-yellow-600 ml-1">
                      {barData.reduce((sum, item) => sum + item.value, 0)}개
                    </span>
                  </div>
                </div>

                {/* AI 분석 */}
                <div className="space-y-3">
                  <div className="text-left">
                    <span className="text-sm text-gray-600">
                      AI가 대신 해주는
                    </span>
                    <h4 className="text-lg font-bold text-gray-800">
                      이번주 학습 분석
                    </h4>
                  </div>

                  <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                    <p className="text-sm text-gray-700 leading-relaxed">
                      {reportData?.report ||
                        "분석 데이터를 불러오는 중입니다..."}
                    </p>
                  </div>
                </div>

                {/* 추가 통계 정보 */}
                {reportData && (
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-blue-50 rounded-lg p-3 text-center">
                      <p className="text-xs text-blue-600">가장 활발한 요일</p>
                      <p className="text-sm font-bold text-blue-800">
                        {barData.length > 0
                          ? barData.reduce((max, item) =>
                              item.value > max.value ? item : max
                            ).name + "요일"
                          : "-"}
                      </p>
                    </div>
                    <div className="bg-green-50 rounded-lg p-3 text-center">
                      <p className="text-xs text-green-600">일평균 학습량</p>
                      <p className="text-sm font-bold text-green-800">
                        {barData.length > 0
                          ? Math.round(
                              (barData.reduce(
                                (sum, item) => sum + item.value,
                                0
                              ) /
                                7) *
                                10
                            ) / 10
                          : 0}
                        개
                      </p>
                    </div>
                  </div>
                )}
              </>
            )}

            {/* 안내 메시지 */}
            <div className="text-center text-xs text-gray-500 pt-2 border-t border-gray-200">
              💡 매주 월요일에 지난주 학습 데이터를 기반으로 분석됩니다
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
