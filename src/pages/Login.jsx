import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { API_URL } from "../config";
import { saveToken } from "../utils/authStorage";

const AuthForm = ({ title, onSubmit, children }) => (
  <form
    onSubmit={onSubmit}
    className="flex flex-col justify-center w-full h-full px-20 text-center bg-white"
  >
    <h1 className="font-bold text-4xl mb-4 text-[#333333]">{title}</h1>
    {children}
  </form>
);

const OverlayPanel = ({ title, buttonText, onClick, alignment, textAlign }) => (
  <div
    className={`absolute top-0 space-y-16 h-full w-1/2 flex flex-col justify-center px-20 transition-transform duration-700 ease-in-out ${alignment} ${textAlign}`}
  >
    <div className="flex items-center">
      <img
        src="assets/images/logo.png"
        alt="손수잇다 로고"
        className="w-20 h-20 mr-5"
      />
      <h1 className="font-bold text-3xl text-[#333333]">손수잇다</h1>
    </div>
    <p className="font-medium text-[25px] text-[#333333]">
      수어를 쉽고, 재미있게!
    </p>
    <div className="text-[#4F4F4F] text-[18px] leading-relaxed">
      <p>손수잇다는 3D 아바타의 수어 동작 애니메이션 게임과</p>
      <p>같은 재미 요소를 활용하여 비장애인이 보다 쉽고 재미있게</p>
      <p>학습할 수 있도록 돕는 서비스입니다.</p>
    </div>
    <div className="flex flex-col items-center justify-center w-full mt-5 text-center">
      <p className="font-medium mb-3 text-[#333333]">{title}</p>
      <button
        type="button"
        onClick={onClick}
        className="bg-[#333333] w-[60%] text-white font-bold py-3 px-12 rounded-full transition-transform duration-100 ease-in active:scale-95"
      >
        {buttonText}
      </button>
    </div>
  </div>
);

const Login = () => {
  const [isSignUp, setIsSignUp] = useState(false);
  const [loginId, setLoginId] = useState("");
  const [password, setPassword] = useState("");
  const [signUpData, setSignUpData] = useState({
    username: "",
    loginId: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  const navigate = useNavigate();

  // 로그인 API
  const handleSignIn = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post(
        `${API_URL}/login`,
        { loginId, password },
        { withCredentials: true }
      );

      if (response.status === 200) {
        console.log("로그인 성공:", response.data);

        // 🔥 중요: 토큰을 localStorage에 저장 (주석 해제 및 수정)
        if (response.data.accessToken) {
          saveToken(response.data.accessToken);
          console.log(
            "토큰이 localStorage에 저장되었습니다:",
            response.data.accessToken.substring(0, 20) + "..."
          );
        }

        console.log("로그인한 유저:", response.data.userInfo);
        alert("로그인 성공!");
        navigate("/home"); // 홈 페이지로 이동 (경로 수정)
      }
    } catch (error) {
      console.error("로그인 실패:", error.response || error.message);

      // 더 구체적인 에러 메시지
      if (error.response?.status === 401) {
        alert("로그인 실패: 잘못된 아이디 또는 비밀번호입니다.");
      } else {
        alert("로그인에 실패했습니다. 다시 시도해주세요.");
      }
    }
  };

  // 회원가입 API (추후 구현)
  const handleSignUp = async (e) => {
    e.preventDefault();

    // 비밀번호 확인 체크
    if (signUpData.password !== signUpData.confirmPassword) {
      alert("비밀번호와 비밀번호 확인이 일치하지 않습니다.");
      return;
    }

    try {
      const response = await axios.post(
        `${API_URL}/register`,
        {
          username: signUpData.username,
          loginId: signUpData.loginId,
          email: signUpData.email,
          password: signUpData.password,
        },
        { withCredentials: true } // 쿠키 기반 인증 필요시
      );

      console.log("회원가입 성공:", response.data);
      alert("회원가입 성공! 로그인 페이지로 이동합니다.");
      setIsSignUp(false); // 회원가입 후 로그인 화면으로 이동
    } catch (error) {
      console.error("회원가입 실패:", error.response?.data || error.message);
      alert(
        `회원가입 실패: ${
          error.response?.data?.message || "서버와 통신 중 오류가 발생했습니다."
        }`
      );
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen font-sans bg-gray-100">
      <div className="relative w-[1280px] h-[720px] bg-white shadow-2xl rounded-3xl overflow-hidden">
        {/* Sign-Up Form */}
        <div
          className={`absolute top-0 h-full left-0 w-1/2 transition-all duration-700 ease-in-out ${
            isSignUp
              ? "translate-x-full opacity-100 z-[50]"
              : "opacity-0 z-[10] pointer-events-none"
          }`}
        >
          <AuthForm title="SIGN UP" onSubmit={handleSignUp}>
            <input
              type="text"
              placeholder="이름"
              className="w-full p-3 my-2 bg-gray-100 border-none rounded-full shadow-inner"
              value={signUpData.username}
              onChange={(e) =>
                setSignUpData({ ...signUpData, username: e.target.value })
              }
            />
            <input
              type="text"
              placeholder="아이디"
              className="w-full p-3 my-2 bg-gray-100 border-none rounded-full shadow-inner"
              value={signUpData.loginId}
              onChange={(e) =>
                setSignUpData({ ...signUpData, loginId: e.target.value })
              }
            />
            <input
              type="email"
              placeholder="이메일"
              className="w-full p-3 my-2 bg-gray-100 border-none rounded-full shadow-inner"
              value={signUpData.email}
              onChange={(e) =>
                setSignUpData({ ...signUpData, email: e.target.value })
              }
            />
            <input
              type="password"
              placeholder="비밀번호"
              className="w-full p-3 my-2 bg-gray-100 border-none rounded-full shadow-inner"
              value={signUpData.password}
              onChange={(e) =>
                setSignUpData({ ...signUpData, password: e.target.value })
              }
            />
            <input
              type="password"
              placeholder="비밀번호 확인"
              className="w-full p-3 my-2 bg-gray-100 border-none rounded-full shadow-inner"
              value={signUpData.confirmPassword}
              onChange={(e) =>
                setSignUpData({
                  ...signUpData,
                  confirmPassword: e.target.value,
                })
              }
            />
            <button
              type="submit"
              className="bg-[#FFE694] text-[#333333] font-bold py-3 px-16 rounded-full mt-4 transition-transform duration-100 ease-in active:scale-95"
            >
              회원가입
            </button>
          </AuthForm>
        </div>

        {/* Sign-In Form */}
        <div
          className={`absolute top-0 h-full left-0 w-1/2 transition-all duration-700 ease-in-out z-[20] ${
            isSignUp ? "translate-x-full" : ""
          }`}
        >
          <AuthForm title="SIGN IN" onSubmit={handleSignIn}>
            <input
              type="text"
              placeholder="아이디"
              className="w-full p-3 my-2 bg-gray-100 border-none rounded-full shadow-inner"
              value={loginId}
              onChange={(e) => setLoginId(e.target.value)}
            />
            <input
              type="password"
              placeholder="비밀번호"
              className="w-full p-3 my-2 bg-gray-100 border-none rounded-full shadow-inner"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            <button
              type="submit"
              className="bg-[#FFE694] text-[#333333] font-bold py-3 px-16 rounded-full mt-6 transition-transform duration-100 ease-in active:scale-95"
            >
              로그인
            </button>
          </AuthForm>
        </div>

        {/* Overlay */}
        <div
          className={`absolute top-0 left-1/2 w-1/2 h-full overflow-hidden transition-transform duration-700 ease-in-out z-[100] ${
            isSignUp ? "-translate-x-full" : ""
          }`}
        >
          <div
            className={`relative -left-full h-full w-[200%] bg-[#FFE694] transition-transform duration-700 ease-in-out ${
              isSignUp ? "translate-x-1/2" : "translate-x-0"
            }`}
          >
            <OverlayPanel
              title="이미 회원이라면?"
              buttonText="로그인"
              onClick={() => setIsSignUp(false)}
              alignment={`transform transition-transform duration-700 ease-in-out ${
                isSignUp ? "translate-x-0" : "-translate-x-[20%]"
              }`}
              textAlign="items-start text-left"
            />
            <OverlayPanel
              title="아직 회원이 아니라면?"
              buttonText="회원가입"
              onClick={() => setIsSignUp(true)}
              alignment={`right-0 transform transition-transform duration-700 ease-in-out ${
                isSignUp ? "translate-x-[20%]" : "translate-x-0"
              }`}
              textAlign="items-end text-right"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;