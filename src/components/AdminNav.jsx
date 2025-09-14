import React, { useState, useEffect } from "react";
import { useNavigate, useLocation, useParams } from "react-router-dom";
import { FaUserCircle } from "react-icons/fa";
import axios from "axios";
import { API_URL } from "../config";
import { getToken, removeToken } from "../utils/authStorage";

export default function AdminNav() {
  const navigate = useNavigate();
  const location = useLocation();
  const { code } = useParams(); // 여기서 code 가져오기
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [hoverUserMenu, setHoverUserMenu] = useState(false);
  const [userInfo, setUserInfo] = useState({});

  const menus = [
    { name: "커리큘럼 관리", path: "/admin/ClassList" },
    { name: "수강그룹 관리", path: `/admin/group/${code}` },
    { name: "수강생 관리", path: `/admin/student/${code}` },
  ];

  const userMenuItems = ["프로필 설정", "수업 참여하기", "손수잇다", "로그아웃"];

  const [selected, setSelected] = useState("");

  useEffect(() => {
    const fetchUserInfo = async () => {
      try {
        const token = getToken();
        if (!token) return;

        const response = await axios.get(`${API_URL}/login/success`, {
          headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
          withCredentials: true,
        });

        if (response.data) setUserInfo(response.data);
      } catch (error) {
        console.error("사용자 정보 가져오기 실패:", error.response?.data);
        if (error.response?.status === 401) navigate("/login");
      }
    };

    fetchUserInfo();
  }, [navigate]);

  // URL 변경에 따라 selected 업데이트
  useEffect(() => {
    const currentMenu = menus.find((menu) => menu.path === location.pathname);
    setSelected(currentMenu ? currentMenu.name : "");
  }, [location.pathname, code]); // code가 바뀌면 다시 체크

  const handleClick = (menu) => {
    setSelected(menu.name);
    navigate(menu.path);
  };

  const handleUserMenuClick = async (item) => {
    if (item === "로그아웃") {
      if (!window.confirm("정말 로그아웃 하시겠습니까?")) return;

      try {
        const token = getToken();
        await axios.post(
          `${API_URL}/logout`,
          {},
          token
            ? { headers: { Authorization: `Bearer ${token}` }, withCredentials: true }
            : { withCredentials: true }
        );
      } catch (error) {
        console.error("로그아웃 중 오류 발생:", error);
      } finally {
        removeToken();
        setUserInfo(null);
        navigate("/");
      }
    } else if (item === "손수잇다") {
      navigate("/");
    } else {
      console.log(item + " 클릭됨");
    }
  };


  return (
    <div className="flex flex-col w-[16%] mx-14 ">
      <div className="flex flex-col my-5">
        <img
          src="/assets/images/Admin/Member/group.png"
          alt="group"
          className="flex w-[120px]"
          onClick={() => navigate("/admin")}
        />

        {menus.map((menu) => (
          <div
            key={menu.name}
            className={`flex my-5 fontBold cursor-pointer ${
              selected === menu.name
                ? "text-white text-[23px]"
                : "text-[#333] text-[20px]"
            }`}
            onClick={() => handleClick(menu)}
          >
            {menu.name}
          </div>
        ))}
      </div>

      {/* User 아이콘 및 메뉴 */}
      <div className="relative flex items-end h-full pb-3">
        <div
          onMouseEnter={() => setHoverUserMenu(true)}
          onMouseLeave={() => setHoverUserMenu(false)}
          className="relative"
        >
          <FaUserCircle
            size={53}
            className="text-[#555552] rounded-3xl cursor-pointer z-50 relative"
            onClick={() => setShowUserMenu((prev) => !prev)}
          />

          {(showUserMenu || hoverUserMenu) && (
            <div className="absolute z-40 flex flex-col px-4 pt-3 pb-5 bg-white shadow-lg bottom-9 left-6 w-52 rounded-3xl">
              <div className="flex justify-center my-3 text-[17px] text-[#555] items-end fontRegular">
                <div className="text-[#FFCA1F] fontSB text-[20px] mr-1">
                  {userInfo?.username || "undefined"}
                </div>
                님, 안녕하세요!
              </div>
              {userMenuItems.map((item) => (
                <div
                  key={item}
                  className="px-4 py-2 hover:bg-gray-100 cursor-pointer rounded-lg text-[#000] font-thin"
                  onClick={() => handleUserMenuClick(item)}
                >
                  {item}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
