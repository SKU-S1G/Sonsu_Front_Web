import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import AdminTitle from "../../../components/AdminTitle";
import AdminNav from "../../../components/AdminNav";
import { IoCheckbox } from "react-icons/io5";
import { BiEditAlt } from "react-icons/bi";
import { IoPersonAdd, IoCopyOutline } from "react-icons/io5";
import { API_URL } from "../../../config";
import axios from "axios";

const colors = [
  "#DEE6F1",
  "#F2F3ED",
  "#F1E3D8",
  "#F4EBCE",
  "#DEE8D0",
  "#F0E4F8",
  "#D9D9D9",
];

const peopleImages = [
  "공준석.png",
  "김정이.png",
  "노태경.png",
  "이호연.png",
  "장원석.png",
  "최유정.png",
];

const getRandomPhoto = () => {
  const randomIndex = Math.floor(Math.random() * peopleImages.length);
  return `/assets/images/peoples/${peopleImages[randomIndex]}`;
};

// ================= SearchModal =================
function SearchModal({
  users,
  filteredUsers,
  handleAddStudents,
  setSearchModalOpen,
  search,
  setSearch,
  clsStudentsIds,
}) {
  const [modalSelected, setModalSelected] = useState([]);

  const toggleSelect = (id) => {
    setModalSelected((prevSelected) => {
      const newSelected = prevSelected.includes(id)
        ? prevSelected.filter((s) => s !== id)
        : [...prevSelected, id];
      return newSelected;
    });
  };

  const handleAddClick = () => {
    handleAddStudents(modalSelected);
    setModalSelected([]); // 추가 후 선택 초기화
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-2xl p-8 w-[500px] flex flex-col">
        {/* 검색 input */}
        <input
          type="text"
          placeholder="학생 이름 검색"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="px-4 py-2 mb-4 border rounded-lg"
        />

        {/* 검색 결과 */}
        <div className="flex flex-col max-h-[400px] overflow-y-auto">
          {filteredUsers.length === 0 ? (
            <div className="py-4 text-center text-gray-500">
              검색 결과가 없습니다.
            </div>
          ) : (
            filteredUsers.map((user, index) => {
              // user.id가 없거나 중복될 경우를 대비해 고유한 키 생성
              const uniqueKey = `search-modal-user-${
                user.id || "no-id"
              }-${index}`;
              const userId = user.id;

              const alreadyAdded = clsStudentsIds.includes(userId);
              const isSelected = modalSelected.includes(userId);

              return (
                <div
                  key={uniqueKey}
                  className={`flex items-center justify-between px-6 py-3 my-1 border rounded-lg ${
                    alreadyAdded
                      ? "opacity-50 bg-gray-100"
                      : isSelected
                      ? "bg-blue-50 border-blue-300"
                      : "hover:bg-gray-100"
                  }`}
                >
                  <span className="text-[18px]">
                    {user.name || "이름 없음"}
                    {alreadyAdded && (
                      <span className="ml-2 text-sm text-gray-500">
                        (이미 추가됨)
                      </span>
                    )}
                  </span>
                  <IoCheckbox
                    size={22}
                    className={`cursor-pointer ${
                      alreadyAdded
                        ? "text-gray-400"
                        : isSelected
                        ? "text-[#5A9CD0]"
                        : "text-[#aaa]"
                    }`}
                    onClick={() => {
                      if (!alreadyAdded && userId) {
                        toggleSelect(userId);
                      } else {
                      }
                    }}
                  />
                </div>
              );
            })
          )}
        </div>

        {/* 선택된 학생 수 표시 */}
        {modalSelected.length > 0 && (
          <div className="p-2 mt-4 text-center rounded-lg bg-blue-50">
            <span className="text-sm text-blue-700">
              {modalSelected.length}명의 학생이 선택되었습니다.
            </span>
          </div>
        )}

        {/* 버튼 */}
        <div className="flex justify-end gap-4 mt-6">
          <button
            className="px-4 py-2 border border-gray-400 rounded-lg hover:bg-gray-50"
            onClick={() => {
              setSearchModalOpen(false);
              setModalSelected([]); // 모달 닫을 때 선택 초기화
            }}
          >
            닫기
          </button>
          <button
            className={`px-4 py-2 rounded-lg text-white ${
              modalSelected.length > 0
                ? "bg-[#5A9CD0] hover:bg-[#4A8BC0]"
                : "bg-gray-400 cursor-not-allowed"
            }`}
            onClick={handleAddClick}
            disabled={modalSelected.length === 0}
          >
            추가하기 ({modalSelected.length})
          </button>
        </div>
      </div>
    </div>
  );
}

// ================= 그룹 수정 모달 =================
function EditModal({
  cls,
  groupName,
  groupDesc,
  selectedColor,
  setGroupName,
  setGroupDesc,
  setSelectedColor,
  setCls,
  setEditModalOpen,
}) {
  const handleSave = async () => {
    try {
      const token = localStorage.getItem("accessToken");
      const body = {
        className: groupName,
        title: groupName,
        description: groupDesc,
        colorId: colors.indexOf(selectedColor) + 1,
      };
      await axios.patch(`${API_URL}/class/edit/${cls.id}`, body, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setCls((prev) => ({
        ...prev,
        name: groupName,
        desc: groupDesc,
        color: selectedColor,
      }));
      alert("그룹이 성공적으로 수정되었습니다.");
      setEditModalOpen(false);
    } catch (error) {
      console.error(error);
      alert("그룹 수정에 실패했습니다.");
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="flex flex-col justify-between py-32 bg-white rounded-2xl w-[40%] px-28 h-[80%]">
        <h2 className="flex justify-center text-[22px] fontSB mb-6">
          그룹 수정하기
        </h2>

        <div className="flex flex-col">
          <label className="text-[18px] fontMedium">
            그룹 이름<span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            value={groupName}
            onChange={(e) => setGroupName(e.target.value)}
            className="bg-transparent mt-5 px-4 py-3 border-[1.5px] border-[#555552] rounded-xl focus:outline-none focus:border-[#5A9CD0] mb-6"
          />
        </div>

        <div className="flex flex-col justify-between">
          <label className="text-[18px] fontMedium">부가 설명</label>
          <input
            type="text"
            value={groupDesc}
            onChange={(e) => setGroupDesc(e.target.value)}
            className="bg-transparent mt-5 px-4 py-3 border-[1.5px] border-[#555552] rounded-xl focus:outline-none focus:border-[#5A9CD0] mb-6"
          />
        </div>

        <div className="flex flex-col justify-between">
          <label className="text-[18px] fontMedium">
            그룹 색상<span className="text-red-500">*</span>
          </label>
          <div className="flex flex-wrap gap-4 mt-5 mb-6">
            {colors.map((color) => (
              <div
                key={color}
                onClick={() => setSelectedColor(color)}
                className={`w-[40px] h-[40px] rounded-full cursor-pointer`}
                style={{
                  backgroundColor: color,
                  border:
                    selectedColor === color ? "3px solid #5A9CD0" : "none",
                }}
              />
            ))}
          </div>
        </div>

        <div className="flex items-center mt-5">
          <div className="flex text-[18px] text-[#333] fontMedium">
            그룹 코드
          </div>
          <div className="flex ml-8 text-[#666]">#{cls.code}</div>
        </div>

        <div className="flex justify-end w-full gap-10 mt-10">
          <div
            className="flex text-[18px] text-[#777] fontSB cursor-pointer"
            onClick={() => setEditModalOpen(false)}
          >
            취소
          </div>
          <div
            className="flex text-[18px] text-[#5A9CD0] fontSB cursor-pointer"
            onClick={handleSave}
          >
            저장
          </div>
        </div>
      </div>
    </div>
  );
}

// ================= AdminGroup =================
export default function AdminGroup() {
  const { code } = useParams();
  const [cls, setCls] = useState(null);
  const [groupName, setGroupName] = useState("");
  const [groupDesc, setGroupDesc] = useState("");
  const [selectedColor, setSelectedColor] = useState("");
  const [selected, setSelected] = useState([]);
  const [searchModalOpen, setSearchModalOpen] = useState(false);
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [users, setUsers] = useState([]);
  const [search, setSearch] = useState("");

  // ================= 클래스 및 학생 초기 조회 =================
  useEffect(() => {
    const fetchClass = async () => {
      try {
        const token = localStorage.getItem("accessToken");
        const res = await axios.get(`${API_URL}/class/${code}/select`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = res.data.data;

        const userRes = await axios.get(
          `${API_URL}/class/${data.class_id}/users`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
        const students = userRes.data.users.map((u) => ({
          id: u.member_id,
          name: u.username,
          photo: getRandomPhoto(),
        }));

        setCls({
          id: data.class_id,
          name: data.class_name,
          desc: data.description,
          code: data.class_code,
          color: data.color_hex,
          students,
        });
        setSelectedColor(data.color_hex);
        setGroupName(data.class_name);
        setGroupDesc(data.description);
        setSelectedColor(data.color_hex);
      } catch (err) {
        alert("클래스 조회에 실패했습니다.");
      }
    };

    fetchClass();
  }, [code]);

  // ================= 개별 선택 =================
  const selectSingle = (id) => {
    setSelected((prevSelected) => {
      if (prevSelected.includes(id)) {
        return prevSelected.filter((s) => s !== id);
      } else {
        return [...prevSelected, id];
      }
    });
  };

  // ================= 전체 선택 =================
  const selectAll = () => {
    if (!cls) return;
    setSelected((prevSelected) => {
      if (prevSelected.length === cls.students.length) {
        return [];
      } else {
        return cls.students.map((s) => s.id);
      }
    });
  };

  // ================= 학생 삭제 =================
  const handleDeleteStudents = async () => {
    if (!cls || selected.length === 0)
      return alert("삭제할 학생을 선택해주세요.");
    if (!window.confirm("선택된 학생을 삭제하시겠습니까?")) return;

    try {
      const token = localStorage.getItem("accessToken");
      await axios.delete(`${API_URL}/class/${cls.id}/delUsers`, {
        headers: { Authorization: `Bearer ${token}` },
        data: { memberIds: selected },
      });
      setCls((prev) => ({
        ...prev,
        students: prev.students.filter((s) => !selected.includes(s.id)),
      }));
      setSelected([]);
      alert("선택한 학생이 삭제되었습니다.");
    } catch (err) {
      alert("학생 삭제에 실패했습니다.");
    }
  };

  // ================= 검색 모달용 전체 유저 조회 =================
  useEffect(() => {
    if (!searchModalOpen) return;
    const fetchUsers = async () => {
      try {
        const token = localStorage.getItem("accessToken");
        const res = await axios.get(`${API_URL}/class/user`, {
          headers: { Authorization: `Bearer ${token}` },
        });

        const processedUsers = res.data.users.map((u, index) => {
          return {
            id: u.member_id || u.id || u.user_id, // 다양한 가능성 시도
            name: u.username || u.name || "이름 없음",
            photo: getRandomPhoto(),
          };
        });

        setUsers(processedUsers);
      } catch (err) {
        console.error(err);
        alert("유저 목록 불러오기에 실패했습니다.");
      }
    };
    fetchUsers();
  }, [searchModalOpen]);

  // ================= 학생 추가 =================
  const handleAddStudents = async (selectedIds) => {
    if (!cls || selectedIds.length === 0)
      return alert("추가할 학생을 선택해주세요.");

    try {
      const token = localStorage.getItem("accessToken");
      await axios.post(
        `${API_URL}/class/${cls.id}/invite`,
        { memberIds: selectedIds },
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );

      const newStudents = users
        .filter((u) => selectedIds.includes(u.id))
        .map((u) => ({ ...u, photo: getRandomPhoto() }));

      setCls((prev) => ({
        ...prev,
        students: [...prev.students, ...newStudents],
      }));
      setSearchModalOpen(false);
      alert("성공적으로 멤버가 추가되었습니다.");
    } catch (err) {
      console.error(err);
      alert("멤버 추가에 실패했습니다.");
    }
  };

  // ================= 검색 필터 =================
  const filteredUsers = users.filter((u) =>
    (u.name || "").toLowerCase().includes(search.toLowerCase())
  );

  // ================= 그룹 ID 복사 =================
  const copyGroupId = () => {
    if (!cls) return;
    navigator.clipboard
      .writeText(`#${cls.code}`)
      .then(() => alert("그룹 ID가 복사되었습니다!"))
      .catch(console.error);
  };

  if (!cls)
    return (
      <div className="flex items-center justify-center min-h-screen">
        클래스를 찾을 수 없습니다.
      </div>
    );

  return (
    <div className="min-h-screen bg-[#5A9CD0]">
      <AdminTitle />
      <div className="flex w-full">
        <AdminNav />

        <div className="flex flex-col mr-10 w-full rounded-3xl bg-[#fafafa] h-[850px] px-10 py-8">
          {/* 그룹 헤더 */}
          <div className="flex flex-col w-[90%] pt-10 pb-6 h-fit border-b-[4px] border-[#5A9CD0]">
            <div className="flex items-end">
              <div className="flex text-[25px] fontSB">{cls.name}</div>
              <BiEditAlt
                size="17"
                className="mb-2 ml-2 cursor-pointer"
                onClick={() => setEditModalOpen(true)}
              />
            </div>
            <div className="flex items-center justify-between w-full mt-3">
              <div className="flex items-center">
                <div className="flex text-[20px] text-[#777] fontSB">
                  #{cls.code}
                </div>
                <div
                  className="flex text-[20px] text-[#777] mx-1 fontSB cursor-pointer"
                  onClick={copyGroupId}
                >
                  <IoCopyOutline />
                </div>
                <div
                  className="flex ml-3 w-[30px] h-[30px] rounded-2xl"
                  style={{ backgroundColor: cls.color }}
                ></div>
              </div>
              <IoPersonAdd
                size="22"
                className="cursor-pointer"
                onClick={() => setSearchModalOpen(true)}
              />
            </div>

            <p className="mt-2 text-sm text-gray-600">{cls.desc}</p>
          </div>

          {/* 학생 리스트 */}
          {cls.students.length === 0 ? (
            <div className="flex flex-col items-center justify-center w-full mt-32">
              <img
                src="/assets/images/Admin/Member/group.png"
                alt="group"
                className="w-[400px] h-fit"
              />
              <div className="text-[20px] fontMedium my-5">
                수강생이 없다면 추가해주세요!
              </div>
              <div
                className="text-[20px] fontSB px-4 py-3 rounded-2xl bg-[#E7E7E7] cursor-pointer"
                onClick={() => setSearchModalOpen(true)}
              >
                수강생 추가하기
              </div>
            </div>
          ) : (
            <div className="flex flex-wrap my-2 justify-start mt-6 w-[80%] gap-2">
              {cls.students.map((student) => (
                <div
                  key={student.id}
                  className="flex justify-center items-center my-5 p-4 w-[32%]"
                >
                  <img
                    src={student.photo}
                    alt={student.name}
                    className="w-16 h-16 rounded-full"
                  />
                  <span className="text-[22px] ml-6 mr-10 fontSB">
                    {student.name}
                  </span>
                  <IoCheckbox
                    size={25}
                    className={`cursor-pointer ${
                      selected.includes(student.id)
                        ? "text-[#5A9CD0]"
                        : "text-[#888]"
                    }`}
                    onClick={() => selectSingle(student.id)}
                  />
                </div>
              ))}
            </div>
          )}

          {/* 전체선택 / 삭제 */}
          {cls.students.length > 0 && (
            <div className="flex w-[90%] h-full pb-12 items-end justify-end">
              <div
                className="flex text-[18px] text-[#777] fontSB cursor-pointer"
                onClick={selectAll}
              >
                {selected.length === cls.students.length
                  ? "전체해제"
                  : "전체선택"}
              </div>
              <div
                className="flex text-[18px] text-[#5A9CD0] fontSB ml-10 cursor-pointer"
                onClick={handleDeleteStudents}
              >
                삭제 {selected.length > 0 && `(${selected.length})`}
              </div>
            </div>
          )}
        </div>
      </div>

      {searchModalOpen && (
        <SearchModal
          users={users}
          filteredUsers={filteredUsers}
          handleAddStudents={handleAddStudents}
          setSearchModalOpen={setSearchModalOpen}
          search={search}
          setSearch={setSearch}
          clsStudentsIds={cls.students.map((s) => s.id)}
        />
      )}

      {editModalOpen && (
        <EditModal
          cls={cls}
          groupName={groupName}
          groupDesc={groupDesc}
          selectedColor={selectedColor}
          setGroupName={setGroupName}
          setGroupDesc={setGroupDesc}
          setSelectedColor={setSelectedColor}
          setCls={setCls}
          setEditModalOpen={setEditModalOpen}
        />
      )}
    </div>
  );
}
