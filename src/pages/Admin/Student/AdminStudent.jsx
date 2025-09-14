import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import AdminTitle from '../../../components/AdminTitle';
import AdminNav from '../../../components/AdminNav';
import { IoCheckbox } from "react-icons/io5";
import { BiEditAlt } from "react-icons/bi";
import { IoPersonAdd, IoCopyOutline } from "react-icons/io5";
import { API_URL } from '../../../config';
import axios from 'axios';

const colors = [
  "#DEE6F1",
  "#F2F3ED",
  "#F1E3D8",
  "#F4EBCE",
  "#DEE8D0",
  "#F0E4F8",
  "#D9D9D9",
];

export default function AdminGroup() {
  const { code } = useParams();
  const [cls, setCls] = useState(null);
  const [groupName, setGroupName] = useState("");
  const [groupDesc, setGroupDesc] = useState("");
  const [selectedColor, setSelectedColor] = useState("");
  const [selected, setSelected] = useState([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [editModalOpen, setEditModalOpen] = useState(false);

  useEffect(() => {
    const fetchClass = async () => {
      try {
        const token = localStorage.getItem("accessToken");

        const res = await axios.get(`${API_URL}/class/${code}/select`, {
          headers: { Authorization: `Bearer ${token}` },
        });

        const data = res.data.data;

        const classData = {
          id: data.class_id,
          name: data.class_name,
          desc: data.description,
          code: data.class_code,
          color: colors[(data.color_id || 1) - 1],
          students: data.students || [], // 백에서 학생 목록 받는 경우
        };

        setCls(classData);
        setGroupName(classData.name);
        setGroupDesc(classData.desc);
        setSelectedColor(classData.color);
      } catch (error) {
        console.error(error);
        alert("클래스 조회에 실패했습니다.");
      }
    };

    fetchClass();
  }, [code]);

  const toggleSelect = (name) => {
    setSelected(prev =>
      prev.includes(name)
        ? prev.filter(n => n !== name)
        : [...prev, name]
    );
  };

  const selectAll = () => {
    if (selected.length === cls.students.length) {
      setSelected([]);
    } else {
      setSelected(cls.students.map(s => s.name));
    }
  };

  const copyGroupId = () => {
    navigator.clipboard.writeText(`#${cls.code}`)
      .then(() => alert("그룹 ID가 복사되었습니다!"))
      .catch((err) => console.error("복사 실패:", err));
  };

  const handleInvite = () => {
    navigator.clipboard.writeText(`#${cls.code}`)
      .then(() => alert("초대코드가 복사되었습니다!"))
      .catch((err) => console.error("복사 실패:", err));
  };

  if (!cls) {
    return <div className='flex items-center justify-center min-h-screen'>클래스를 찾을 수 없습니다.</div>;
  }

  return (
    <div className='min-h-screen bg-[#5A9CD0]'>
      <AdminTitle />

      <div className='flex w-full'>
        <AdminNav />

        <div className='flex flex-col items-center mr-10 w-full my-10 rounded-3xl bg-[#fafafa] min-h-[930px]'>
          {/* 그룹 이름 */}
          <div className='flex flex-col w-[90%] pt-10 pb-6 h-fit border-b-[4px] border-[#5A9CD0]'>
            <div className='flex items-end'>
              <div className='flex text-[25px] fontSB'>{cls.name}</div>
              <BiEditAlt
                size='17'
                className='mb-2 ml-2 cursor-pointer'
                onClick={() => setEditModalOpen(true)}
              />
            </div>
            <div className='flex items-center justify-between w-full mt-3'>
              <div className='flex items-center'>
                <div className='flex text-[20px] text-[#777] fontSB'>#{cls.code}</div>
                <div className='flex text-[20px] text-[#777] mx-1 fontSB cursor-pointer' onClick={copyGroupId}>
                  <IoCopyOutline />
                </div>
                <div
                  className='flex ml-3 w-[30px] h-[30px] rounded-2xl'
                  style={{ backgroundColor: cls.color }}
                ></div>
              </div>

              <IoPersonAdd 
                size='22' 
                className="cursor-pointer"
                onClick={() => setModalOpen(true)} 
              />
            </div>
          </div>

          {/* 수강생이 없는 경우 */}
          {cls.students.length === 0 ? (
            <div className='flex flex-col items-center justify-center w-full mt-32'>
              <img
                src="/assets/images/Admin/Member/group.png"
                alt="group"
                className="w-[400px] h-fit"
              />

              <div className='text-[20px] fontMedium my-5'>
                수강생이 없다면 추가해주세요!
              </div>

              <div
                className='text-[20px] fontSB px-4 py-3 rounded-2xl bg-[#E7E7E7] cursor-pointer'
                onClick={() => setModalOpen(true)}
              >
                수강생 추가하기
              </div>
            </div>
          ) : (
            <div className='flex flex-wrap my-2 justify-center mt-6 w-[80%] gap-2'>
              {cls.students.map((student) => (
                <div
                  key={student.id}
                  className='flex justify-center items-center my-5 p-4 w-[32%]'
                >
                  <img
                    src={student.photo}
                    alt={student.name}
                    className='w-16 h-16 rounded-full'
                  />
                  <span className='text-[22px] ml-6 mr-10 fontSB'>{student.name}</span>
                  <IoCheckbox
                    size={25}
                    className={`cursor-pointer ${selected.includes(student.name) ? 'text-[#5A9CD0]' : 'text-[#888]'}`}
                    onClick={() => toggleSelect(student.name)}
                  />
                </div>
              ))}
            </div>
          )}

          {/* 전체선택 / 삭제 */}
          {cls.students.length > 0 && (
            <div className='flex w-[90%] h-full pb-12 items-end justify-end'>
              <div className='flex text-[18px] text-[#777] fontSB cursor-pointer' onClick={selectAll}>
                전체선택
              </div>
              <div className='flex text-[18px] text-[#5A9CD0] fontSB ml-10'>삭제</div>
            </div>
          )}
        </div>
      </div>

      {/* 초대 모달 */}
      {modalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-white rounded-2xl p-8 w-[400px] flex flex-col items-center">
            <h2 className="text-[22px] fontSB mb-4">수강생 초대</h2>
            <p className="mb-4 text-center">아래 초대코드를 복사하여 수강생을 초대해주세요!</p>
            <div className="flex items-center mb-6">
              <input
                type="text"
                readOnly
                value={`#${cls.code}`}
                className="w-64 px-4 py-2 border rounded-l-2xl"
              />
              <button
                className="bg-[#5A9CD0] text-white px-4 py-2 rounded-r-2xl"
                onClick={handleInvite}
              >
                복사
              </button>
            </div>
            <button
              className="mt-2 text-[#777] px-4 py-2 rounded-2xl border border-[#ccc]"
              onClick={() => setModalOpen(false)}
            >
              닫기
            </button>
          </div>
        </div>
      )}

      {/* 그룹 수정 모달 */}
      {editModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="flex flex-col justify-between py-32 bg-white rounded-2xl w-[40%] px-28 h-[80%]">
            <h2 className="flex justify-center text-[22px] fontSB mb-6">그룹 수정하기</h2>

            <div className='flex flex-col'>
              <label className='text-[18px] fontMedium'>그룹 이름<span className='text-red-500'>*</span></label>
              <input
                type="text"
                value={groupName}
                onChange={(e) => setGroupName(e.target.value)}
                className="bg-transparent mt-5 px-4 py-3 border-[1.5px] border-[#555552] rounded-xl focus:outline-none focus:border-[#5A9CD0] mb-6"
              />
            </div>

            <div className='flex flex-col justify-between'>
              <label className='text-[18px] fontMedium'>부가 설명</label>
              <input
                type="text"
                value={groupDesc}
                onChange={(e) => setGroupDesc(e.target.value)}
                className="bg-transparent mt-5 px-4 py-3 border-[1.5px] border-[#555552] rounded-xl focus:outline-none focus:border-[#5A9CD0] mb-6"
              />
            </div>

            <div className='flex flex-col justify-between'>
              <label className='text-[18px] fontMedium'>그룹 색상<span className='text-red-500'>*</span></label>
              <div className="flex flex-wrap gap-4 mt-5 mb-6">
                {colors.map((color) => (
                  <div
                    key={color}
                    onClick={() => setSelectedColor(color)}
                    className={`w-[40px] h-[40px] rounded-full cursor-pointer`}
                    style={{
                      backgroundColor: color,
                      border: selectedColor === color ? "3px solid #5A9CD0" : "none",
                    }}
                  />
                ))}
              </div>
            </div>

            <div className='flex items-center mt-5'>
              <div className='flex text-[18px] text-[#333] fontMedium'>
                그룹 코드
              </div>
              <div className='flex ml-8 text-[#666]'>#{cls.code}</div>
            </div>

            <div className='flex justify-end w-full gap-10 mt-10'>
              <div
                className='flex text-[18px] text-[#777] fontSB cursor-pointer'
                onClick={() => setEditModalOpen(false)}
              >
                취소
              </div>
              <div
                className='flex text-[18px] text-[#5A9CD0] fontSB cursor-pointer'
                onClick={async () => {
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

                    setCls(prev => ({
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
                }}
              >
                저장
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
