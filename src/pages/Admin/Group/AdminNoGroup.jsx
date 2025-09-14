import React from 'react'
import { useNavigate } from "react-router-dom";
import AdminTitle from '../../../components/AdminTitle';
import AdminNav from '../../../components/AdminNav';

export default function AdminNoGroup() {
  const navigate = useNavigate();

  return (
    <div className='min-h-screen bg-[#5A9CD0]'>
      <AdminTitle/>

      <div className='flex w-full'>
        <AdminNav />

        <div className='flex mr-10 w-full my-10 rounded-3xl bg-[#fafafa] h-[930px]'>
          <div className='flex flex-col items-center justify-center w-full'>
            <img
              src="/assets/images/Admin/Member/group.png"
              alt="group"
              className="flex w-[400px] h-fit"
            />

            <div className='text-[20px] fontMedium my-5'>수강그룹이 없다면 생성해주세요!</div>

            <div
              className='text-[20px] fontSB px-4 py-3 rounded-2xl bg-[#E7E7E7] cursor-pointer'
              onClick={() => navigate("/admin/group/add")}
            >
              수강 그룹 생성하기
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
