import { Routes, Route } from "react-router-dom";
import AdminMain from "../pages/Admin/Main/AdminMain";
import AdminNoGroup from "../pages/Admin/Group/AdminNoGroup";
import AdminGroup from "../pages/Admin/Group/AdminGroup";
import AdminGroupAdd from "../pages/Admin/Group/AdminGroupAdd";
import ClassList from "../pages/Admin/Class/ClassList";
import ClassMenu from "../pages/Admin/Class/ClassMenu";
import Curri_Part from "../pages/Admin/Curriculum/Curri_Part";
import Curri_Step from "../pages/Admin/Curriculum/Curri_Step";
import AdminStudent from '../pages/Admin/Student/AdminStudent';

export default function Admin() {
  return (
    <Routes>
      <Route path="/" element={<AdminMain />} />
      <Route path="/nogroup" element={<AdminNoGroup />} />
      <Route path="/group/add" element={<AdminGroupAdd />} />
      <Route path="/ClassList" element={<ClassList />} />
      <Route path="/ClassMenu/:code" element={<ClassMenu />} />
      <Route path="/Curri_Part/:code" element={<Curri_Part />} />
      <Route path="/curri/step" element={<Curri_Step />} />
      <Route path="/group/:code" element={<AdminGroup />} />
      <Route path="/student/:code" element={<AdminStudent />} />
    </Routes>
  );
}
