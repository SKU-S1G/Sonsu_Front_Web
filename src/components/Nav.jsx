import React, { useState, useEffect, useRef } from "react";
import {
FaBars,
FaChalkboardTeacher,
FaClipboardList,
} from "react-icons/fa";
import { BsFillPersonFill } from "react-icons/bs";
import { PiTimerBold } from "react-icons/pi";
import { Link, NavLink } from "react-router-dom"; // Import NavLink from react-router-dom

export default function Nav({ toggleSidebar, isOpen }) {
const [isSidebarOpen, setIsSidebarOpen] = useState(isOpen);
const sidebarRef = useRef(null);

// Close the sidebar when clicking outside of it
useEffect(() => {
    const handleClickOutside = (event) => {
    if (sidebarRef.current && !sidebarRef.current.contains(event.target)) {
        setIsSidebarOpen(false); // Close sidebar when clicking outside
    }
    };

    document.addEventListener("mousedown", handleClickOutside);

    return () => {
    document.removeEventListener("mousedown", handleClickOutside);
    };
}, []);

return (
    <div
    ref={sidebarRef} // Reference for detecting clicks outside
    className={`${
        isSidebarOpen ? "bg-amber-50 w-64" : "bg-transparent w-16"
    } fixed top-0 left-0 h-full text-white transition-all duration-300`}
    >
    {/* Toggle Icon */}
    <div
        className="flex items-center h-16 px-4 cursor-pointer"
        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
    >
        <FaBars size={24} className="text-gray-800" />
    </div>

    {/* Navigation Items */}
    {isSidebarOpen && (
        <div>
        <NavLink
            to="/"
            className="text-[40px] font-extrabold text-gray-800 mt-4 px-4"
            activeClassName="text-yellow-500" // You can customize the active link style
            onClick={() => setIsSidebarOpen(false)} // Close sidebar when clicked
        >
            손수잇다 🙌🏻
        </NavLink>

        <nav className="flex flex-col mt-6">
            <NavItem
            isOpen={isSidebarOpen}
            icon={<FaChalkboardTeacher color="#000000" />}
            label="강의실"
            to="/"
            closeSidebar={() => setIsSidebarOpen(false)} // Close sidebar when clicked
            />
            <NavItem
            isOpen={isSidebarOpen}
            icon={<FaClipboardList color="#000000" />}
            label="O/X 퀴즈"
            to="/QuizStart"
            closeSidebar={() => setIsSidebarOpen(false)} // Close sidebar when clicked
            />
            <NavItem
            isOpen={isSidebarOpen}
            icon={<PiTimerBold color="#000000" />}
            label="스피드퀴즈"
            to="/SpeedStart"
            closeSidebar={() => setIsSidebarOpen(false)} // Close sidebar when clicked
            />
            <NavItem
            isOpen={isSidebarOpen}
            icon={<BsFillPersonFill color="#000000" />}
            label="마이페이지"
            to="/MyPage"
            closeSidebar={() => setIsSidebarOpen(false)} // Close sidebar when clicked
            />
        </nav>
        </div>
    )}

    {/* Login Link at the bottom */}
    {isSidebarOpen && (
        <div className="absolute left-0 w-full px-4 bottom-4">
        <Link
            to="/Login" // Add the path to your login page
            className="flex items-center justify-center w-full py-2 text-white transition-colors bg-yellow-500 rounded-lg hover:bg-yellow-400"
            onClick={() => setIsSidebarOpen(false)} // Close sidebar when clicked
        >
            로그인
        </Link>
        </div>
    )}
    </div>
);
}

function NavItem({ isOpen, icon, label, to, closeSidebar }) {
return (
    <NavLink
    to={to} // Use NavLink to handle navigation
    className={`flex items-center px-4 py-3 hover:bg-yellow-500 cursor-pointer ${
        isOpen ? "justify-start" : "hidden"
    }`}
    activeClassName="bg-yellow-200" // Highlight active link with background color
    title={!isOpen ? label : ""}
    onClick={closeSidebar} // Close sidebar when this item is clicked
    >
    <div className="text-xl">{icon}</div>
    {isOpen && <span className="ml-4 text-sm text-gray-800">{label}</span>}
    </NavLink>
);
}
