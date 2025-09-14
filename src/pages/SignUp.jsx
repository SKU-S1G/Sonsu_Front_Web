import React, { useEffect, useState } from "react";

export default function SignUp() {
const [id, setId] = useState("");
const [pw, setPw] = useState("");
const [pwConfirm, setPwConfirm] = useState("");
const [email, setEmail] = useState("");
const [name, setName] = useState("");
const [birthYear, setBirthYear] = useState("");
const [birthMonth, setBirthMonth] = useState("");
const [birthDay, setBirthDay] = useState("");
const [gender, setGender] = useState("");
const [phone, setPhone] = useState("");
const [job, setJob] = useState("");

const [idMessage, setIdMessage] = useState("");
const [pwMessage, setPwMessage] = useState("");
const [pwConfirmMessage, setPwConfirmMessage] = useState("");
const [emailMessage, setEmailMessage] = useState("");
const [nameMessage, setNameMessage] = useState("");
const [phoneMessage, setPhoneMessage] = useState("");

const [isId, setIsId] = useState(false);
const [isPw, setIsPw] = useState(false);
const [isPwConfirm, setIsPwConfirm] = useState(false);
const [isEmail, setIsEmail] = useState(false);
const [isName, setIsName] = useState(false);
const [isPhone, setIsPhone] = useState(false);

const [notAllow, setNotAllow] = useState(true);

const handleId = (e) => {
    setId(e.target.value);
    const regex = /^(?=.*[A-Za-z])(?=.*?[0-9]).{6,}$/;

    if (regex.test(e.target.value)) {
    setIdMessage("사용 가능한 아이디 입니다.");
    setIsId(true);
    } else {
    setIdMessage("ID를 영문, 숫자 포함 6자 이상 입력해주세요");
    setIsId(false);
    }
};

const handlePw = (e) => {
    setPw(e.target.value);
    const regex =
    /^(?=.*[A-Za-z])(?=.*\d)(?=.*[$@$!%*#?&])[A-Za-z\d$@$!%*#?&]{8,}$/;

    if (regex.test(e.target.value)) {
    setPwMessage("사용 가능한 비밀번호 입니다.");
    setIsPw(true);
    } else {
    setPwMessage("PW를 영문, 숫자, 특수문자 포함 8자 이상 입력해주세요.");
    setIsPw(false);
    }
};

const handlePwConfirm = (e) => {
    setPwConfirm(e.target.value);
    if (pw !== e.target.value) {
    setPwConfirmMessage("비밀번호가 일치하지 않습니다.");
    setIsPwConfirm(false);
    } else {
    setPwConfirmMessage("비밀번호가 일치합니다.");
    setIsPwConfirm(true);
    }
};

const handleEmail = (e) => {
    setEmail(e.target.value);
    const emailReExp =
    /^[A-Za-z0-9_]+[A-Za-z0-9]*[@]{1}[A-Za-z0-9]+[A-Za-z0-9]*[.]{1}[A-Za-z]{1,3}$/;
    if (!emailReExp.test(e.target.value)) {
    setEmailMessage("이메일 형식이 올바르지 않습니다.");
    setIsEmail(false);
    } else {
    setEmailMessage("사용 가능한 이메일 입니다.");
    setIsEmail(true);
    }
};

const handleName = (e) => {
    setName(e.target.value);
    if (e.target.value.length < 2 || e.target.value.length > 5) {
    setNameMessage("이름은 2글자 이상, 5글자 이하로 입력해주세요");
    setIsName(false);
    } else {
    setNameMessage("사용 가능한 이름입니다");
    setIsName(true);
    }
};

const handlePhone = (e) => {
    setPhone(e.target.value);
    const phoneReExp = /^01([0|1|6|7|8|9])-?([0-9]{3,4})-?([0-9]{4})$/;
    if (!phoneReExp.test(e.target.value)) {
    setPhoneMessage("올바른 형식이 아닙니다");
    setIsPhone(false);
    } else {
    setPhoneMessage("사용 가능한 번호입니다");
    setIsPhone(true);
    }
};

const handleJob = (e) => {
    setJob(e.target.value);
};

useEffect(() => {
    if (isId && isPw && isPwConfirm && isEmail && isName && isPhone) {
    setNotAllow(false);
    return;
    }
    setNotAllow(true);
}, [isId, isPw, isPwConfirm, isEmail, isName, isPhone]);

//   const handleSubmit = async () => {
//     if (!notAllow) {
//       const birth_date = `${birthYear}-${birthMonth}-${birthDay}`;
//       const userData = {
//         user_name: name,
//         id: id,
//         email: email,
//         password: pw,
//         phone_number: phone,
//         birth_date: birth_date,
//         gender: gender,
//         job: job,
//       };

//       try {
//         const response = await axios.post(`${API_URL}/user/signup`, userData);
//         if (response.status === 201) {
//           // const { id } = response.data;
//           // signup({ username: id }); // 회원가입 후 username을 로컬 저장소에 저장하고 상태 업데이트
//           alert("회원가입에 성공했습니다.");
//           navigate("/signupcompleted");
//         } else if (response.status === 409) {
//           alert("중복된 아이디입니다.");
//         }
//       } catch (error) {
//         console.error("There was an error submitting the form!", error);
//         alert("회원가입에 실패했습니다. 다시 시도해주세요.");
//       }
//     }
//   };

return (
    <>
    <div className="flex justify-center">
        <div className="container mt-7 w-[40%] p-2 mb-8">
        <span className="text-[30px] font-semibold">회원가입</span>
        <span className="flex mt-2 ml-2 text-sm">
            * 은 필수로 입력해주세요
        </span>
        <div className="inputbox mt-7">
            <div className="flex flex-col mt-5 idbox">
            <span className="ml-1 text-base id">* ID</span>
            <input
                type="text"
                value={id}
                onChange={handleId}
                className="p-3 mt-1 border rounded-lg idinput"
                placeholder="test1234"
            />
            <p
                className={`message ${
                isId ? "text-green-600" : "text-red-600"
                }`}
            >
                {idMessage}
            </p>
            </div>
            <div className="flex flex-col mt-5 pwbox">
            <span className="ml-1 text-base pw">* PW</span>
            <input
                type="password"
                value={pw}
                onChange={handlePw}
                className="p-3 mt-1 border rounded-lg pwinput"
                placeholder="영문, 숫자, 특수문자 포함 8자 이상"
            />
            <p
                className={`message ${
                isPw ? "text-green-600" : "text-red-600"
                }`}
            >
                {pwMessage}
            </p>
            </div>
            <div className="flex flex-col mt-5 pwbox">
            <span className="ml-1 text-base pw">* confirm PW</span>
            <input
                type="password"
                value={pwConfirm}
                onChange={handlePwConfirm}
                className="p-3 mt-1 border rounded-lg pwinput"
                placeholder="패스워드를 한번 더 입력해주세요"
            />
            <p
                className={`message ${
                isPwConfirm ? "text-green-600" : "text-red-600"
                }`}
            >
                {pwConfirmMessage}
            </p>
            </div>
            <div className="flex flex-col mt-5 emailbox">
            <span className="ml-1 text-base id">* EMAIL</span>
            <input
                type="email"
                value={email}
                onChange={handleEmail}
                className="p-3 mt-1 border rounded-lg idinput"
                placeholder="test1234@naver.com"
            />
            <p
                className={`message ${
                isEmail ? "text-green-600" : "text-red-600"
                }`}
            >
                {emailMessage}
            </p>
            </div>
            <div className="flex flex-col mt-5">
            <span className="ml-1 text-base">* 이름</span>
            <input
                value={name}
                onChange={handleName}
                className="p-3 mt-1 border rounded-lg"
                placeholder="이호연"
            />
            <p
                className={`message ${
                isName ? "text-green-600" : "text-red-600"
                }`}
            >
                {nameMessage}
            </p>
            </div>
            <div className="flex flex-col mt-5 ">
            <span className="ml-1 text-base">생년월일</span>
            <div className="flex justify-between">
                <input
                className="border mt-1 p-3 rounded-lg w-[30%]"
                placeholder="년(4자)"
                value={birthYear}
                onChange={(e) => setBirthYear(e.target.value)}
                />
                <select
                className="w-[30%] border mt-1 p-3 rounded-lg"
                value={birthMonth}
                onChange={(e) => setBirthMonth(e.target.value)}
                >
                <option value="">월</option>
                <option value="01">1</option>
                <option value="02">2</option>
                <option value="03">3</option>
                <option value="04">4</option>
                <option value="05">5</option>
                <option value="06">6</option>
                <option value="07">7</option>
                <option value="08">8</option>
                <option value="09">9</option>
                <option value="10">10</option>
                <option value="11">11</option>
                <option value="12">12</option>
                </select>
                <select
                className="w-[30%] border mt-1 p-3 rounded-lg"
                value={birthDay}
                onChange={(e) => setBirthDay(e.target.value)}
                >
                <option value="">일</option>
                <option value="01">1</option>
                <option value="02">2</option>
                <option value="03">3</option>
                <option value="04">4</option>
                <option value="05">5</option>
                <option value="06">6</option>
                <option value="07">7</option>
                <option value="08">8</option>
                <option value="09">9</option>
                <option value="10">10</option>
                <option value="11">11</option>
                <option value="12">12</option>
                <option value="01">13</option>
                <option value="14">14</option>
                <option value="15">15</option>
                <option value="16">16</option>
                <option value="17">17</option>
                <option value="18">18</option>
                <option value="19">19</option>
                <option value="20">20</option>
                <option value="21">21</option>
                <option value="22">22</option>
                <option value="23">23</option>
                <option value="24">24</option>
                <option value="25">25</option>
                <option value="26">26</option>
                <option value="27">27</option>
                <option value="28">28</option>
                <option value="29">29</option>
                <option value="30">30</option>
                <option value="31">31</option>
                </select>
                {/* <input
                    className="border mt-1 p-3 rounded-lg w-[30%]"
                    placeholder="일"
                    value={birthDay}
                    onChange={(e) => setBirthDay(e.target.value)}
                /> */}
            </div>
            </div>
            <div className="flex flex-col mt-5">
            <span className="ml-1 text-base">성별</span>
            <select
                className="p-4 mt-1 border rounded-lg"
                value={gender}
                onChange={(e) => setGender(e.target.value)}
            >
                <option value="">성별</option>
                <option value="M">남</option>
                <option value="F">여</option>
            </select>
            </div>
            <div className="flex flex-col mt-5">
            <span className="ml-1 text-base">* 전화번호</span>
            <input
                value={phone}
                onChange={handlePhone}
                type="text"
                className="p-3 mt-1 border rounded-lg"
                placeholder="01012345678"
            />
            <p
                className={`message ${
                isPhone ? "text-green-600" : "text-red-600"
                }`}
            >
                {phoneMessage}
            </p>
            </div>
            <div className="flex flex-col mt-5 ">
            <span className="ml-1 text-base">직업</span>
            <select
                className="p-3 mt-1 border rounded-lg"
                value={job}
                onChange={handleJob}
            >
                <option value="">직업 선택</option>
                <option value="학생">학생</option>
                <option value="주부">주부</option>
                <option value="직장인">직장인</option>
                <option value="기타">기타</option>
            </select>
            </div>
        </div>
        <button
            disabled={notAllow}
            // onClick={handleSubmit}
            className="
                submit border mt-10 w-[100%] h-14 rounded-full bg-yellow-500 text-white 
                disabled:bg-[#dadada] disabled:text-white"
        >
            회원가입
        </button>
        </div>
    </div>
    </>
);
}
