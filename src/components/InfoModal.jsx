import React from "react";

const InfoModal = ({ visible, onClose, title, content }) => {
  if (!visible) return null;

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-[999]">
      <div className="w-4/5 max-w-md bg-white rounded-xl p-6 text-center shadow-lg">
        <h2 className="text-lg font-bold mb-3">{title}</h2>
        <p className="text-sm text-gray-700 whitespace-pre-line mb-5">
          {content}
        </p>
        <button
          onClick={onClose}
          className="bg-yellow-100 px-5 py-2 rounded shadow hover:bg-yellow-200 transition"
        >
          <span className="text-sm font-medium text-black">확인</span>
        </button>
      </div>
    </div>
  );
};

export default InfoModal;
