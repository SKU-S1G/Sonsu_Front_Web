import React from "react";
import { motion } from "framer-motion";

export default function Main6() {
  return (
    <div className="flex items-center justify-center px-32 h-fit">
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: false }} // false면 화면에 들어올 때마다 실행
        transition={{ duration: 1, ease: "easeOut" }}
        className="text-[35px] mb-44 fontSB tracking-widest"
      >
        지금, 손수잇다를 시작해보세요.
      </motion.div>
    </div>
  );
}
