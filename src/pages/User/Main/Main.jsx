import React from "react";
import Main1 from "./Main1";
import Main2 from "./Main2";
import Main3 from "./Main3";
import Main4 from "./Main4";
import Main5 from "./Main5";
import Main6 from "./Main6";
import Footer from "../../../components/Footer";

export default function Main() {
  return (
    <div
      className="w-screen"
      style={{
        background: "linear-gradient(180deg, #FFF 0%, #FFF7DF 100%)",
      }}
    >
      <Main1 />
      <Main2 />
      <Main3 />
      <Main4 />
      <Main5 />
      <Main6 />
      <Footer />
    </div>
  );
}
