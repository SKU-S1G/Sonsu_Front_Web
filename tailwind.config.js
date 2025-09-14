/** @type {import('tailwindcss').Config} */

module.exports = {
  content: ["./src/**/*.{html,js,jsx}"],
  theme: {
    extend: {
      keyframes: {
        scaleUp: {
          "0%": { transform: "translate(-50%, -50%) scale(1)", opacity: 0.5 },
          "100%": { transform: "translate(-50%, -50%) scale(5)", opacity: 1 },
        },
      },
      animation: {
        "scale-up": "scaleUp 2s ease-in-out forwards",
      },
    },
  },
  plugins: [],
};
