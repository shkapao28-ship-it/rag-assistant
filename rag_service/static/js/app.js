// rag_service/static/js/app.js

document.addEventListener("DOMContentLoaded", () => {
    const toggle = document.querySelector("[data-toggle-chunks]");
    const list = document.querySelector("[data-chunks-list]");
    if (toggle && list) {
      toggle.addEventListener("click", () => {
        const isHidden = list.style.display === "none" || list.style.display === "";
        list.style.display = isHidden ? "block" : "none";
      });
    }
  });