try {
  document.documentElement.dataset.theme =
    localStorage.getItem("vx-theme") ||
    (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
} catch (e) {
  document.documentElement.dataset.theme = "light";
}
