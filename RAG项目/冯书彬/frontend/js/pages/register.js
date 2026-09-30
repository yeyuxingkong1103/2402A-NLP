const links = document.querySelectorAll("a")
links.forEach((link) => link.addEventListener("click", () => { localStorage.setItem("registration_notice_seen", "true") }))
