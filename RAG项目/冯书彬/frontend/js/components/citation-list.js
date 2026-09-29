function renderCitations(container, citations = []) {
  container.replaceChildren()
  container.dataset.testid = "citation-list"
  if (!citations.length) {
    container.hidden = true
    return
  }
  container.hidden = false
  const title = document.createElement("h3")
  title.textContent = "参考依据"
  container.append(title)
  citations.forEach((citation) => {
    const item = document.createElement("li")
    item.textContent = `${citation.title || "法律材料"} ${citation.location || ""}`
    container.append(item)
  })
}

export { renderCitations }
