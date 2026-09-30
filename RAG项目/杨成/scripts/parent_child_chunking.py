import argparse
import json
import re
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
LIST_ITEM_RE = re.compile(r"^\s*(?:[（(]?\d+[）)]|\d+[.、](?!\d)|[-*+])\s*")
CAPTION_TITLE_RE = re.compile(r"^(?:图|表)[0-9一二三四五六七八九十]+")
CAPTION_KEYWORDS = ("流程图", "示意图", "表格")
SKIP_TITLE_KEYWORDS = (
    "National Clinical Practice Guidelines",
    "Abstract",
    "专家委员会",
    "参考文献",
    "利益冲突",
    "常务委员",
    "委员",
    "主任委员",
    "副主任委员",
    "秘书长",
    "副秘书长",
    "秘书",
    "名单",
)
TABLE_SPLIT_THRESHOLD = 2000
NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)(?=\D|$)")
BARE_NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+){1,3}\s*[^。！？；;\n]{0,80}$")
INLINE_NUMBERED_HEADING_RE = re.compile(r"(?<=[。！？；;])(?=\d+(?:\.\d+){1,3}\s*[^\d\s])")
SENTENCE_RE = re.compile(r"[^。！？；;\n]+[。！？；;]?")
LEADING_PUNCTUATION_RE = re.compile(r"^[。！？；;，、,.!?]")
CONTINUED_TABLE_RE = re.compile(r"^（?续表\d+）?$")


def should_skip_title(title):
    return any(keyword in title for keyword in SKIP_TITLE_KEYWORDS)


def is_caption_title(title):
    compact_title = title.strip()
    return CAPTION_TITLE_RE.match(compact_title) is not None or any(keyword in compact_title for keyword in CAPTION_KEYWORDS)


def is_numbered_section_heading(text):
    text = text.strip()
    return BARE_NUMBERED_HEADING_RE.match(text) is not None and re.search(r"[一-鿿]", text) is not None


def effective_heading_level(markdown_level, heading):
    numbered_match = NUMBERED_HEADING_RE.match(heading)
    if numbered_match:
        return min(numbered_match.group(1).count(".") + 1, 3)
    return markdown_level


def finalize_parent(parent, parents):
    if parent is None:
        return
    parent["content"] = "\n".join(parent.pop("lines")).strip()
    if not parent["content"]:
        parent["is_empty"] = True
    parents.append(parent)


def split_parents(markdown):
    parents = []
    stack = []
    current = None

    def start_parent(heading, markdown_level=None):
        nonlocal current, stack
        level = effective_heading_level(markdown_level or 1, heading)
        stack = [(item_level, item_heading) for item_level, item_heading in stack if item_level < level]
        candidate_stack = stack + [(level, heading)]
        title = " > ".join(item_heading for _, item_heading in candidate_stack)
        if should_skip_title(title):
            current = None
            return

        stack = candidate_stack
        current = {
            "parent_id": f"parent_{len(parents) + 1:04d}",
            "title": title,
            "lines": [],
        }

    def handle_heading(heading, markdown_level):
        nonlocal current
        if is_caption_title(heading):
            if current is not None:
                current["lines"].append(heading)
            return

        finalize_parent(current, parents)
        current = None
        start_parent(heading, markdown_level)

    def handle_bare_heading(heading):
        nonlocal current
        finalize_parent(current, parents)
        current = None
        start_parent(heading)

    for line in markdown.splitlines():
        match = HEADING_RE.match(line)
        if match:
            markdown_level = len(match.group(1))
            heading = match.group(2).strip()
            handle_heading(heading, markdown_level)
            continue

        split_line = INLINE_NUMBERED_HEADING_RE.split(line, maxsplit=1)
        if len(split_line) == 2 and is_numbered_section_heading(split_line[1]):
            if current is not None and split_line[0].strip():
                current["lines"].append(split_line[0])
            handle_bare_heading(split_line[1].strip())
            continue

        if is_numbered_section_heading(line):
            handle_bare_heading(line.strip())
            continue

        if current is not None:
            current["lines"].append(line)

    finalize_parent(current, parents)
    return parents


def split_table_segments(text):
    segments = []
    last = 0
    for match in TABLE_RE.finditer(text):
        if match.start() > last:
            segments.extend(split_list_segments(text[last : match.start()]))
        segments.append(("table", match.group(0).strip()))
        last = match.end()
    if last < len(text):
        segments.extend(split_list_segments(text[last:]))
    return [(kind, body) for kind, body in segments if body.strip()]


def split_list_segments(text):
    paragraphs = re.split(r"\n\s*\n", text.strip()) if text.strip() else []
    segments = []
    list_buffer = []
    text_buffer = []

    def flush_text():
        if text_buffer:
            segments.append(("text", "\n\n".join(text_buffer).strip()))
            text_buffer.clear()

    def flush_list():
        if list_buffer:
            segments.append(("list", "\n\n".join(list_buffer).strip()))
            list_buffer.clear()

    for paragraph in paragraphs:
        stripped = paragraph.strip()
        if LIST_ITEM_RE.match(stripped):
            flush_text()
            list_buffer.append(stripped)
        else:
            flush_list()
            text_buffer.append(stripped)

    flush_list()
    flush_text()
    return segments


def split_text(text, min_chars=500, max_chars=800, overlap=80):
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [cleanup_chunk(text)]

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current = []

    for sentence in sentences:
        candidate = join_sentences(current + [sentence])
        if current and len(candidate) > max_chars:
            chunks.append(join_sentences(current))
            current = sentence_overlap(current, overlap)
            if current and len(join_sentences(current + [sentence])) > max_chars and len(sentence) <= max_chars:
                current = []
        current.append(sentence)

    if current:
        chunks.append(join_sentences(current))

    cleaned_chunks = []
    for chunk in chunks:
        cleaned = cleanup_chunk(chunk)
        if cleaned:
            cleaned_chunks.append(cleaned)
    return cleaned_chunks


def split_sentences(text):
    sentences = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if is_numbered_section_heading(stripped):
            sentences.append(stripped)
            continue
        sentences.extend(match.group(0).strip() for match in SENTENCE_RE.finditer(stripped) if match.group(0).strip())
    return sentences


def join_sentences(sentences):
    return "".join(sentences).strip()


def sentence_overlap(sentences, overlap):
    if overlap <= 0 or not sentences:
        return []
    count = min(3, len(sentences))
    return sentences[-count:]


def cleanup_chunk(chunk):
    chunk = chunk.strip()
    while LEADING_PUNCTUATION_RE.match(chunk):
        chunk = chunk[1:].strip()
    return chunk


def build_parent_child_chunks(markdown, source_file, min_chars=500, max_chars=800, overlap=80):
    parents = split_parents(markdown)
    children = []

    for parent in parents:
        for segment_type, segment_text in split_table_segments(parent["content"]):
            if segment_type == "text":
                pieces = split_text(segment_text, min_chars=min_chars, max_chars=max_chars, overlap=overlap)
            else:
                pieces = [segment_text.strip()]

            for piece in pieces:
                piece = piece.strip()
                if not piece or CONTINUED_TABLE_RE.match(piece):
                    continue
                child = {
                    "child_id": f"child_{len(children) + 1:05d}",
                    "parent_id": parent["parent_id"],
                    "section_path": parent["title"],
                    "text": piece,
                    "chunk_type": segment_type,
                    "source_file": source_file,
                }
                if segment_type == "table" and len(piece) > TABLE_SPLIT_THRESHOLD:
                    child["needs_split"] = True
                children.append(child)

    return parents, children


def write_outputs(parents, children, parents_path, children_path):
    parents_path.write_text(json.dumps(parents, ensure_ascii=False, indent=2), encoding="utf-8")
    with children_path.open("w", encoding="utf-8") as output:
        for child in children:
            output.write(json.dumps(child, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--parents", required=True)
    parser.add_argument("--children", required=True)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=80)
    args = parser.parse_args()

    input_path = Path(args.input)
    markdown = input_path.read_text(encoding="utf-8")
    parents, children = build_parent_child_chunks(
        markdown,
        source_file=input_path.as_posix(),
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        overlap=args.overlap,
    )
    write_outputs(parents, children, Path(args.parents), Path(args.children))
    print(json.dumps({"parents": len(parents), "children": len(children)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
