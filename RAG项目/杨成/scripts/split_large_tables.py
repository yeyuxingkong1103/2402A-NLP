import argparse
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path


DEFAULT_INPUT = Path("data/processed/chunks/parent_child_chunks.jsonl")
DEFAULT_OUTPUT = Path("data/processed/tables/table_rows.jsonl")
TABLE_TITLE = "表4基层常用降压药物"
BROKEN_WORD_REPLACEMENTS = {
    "高；血管": "高钾血症；血管",
    "两；种": "两种",
    "心；肌": "心肌",
    "梗；死": "梗死",
}
BROKEN_CONTINUATION_PREFIXES = (
    "梗死后",
    "化",
    "病",
    "侧",
    "mg",
    "mg/dI",
    "mg/dL",
    "mmHg",
    "mmol",
    "umol",
    "μmol",
)


class TableHtmlParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.current_row = None
        self.current_cell = None
        self.in_cell = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"} and self.current_row is not None:
            self.in_cell = True
            self.current_cell = {
                "text_parts": [],
                "rowspan": parse_span(attrs.get("rowspan")),
                "colspan": parse_span(attrs.get("colspan")),
            }

    def handle_data(self, data):
        if self.in_cell and self.current_cell is not None:
            self.current_cell["text_parts"].append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.current_cell is not None and self.current_row is not None:
            self.current_row.append(
                {
                    "text": normalize_text("".join(self.current_cell["text_parts"])),
                    "rowspan": self.current_cell["rowspan"],
                    "colspan": self.current_cell["colspan"],
                }
            )
            self.current_cell = None
            self.in_cell = False
        elif tag == "tr" and self.current_row is not None:
            self.rows.append(self.current_row)
            self.current_row = None


def parse_span(value):
    try:
        span = int(value or 1)
    except ValueError:
        span = 1
    return max(span, 1)


def normalize_text(text):
    text = html.unescape(text or "")
    return re.sub(r"\s+", " ", text).strip()


def parse_html_table_rows(table_html):
    parser = TableHtmlParser()
    parser.feed(table_html)
    return expand_spans(parser.rows)


def expand_spans(raw_rows):
    expanded_rows = []
    active_rowspans = {}

    for raw_row in raw_rows:
        expanded_row = []
        column_index = 0

        def fill_active_cells():
            nonlocal column_index
            while column_index in active_rowspans:
                active = active_rowspans[column_index]
                expanded_row.append(active["text"])
                active["remaining"] -= 1
                if active["remaining"] <= 0:
                    del active_rowspans[column_index]
                column_index += 1

        fill_active_cells()
        for cell in raw_row:
            fill_active_cells()
            for colspan_offset in range(cell["colspan"]):
                expanded_row.append(cell["text"])
                if cell["rowspan"] > 1:
                    active_rowspans[column_index] = {
                        "text": cell["text"],
                        "remaining": cell["rowspan"] - 1,
                    }
                column_index += 1
        fill_active_cells()
        expanded_rows.append(expanded_row)

    return expanded_rows


def clean_field_value(value):
    value = normalize_text(value)
    if not value:
        return value
    value = re.sub(r"；{2,}", "；", value)
    for broken, fixed in BROKEN_WORD_REPLACEMENTS.items():
        value = value.replace(broken, fixed)
    for prefix in BROKEN_CONTINUATION_PREFIXES:
        value = value.replace(f"；{prefix}", prefix)
    value = value.replace("外周；动脉", "外周动脉")
    value = re.sub(r"(?<=[A-Za-z0-9>）)])；(?=[A-Za-z0-9μ])", "", value)
    value = re.sub(r"(?<=[一-鿿])；(?=[一-鿿](?:；|$))", "", value)
    value = re.sub(r"；{2,}", "；", value)
    return value.strip("；")


def row_to_record(source_record, row, row_index):
    values = [clean_field_value(value) for value in normalize_row_values(row)]
    text = (
        f"{TABLE_TITLE} | 分类: {values[0]} | 名称: {values[1]} | 每次剂量: {values[2]} | "
        f"服药频率: {values[3]} | 推荐常用起始用法: {values[4]} | 适应证: {values[5]} | "
        f"禁忌证: {values[6]} | 主要不良反应: {values[7]}"
    )
    return {
        "child_id": f"{source_record['child_id']}_row_{row_index:03d}",
        "parent_id": source_record["parent_id"],
        "section_path": source_record["section_path"],
        "text": text,
        "chunk_type": "table_row",
        "source_file": source_record["source_file"],
        "row_index": row_index,
    }


def generic_row_to_record(source_record, headers, row, row_index):
    values = normalize_row_values(row)[: len(headers)]
    fields = []
    for index, value in enumerate(values):
        value = clean_field_value(value)
        if not value:
            continue
        header = clean_field_value(headers[index]) or f"第{index + 1}列"
        fields.append(f"{header}: {value}")
    return {
        "child_id": f"{source_record['child_id']}_row_{row_index:03d}",
        "parent_id": source_record["parent_id"],
        "section_path": source_record["section_path"],
        "text": "表格行 | " + " | ".join(fields),
        "chunk_type": "table_row",
        "source_file": source_record["source_file"],
        "row_index": row_index,
    }


def normalize_row_values(row):
    values = [normalize_text(value) for value in row[:8]]
    if len(values) < 8:
        values.extend([""] * (8 - len(values)))
    return values


def is_header_row(row):
    joined = "|".join(row)
    return "分类" in joined and "名称" in joined and "剂量" in joined


def has_drug_name(row):
    values = normalize_row_values(row)
    return bool(values[1]) and not is_header_row(values)


def merge_non_empty(values):
    merged = []
    seen = set()
    for value in values:
        value = normalize_text(value)
        if not value or value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return "；".join(merged)


def complete_group_fields(rows):
    normalized_rows = [normalize_row_values(row) for row in rows]
    groups = {}
    for row in normalized_rows:
        groups.setdefault(row[0], []).append(row)

    group_defaults = {}
    for category, group_rows in groups.items():
        group_defaults[category] = {
            column_index: merge_non_empty(row[column_index] for row in group_rows)
            for column_index in (5, 6, 7)
        }

    completed_rows = []
    for row in normalized_rows:
        completed = list(row)
        defaults = group_defaults[row[0]]
        for column_index in (5, 6, 7):
            if defaults[column_index]:
                completed[column_index] = defaults[column_index]
        completed_rows.append(completed)
    return completed_rows


def table_record_to_rows(record):
    rows = parse_html_table_rows(record["text"])
    if not rows:
        return []
    if len(rows[0]) == 8:
        data_rows = [row for row in rows if has_drug_name(row)]
        data_rows = complete_group_fields(data_rows)
        return [row_to_record(record, row, row_index) for row_index, row in enumerate(data_rows, start=1)]

    headers = [clean_field_value(value) for value in rows[0]]
    data_rows = [row for row in rows[1:] if any(normalize_text(value) for value in row)]
    return [generic_row_to_record(record, headers, row, row_index) for row_index, row in enumerate(data_rows, start=1)]


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 第 {line_number} 行解析失败：{exc}") from exc


def split_large_tables(input_path, output_path):
    source_tables = [record for record in iter_jsonl(input_path) if record.get("needs_split") is True]
    table_rows = []
    for source_table in source_tables:
        table_rows.extend(table_record_to_rows(source_table))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in table_rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return source_tables, table_rows


def find_example(table_rows, keywords):
    for row in table_rows:
        if any(keyword in row["text"] for keyword in keywords):
            return row
    return None


def main():
    parser = argparse.ArgumentParser(description="Split oversized HTML tables into row-level chunks.")
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    args = parser.parse_args()

    source_tables, table_rows = split_large_tables(args.input, args.output)
    examples = {
        "ACEI": find_example(table_rows, ["ACEI"]),
        "CCB": find_example(table_rows, ["CCB", "钙拮抗剂", "氨氯地平"]),
    }

    print(
        json.dumps(
            {
                "source_table_count": len(source_tables),
                "source_table_ids": [record["child_id"] for record in source_tables],
                "table_row_count": len(table_rows),
                "output": args.output.as_posix(),
                "examples": examples,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
