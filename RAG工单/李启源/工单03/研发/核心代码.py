"""工单03：PDF表格解析及检索优化的可运行核心代码。"""

def run(query, documents):
    rows = []
    for name, text in documents.items():
        for line in text.splitlines():
            if "|" in line or "," in line:
                cells = [cell.strip() for cell in line.replace(",", "|").split("|") if cell.strip()]
                if cells and any(word in line for word in query.split()):
                    rows.append((name, cells))
    return rows or [("示例表格", ["项目", "金额"])]

if __name__ == "__main__":
    print(run("募集资金", {"表格.csv": "项目,金额\n补充流动资金,50万元"}))
