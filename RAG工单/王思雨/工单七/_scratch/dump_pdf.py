import sys, fitz as pymupdf
doc = pymupdf.open(sys.argv[1])
for i, p in enumerate(doc):
    t = p.get_text()
    print(f"\n===== PAGE {i+1}/{len(doc)} =====")
    print(t.strip())
