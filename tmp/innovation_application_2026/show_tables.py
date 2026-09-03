from docx import Document
from pathlib import Path

source = Path(r"C:\\Users\\lxj\\Desktop") / (
    "\u674e\u5c0f\u56092026\u5e74\u5c71\u897f\u7701\u7814\u7a76\u751f\u5b9e\u8df5\u521b\u65b0\u9879\u76ee\u7533\u8bf7\u4e66.docx"
)
doc = Document(source)
for ti, table in enumerate(doc.tables[:3]):
    print(f"--- TABLE {ti} ---")
    for ri, row in enumerate(table.rows):
        cells = []
        seen = set()
        for ci, cell in enumerate(row.cells):
            if id(cell._tc) in seen:
                cells.append("<merged>")
            else:
                seen.add(id(cell._tc))
                cells.append(cell.text.replace("\n", " / "))
        print(ri, " || ".join(cells))
