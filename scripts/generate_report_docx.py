import os
import re
import zipfile
from html import escape


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_MD = os.path.join(BASE_DIR, "docs", "Project_Report.md")
REPORT_DOCX = os.path.join(BASE_DIR, "docs", "NLP_Project_Report.docx")


def inline_runs(text):
    text = text.replace("**", "")
    text = text.replace("`", "")
    return f"<w:r><w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r>"


def paragraph(text="", style=None):
    style_xml = ""
    if style:
        style_xml = f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>"
    return f"<w:p>{style_xml}{inline_runs(text)}</w:p>"


def bullet(text):
    return paragraph("- " + text)


def table(lines):
    rows = []
    for line in lines:
        if re.match(r"^\|\s*-", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        cell_xml = "".join(
            "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
            f"{paragraph(cell)}</w:tc>"
            for cell in cells
        )
        rows.append(f"<w:tr>{cell_xml}</w:tr>")
    borders = (
        "<w:tblPr><w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>"
        "</w:tblBorders></w:tblPr>"
    )
    return f"<w:tbl>{borders}{''.join(rows)}</w:tbl>"


def md_to_body(markdown):
    blocks = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            blocks.append(paragraph(""))
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            blocks.append(table(table_lines))
            continue
        if line.startswith("# "):
            blocks.append(paragraph(line[2:].strip(), "Title"))
        elif line.startswith("## "):
            text = line[3:].strip()
            style = "Heading1" if text.lower().startswith("chapter") else "Heading2"
            blocks.append(paragraph(text, style))
        elif line.startswith("### "):
            blocks.append(paragraph(line[4:].strip(), "Heading3"))
        elif line.startswith("- "):
            blocks.append(bullet(line[2:].strip()))
        elif re.match(r"^\d+\. ", line):
            blocks.append(paragraph(line.strip()))
        else:
            blocks.append(paragraph(line))
        i += 1
    return "\n".join(blocks)


def content_types():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""


def rels():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def styles():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:qFormat/><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:qFormat/><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:qFormat/><w:rPr><w:b/><w:sz w:val="22"/></w:rPr></w:style>
</w:styles>"""


def document_xml(body):
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>"""


def main():
    with open(REPORT_MD, "r", encoding="utf-8") as f:
        markdown = f.read()
    body = md_to_body(markdown)
    os.makedirs(os.path.dirname(REPORT_DOCX), exist_ok=True)
    with zipfile.ZipFile(REPORT_DOCX, "w", zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types())
        docx.writestr("_rels/.rels", rels())
        docx.writestr("word/document.xml", document_xml(body))
        docx.writestr("word/styles.xml", styles())
    print(REPORT_DOCX)


if __name__ == "__main__":
    main()
