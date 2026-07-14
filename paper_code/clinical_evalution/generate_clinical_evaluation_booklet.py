"""
generate_option2_docx.py

Run this script LOCALLY on your machine.
Reads all case images from option2_images/ and builds a Word document
with one case per section: header, three images, response table.

Requirements:
    pip install python-docx

Usage:
    python generate_option2_docx.py \\
        --images_dir /Users/ieb123/Documents/PhD/AKI_GOSH_prediction/new_approach/clinical_evaluation/option2_images \\
        --out        clinical_evaluation_option2.docx

        python generate_word_doc.py \
    --images_dir /Users/ieb123/Documents/PhD/AKI_GOSH_prediction/new_approach/clinical_evaluation/option2_images \
    --out        clinical_evaluation_option2.docx

    python generate_word_doc.py \
    --images_dir /Users/ieb123/Documents/PhD/AKI_GOSH_prediction/new_approach/clinical_evaluation/option2_images \
    --case_list  /Users/ieb123/Documents/PhD/AKI_GOSH_prediction/new_approach/clinical_evaluation/option2_images/case_list_CONFIDENTIAL.csv \
    --out        new_approach/clinical_evaluation/clinical_evaluation_option2.docx
"""

import argparse
import re
from pathlib import Path
import pandas as pd

from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


# ─────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color):
    """Set table cell background colour."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  hex_color)
    tcPr.append(shd)


def add_page_break(doc):
    p  = doc.add_paragraph()
    run = p.add_run()
    run.add_break(docx_enum_break())


def docx_enum_break():
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    br = OxmlElement('w:br')
    br.set(qn('w:type'), 'page')
    return br


def add_page_break_proper(doc):
    """Add a proper page break as a paragraph."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    p   = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    run = p.add_run()
    br  = OxmlElement('w:br')
    br.set(qn('w:type'), 'page')
    run._r.append(br)


def cover_page(doc):
    """Add a cover/instruction page."""
    doc.add_heading('CLINICAL EVALUATION STUDY', level=1)
    doc.add_heading('Option 2: Clinical Recognition Time Task', level=2)

    p = doc.add_paragraph()
    p.add_run(
        'This booklet presents 10 anonymised admissions from a paediatric '
        'intensive care unit. For each case, the EHR trajectory is shown '
        'across three charts:\n'
        '  • Laboratory values over time\n'
        '  • Blood gas and oxygen parameters over time\n'
        '  • Procedure and medication exposure timeline'
    )

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.add_run(
        'TASK: For each case, review the trajectory and indicate at what '
        'hour since admission you would first escalate care, request '
        'nephrology review, or suspect that the patient may be at risk '
        'of developing acute kidney injury.'
    ).bold = True

    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run(
        'IMPORTANT: The clinical outcome of each admission is withheld. '
        'Please respond based only on the trajectory shown. '
        'Complete this booklet before reviewing any model outputs.'
    )
    run.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)
    run.bold = True

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.add_run(
        'ANONYMISED — FOR RESEARCH USE ONLY. '
        'No patient-identifiable information is contained in this document.'
    ).italic = True

    add_page_break_proper(doc)


def add_response_table(doc):
    """Add the three-question response table."""
    # Header row + 3 question rows
    table = doc.add_table(rows=4, cols=2)
    table.style          = 'Table Grid'
    table.alignment      = WD_TABLE_ALIGNMENT.CENTER

    # Column widths: question 11cm, answer 5.5cm
    for row in table.rows:
        row.cells[0].width = Cm(11)
        row.cells[1].width = Cm(5.5)

    # Header row
    hdr = table.rows[0]
    set_cell_bg(hdr.cells[0], '1A3A5C')
    set_cell_bg(hdr.cells[1], '1A3A5C')
    for i, txt in enumerate(['Question', 'Your answer']):
        p   = hdr.cells[i].paragraphs[0]
        run = p.add_run(txt)
        run.bold             = True
        run.font.color.rgb   = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size        = Pt(9)

    # Question 1
    q1 = table.rows[1]
    set_cell_bg(q1.cells[0], 'F4F6F8')
    set_cell_bg(q1.cells[1], 'F4F6F8')
    q1.cells[0].paragraphs[0].add_run(
        'At what hour since admission would you first escalate care, '
        'request nephrology review, or suspect AKI?'
    ).font.size = Pt(9)
    q1.cells[1].paragraphs[0].add_run(
        'Hour since admission:\n\n__________'
    ).font.size = Pt(9)

    # Question 2
    q2 = table.rows[2]
    q2.cells[0].paragraphs[0].add_run(
        'I had no concern about AKI during this admission.'
    ).font.size = Pt(9)
    q2.cells[1].paragraphs[0].add_run(
        '☐  Yes        ☐  No'
    ).font.size = Pt(9)

    # Question 3
    q3 = table.rows[3]
    set_cell_bg(q3.cells[0], 'F4F6F8')
    set_cell_bg(q3.cells[1], 'F4F6F8')
    q3.cells[0].paragraphs[0].add_run(
        'What clinical feature(s) prompted your concern? (free text)'
    ).font.size = Pt(9)
    # Tall empty cell for writing space
    for _ in range(4):
        q3.cells[1].add_paragraph('')


def add_case(doc, case_num, img_labs, img_gases, img_procs,
             age='?', sex='?'):
    """Add one case: header + three images + response table."""

    # Case header
    p = doc.add_heading(f'Case {case_num:02d}', level=2)
    p.runs[0].font.color.rgb = RGBColor(0x1A, 0x3A, 0x5C)

    # Demographics line
    demo_p = doc.add_paragraph()
    demo_p.add_run(f'Age: {age}   |   Sex: {sex}').font.size = Pt(9)

    anon = doc.add_paragraph()
    run  = anon.add_run('ANONYMISED — FOR RESEARCH USE ONLY')

    run.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)
    run.font.size      = Pt(8)
    run.italic         = True

    # Three images — full width
    IMG_WIDTH = Inches(6.2)
    for img_path, subtitle in [
        (img_labs,  'Laboratory values over time'),
        (img_gases, 'Blood gas and oxygen parameters'),
        (img_procs, 'Procedure and medication exposure'),
    ]:
        if img_path and img_path.exists():
            cap = doc.add_paragraph()
            cap.add_run(subtitle).font.size = Pt(8)
            cap.runs[0].bold = True
            doc.add_picture(str(img_path), width=IMG_WIDTH)
        else:
            doc.add_paragraph(f'[Image not found: {img_path}]')

    doc.add_paragraph()  # spacer

    # Response table
    p = doc.add_paragraph()
    p.add_run('Your response:').bold = True

    add_response_table(doc)


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images_dir', required=True,
                   help='Path to option2_images/ folder')
    p.add_argument('--out', default='clinical_evaluation_option2.docx')
    p.add_argument('--case_list', default=None,
                   help='case_list_CONFIDENTIAL.csv — contains age and sex '
                        'per case number.')
    return p.parse_args()


def main():
    args      = parse_args()

    # Load demographics if provided
    demo = {}
    if args.case_list and Path(args.case_list).exists():
        conf = pd.read_csv(args.case_list)
        conf.columns = conf.columns.str.lower()
        print(conf.columns)
        for _, row in conf.iterrows():
            cn = int(row['case_number']) if 'case_number' in row.index else None
            print(cn)
            if cn:
                demo[cn] = {
                    'age': str(row['age']) if 'age' in row.index else '?',
                    'sex': str(row['sex']) if 'sex' in row.index else '?',
                }
        print(f"  Demographics loaded for {len(demo)} cases")

    img_dir   = Path(args.images_dir)
    out_path  = Path(args.out)

    # Find all case numbers from filenames
    labs_files = sorted(img_dir.glob('case_*_labs.png'))
    case_nums  = []
    for f in labs_files:
        m = re.search(r'case_(\d+)_labs', f.name)
        if m:
            case_nums.append(int(m.group(1)))
    case_nums = sorted(case_nums)

    if not case_nums:
        print(f"ERROR: no case_*_labs.png files found in {img_dir}")
        return

    print(f"Found {len(case_nums)} cases: {case_nums}")

    doc = Document()

    # Page margins — narrower to give images more space
    for section in doc.sections:
        section.page_width   = Cm(21)
        section.page_height  = Cm(29.7)
        section.left_margin  = Cm(2)
        section.right_margin = Cm(2)
        section.top_margin   = Cm(2)
        section.bottom_margin = Cm(2)

    # Cover page
    cover_page(doc)

    # One case per section
    for i, case_num in enumerate(case_nums):
        img_labs  = img_dir / f'case_{case_num:02d}_labs.png'
        img_gases = img_dir / f'case_{case_num:02d}_gases.png'
        img_procs = img_dir / f'case_{case_num:02d}_procedures.png'

        d = demo.get(case_num, {})
        add_case(doc, case_num, img_labs, img_gases, img_procs,
                 age=d.get('age', '?'), sex=d.get('sex', '?'))

        # Page break between cases (not after last)
        if i < len(case_nums) - 1:
            add_page_break_proper(doc)

    doc.save(str(out_path))
    print(f"\nSaved → {out_path}")
    print(f"  {len(case_nums)} cases, cover page + {len(case_nums)} case pages")


if __name__ == '__main__':
    main()