// Supplementary Table S4 — selection-bias audit for the evaluation set.
// Values from results/tables/bias_kept_vs_dropped.csv; group counts verified
// against reconstruct_116_audit.csv + dropped_253_to_116.csv.
const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, ShadingType, BorderStyle, HeadingLevel,
} = require('docx');

const DXA = 9360;                     // US Letter, 1" margins
const COLS = [2360, 1560, 1560, 1500, 1300, 1080];
const HDR = 'EDEADF';
const NONE = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
const RULE = { style: BorderStyle.SINGLE, size: 6, color: '333333' };
const HAIR = { style: BorderStyle.SINGLE, size: 2, color: 'BFBFBF' };

const txt = (t, o = {}) => new TextRun({ text: t, size: 17, font: 'Arial', ...o });

function cell(content, i, opts = {}) {
  const { bold = false, shade = null, top = NONE, bottom = HAIR, align } = opts;
  const lines = Array.isArray(content) ? content : [content];
  return new TableCell({
    width: { size: COLS[i], type: WidthType.DXA },
    margins: { top: 70, bottom: 70, left: 70, right: 70 },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade, color: 'auto' } : undefined,
    borders: { top, bottom, left: NONE, right: NONE },
    children: lines.map((l) => new Paragraph({
      alignment: align ?? (i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER),
      spacing: { after: 0, line: 250 },
      children: [txt(l, { bold })],
    })),
  });
}

const HEAD = ['Characteristic', 'Groups (n)', 'Retention rate',
  'Effect size', 'Test', 'P'];

const ROWS = [
  ['Oncology', '116 / 137', '41% vs 50%', 'OR = 0.72', "Fisher's exact", '0.21'],
  ['Biologic modality', '119 / 134', '50% vs 42%', 'OR = 1.42', "Fisher's exact", '0.21'],
  ['Multi-agency approval', '18 / 235', '39% vs 46%', 'OR = 0.74', "Fisher's exact", '0.63'],
  ['Strict repurposing', '212 / 41', '47% vs 39%', 'OR = 1.40', "Fisher's exact", '0.39'],
  ['Therapeutic area', '8 groups', '—', 'χ² = 10.8', 'Chi-square', '0.15'],
  ['Drug age (years since\noriginal approval)', '116 / 137',
    'median 6 vs 8', 'Δ = 2 years', 'Mann–Whitney', '0.013'],
];

const TA = [
  ['Oncology', '48', '68', '116'],
  ['Immunology', '28', '18', '46'],
  ['Cardiovascular', '8', '10', '18'],
  ['Infectious', '7', '9', '16'],
  ['Metabolic', '9', '7', '16'],
  ['CNS', '6', '4', '10'],
  ['Hematology', '5', '5', '10'],
  ['Ophthalmology', '4', '2', '6'],
  ['Other', '1', '14', '15'],
  ['Total', '116', '137', '253'],
];
const TACOLS = [3000, 2120, 2120, 2120];

function taCell(t, i, opts = {}) {
  const { bold = false, shade = null, top = NONE, bottom = HAIR } = opts;
  return new TableCell({
    width: { size: TACOLS[i], type: WidthType.DXA },
    margins: { top: 70, bottom: 70, left: 70, right: 70 },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade, color: 'auto' } : undefined,
    borders: { top, bottom, left: NONE, right: NONE },
    children: [new Paragraph({
      alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      spacing: { after: 0, line: 250 },
      children: [txt(t, { bold })],
    })],
  });
}

const mainTable = new Table({
  width: { size: DXA, type: WidthType.DXA },
  columnWidths: COLS,
  rows: [
    new TableRow({
      tableHeader: true,
      children: HEAD.map((h, i) =>
        cell(h, i, { bold: true, shade: HDR, top: RULE, bottom: RULE })),
    }),
    ...ROWS.map((r, ri) => new TableRow({
      children: r.map((c, i) => cell(c.split('\n'), i, {
        bottom: ri === ROWS.length - 1 ? RULE : HAIR,
      })),
    })),
  ],
});

const taTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: TACOLS,
  rows: [
    new TableRow({
      tableHeader: true,
      children: ['Therapeutic area', 'Retained', 'Discarded', 'Total']
        .map((h, i) => taCell(h, i, { bold: true, shade: HDR, top: RULE, bottom: RULE })),
    }),
    ...TA.map((r, ri) => new TableRow({
      children: r.map((c, i) => taCell(c, i, {
        bold: ri === TA.length - 1,
        top: ri === TA.length - 1 ? RULE : NONE,
        bottom: ri === TA.length - 1 ? RULE : HAIR,
      })),
    })),
  ],
});

const P = (runs, opts = {}) => new Paragraph({
  spacing: { before: opts.before ?? 0, after: opts.after ?? 160, line: 280 },
  children: runs,
});

const doc = new Document({
  styles: { default: { document: { run: { font: 'Arial', size: 20 } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    children: [
      P([txt('Supplementary Table S4 | Selection-bias audit of the prospective evaluation set.',
        { bold: true, size: 21 })]),
      P([txt('Comparison of the 116 retained pairs against the 137 discarded at any filter stage, '
        + 'across six characteristics recorded before filtering. Retention rate gives the '
        + 'percentage of each group retained. Only drug age differs at P < 0.05, retained drugs '
        + 'having received their original approval a median of two years more recently. '
        + 'Two-sided tests throughout; no correction for multiple comparisons, which would '
        + 'render all comparisons non-significant.')], { after: 220 }),
      mainTable,
      P([txt('Therapeutic-area composition', { bold: true })], { before: 380, after: 140 }),
      P([txt('Counts of retained and discarded pairs by therapeutic area. "Other" aggregates '
        + 'areas with fewer than six pairs (Respiratory, GI, GI/Metabolic, Nephrology, '
        + 'Endocrine, Other). Oncology is the largest area in both groups and is retained at a '
        + 'lower rate than the remainder (41% against 50%), though the difference is not '
        + 'significant.')], { after: 220 }),
      taTable,
    ],
  }],
});

Packer.toBuffer(doc).then((b) => {
  fs.writeFileSync(process.argv[2], b);
  console.log('written:', process.argv[2]);
});
