const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, LevelFormat, TabStopType, TabStopPosition
} = require("docx");

// ── Shared styles ──────────────────────────────────────────────────
const FONT = "Arial";
const FONT_SERIF = "Times New Roman";
const COLOR_ACCENT = "1B4F72";
const COLOR_HEADER_BG = "D6EAF8";
const COLOR_ROW_ALT = "F2F8FD";
const border = { style: BorderStyle.SINGLE, size: 1, color: "B0C4DE" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: COLOR_HEADER_BG, type: ShadingType.CLEAR },
    margins: cellMargins,
    verticalAlign: "center",
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, bold: true, font: FONT, size: 18 })] })]
  });
}

function dataCell(text, width, shaded = false, bold = false, align = AlignmentType.CENTER) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: shaded ? { fill: COLOR_ROW_ALT, type: ShadingType.CLEAR } : undefined,
    margins: cellMargins,
    children: [new Paragraph({ alignment: align, children: [new TextRun({ text, font: FONT, size: 18, bold })] })]
  });
}

function makeRow(cells, widths, shaded) {
  return new TableRow({
    children: cells.map((t, i) => dataCell(t, widths[i], shaded, i === 0, i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER))
  });
}

// ── Table 1: Overall Benchmark ──────────────────────────────────────
const t1Widths = [1500, 1120, 1120, 1120, 1120, 1120, 1120, 1240];
const t1Total = t1Widths.reduce((a, b) => a + b, 0);
const t1Data = [
  ["PrimeKG",    "0.580", "1.000", "0.250", "0.843", "0.977", "0.601", "0.831"],
  ["Hetionet",   "0.266", "1.000", "0.000", "0.737", "0.986", "0.401", "0.776"],
  ["DRKG",       "0.463", "0.999", "0.500", "0.776", "0.957", "0.507", "0.809"],
  ["OpenBioLink","0.421", "1.000", "0.500", "0.713", "0.894", "0.530", "0.731"],
];

const table1 = new Table({
  width: { size: t1Total, type: WidthType.DXA },
  columnWidths: t1Widths,
  rows: [
    new TableRow({ children: ["KG", "Cov.", "Ann.", "Trust.", "Topo.", "Stab.", "Task", "Gen."].map((h, i) => headerCell(h, t1Widths[i])) }),
    ...t1Data.map((r, idx) => makeRow(r, t1Widths, idx % 2 === 1))
  ]
});

// ── Table 2: Task Performance ──────────────────────────────────────
const t2Widths = [1800, 2100, 2100, 2100, 1260];
const t2Total = t2Widths.reduce((a, b) => a + b, 0);
const t2Headers = ["KG", "Link Pred. (AUROC)", "Nbr. Retrieval (R@100)", "Multi-hop (H@100)", "Score"];
const t2Data = [
  ["PrimeKG",    "0.947", "0.490", "0.366", "0.601"],
  ["Hetionet",   "0.884", "0.149", "0.170", "0.401"],
  ["DRKG",       "0.928", "0.270", "0.324", "0.507"],
  ["OpenBioLink","0.759", "0.523", "0.306", "0.530"],
];

const table2 = new Table({
  width: { size: t2Total, type: WidthType.DXA },
  columnWidths: t2Widths,
  rows: [
    new TableRow({ children: t2Headers.map((h, i) => headerCell(h, t2Widths[i])) }),
    ...t2Data.map((r, idx) => makeRow(r, t2Widths, idx % 2 === 1))
  ]
});

// ── Table 3: Generalization ──────────────────────────────────────
const t3Widths = [1800, 2300, 2300, 2960];
const t3Total = t3Widths.reduce((a, b) => a + b, 0);
const t3Headers = ["KG", "Sparse (Q1 AUROC)", "Cross-Domain (CV%)", "Prospective (AUROC)"];
const t3Data = [
  ["PrimeKG",    "0.691", "1.9%", "0.834"],
  ["Hetionet",   "0.784", "2.1%", "0.650"],
  ["DRKG",       "0.561", "1.2%", "0.905"],
  ["OpenBioLink","0.642", "9.6%", "0.716"],
];

const table3 = new Table({
  width: { size: t3Total, type: WidthType.DXA },
  columnWidths: t3Widths,
  rows: [
    new TableRow({ children: t3Headers.map((h, i) => headerCell(h, t3Widths[i])) }),
    ...t3Data.map((r, idx) => makeRow(r, t3Widths, idx % 2 === 1))
  ]
});

// ── Helper: paragraph shorthand ────────────────────────────────────
function p(runs, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: 276 },
    ...opts,
    children: runs.map(r =>
      typeof r === "string"
        ? new TextRun({ text: r, font: FONT_SERIF, size: 22 })
        : new TextRun({ font: FONT_SERIF, size: 22, ...r })
    )
  });
}

function heading(text, level) {
  return new Paragraph({
    heading: level,
    spacing: { before: level === HeadingLevel.HEADING_1 ? 320 : 220, after: 120 },
    children: [new TextRun({ text, font: FONT, size: level === HeadingLevel.HEADING_1 ? 28 : 24, bold: true, color: COLOR_ACCENT })]
  });
}

function caption(text) {
  return new Paragraph({
    spacing: { before: 80, after: 200 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, font: FONT, size: 18, italics: true, color: "555555" })]
  });
}

// ── Build document ─────────────────────────────────────────────────
const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT_SERIF, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: COLOR_ACCENT },
        paragraph: { spacing: { before: 320, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: COLOR_ACCENT },
        paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "BioKGSuite \u2014 Evaluation Report", font: FONT, size: 16, italics: true, color: "888888" })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Page ", font: FONT, size: 16, color: "888888" }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: "888888" })]
        })]
      })
    },
    children: [
      // ── Title ──
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 60 },
        children: [new TextRun({ text: "BioKGSuite: A Systematic Evaluation of Biomedical", font: FONT, size: 36, bold: true, color: COLOR_ACCENT })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 120 },
        children: [new TextRun({ text: "Knowledge Graphs for Drug Repurposing", font: FONT, size: 36, bold: true, color: COLOR_ACCENT })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 40 },
        children: [new TextRun({ text: "Emily Molins", font: FONT, size: 22 })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 320 },
        children: [new TextRun({ text: "March 2026", font: FONT, size: 20, color: "666666" })]
      }),

      // ── 1. Introduction ──
      heading("1  Introduction", HeadingLevel.HEADING_1),
      p([
        "Biomedical knowledge graphs (KGs) integrate heterogeneous biological data\u2014genes, diseases, drugs, pathways, and their relationships\u2014into unified graph structures that support computational drug repurposing. While several KGs are now widely used in the literature, choosing among them remains largely ad hoc, as no standardized evaluation framework exists. BioKGSuite addresses this gap by providing a reproducible, multi-dimensional benchmarking suite that evaluates KGs across seven complementary dimensions: ",
        { text: "coverage, annotation accuracy, trustworthiness, topology, stability, task performance, ", italics: true },
        "and ",
        { text: "generalisation", italics: true },
        ". The framework is demonstrated on four publicly available KGs\u2014PrimeKG, Hetionet, DRKG, and OpenBioLink\u2014against curated gold standards drawn from DrugBank, UniProt, Disease Ontology, Reactome, Open Targets, and CTD."
      ]),
      p([
        "The 19 constituent metrics are organized into three conceptual categories. ",
        { text: "Content", bold: true },
        " metrics (coverage, annotation accuracy, trustworthiness) assess the breadth, correctness, and provenance of the data. ",
        { text: "Structure", bold: true },
        " metrics (topology, stability) capture the graph\u2019s organizational properties and robustness to perturbation. ",
        { text: "Inference", bold: true },
        " metrics (task performance, generalisation) measure the degree to which the KG supports accurate biological predictions across diverse settings."
      ]),

      // ── 2. Key Findings ──
      heading("2  Key Findings", HeadingLevel.HEADING_1),
      p([
        "Table 1 presents normalized dimension scores (0\u20131 scale, higher is better) for each KG. PrimeKG achieves the highest aggregate score, driven by broad entity coverage and strong inference capacity. DRKG follows with balanced performance across content, structure, and generalisability. Hetionet, despite narrow coverage, scores highest on stability, reflecting the resilience of its tightly curated graph. OpenBioLink provides competitive task performance but shows weaker structural cohesion."
      ]),
      table1,
      caption("Table 1. Normalized dimension scores across all seven evaluation dimensions."),

      heading("2.1  Content Quality", HeadingLevel.HEADING_2),
      p([
        "Entity coverage relative to gold-standard references (DrugBank, UniProt, Disease Ontology, Reactome) ranges from 26.6% (Hetionet) to 58.0% (PrimeKG), highlighting substantive gaps even in the most comprehensive graph. All four KGs achieve near-perfect annotation accuracy (>0.999), confirming that entity identifiers resolve to valid ontology entries and that edges conform to declared schemas. Trustworthiness, however, reveals the starkest differences: DRKG and OpenBioLink provide per-edge provenance (enabling full audit trails), PrimeKG offers per-node attribution, while Hetionet supplies only type-level source metadata. Notably, no KG provides per-edge confidence scores or uncertainty quantification\u2014a finding that points to a major gap in current resources."
      ]),

      heading("2.2  Structural Properties", HeadingLevel.HEADING_2),
      p([
        "All four KGs exhibit small-world topology, with empirical clustering coefficients 250\u2013600 times greater than degree-matched Erd\u0151s\u2013R\u00e9nyi random graphs. Community detection (Louvain) reveals moderate alignment between detected communities and entity types, with PrimeKG showing the highest purity (NMI = 0.55). Stability testing via edge dropout indicates that all KGs maintain strong rank-order consistency (Spearman r > 0.93) under 10% random removal, though periphery-targeted dropout disproportionately degrades DRKG and OpenBioLink\u2014suggesting that hub structures play a critical role in their link-prediction signal."
      ]),

      heading("2.3  Inference Capacity", HeadingLevel.HEADING_2),
      p([
        "Task performance is evaluated across three biologically grounded inference tasks: link prediction of drug\u2013disease indications, neighbourhood retrieval of disease-associated genes, and multi-hop mechanistic reasoning using curated CTD triplets. Table 2 summarizes these results."
      ]),
      table2,
      caption("Table 2. Task performance across three inference benchmarks."),
      p([
        "PrimeKG leads on link prediction (AUROC 0.947) and multi-hop reasoning (Hits@100 = 0.366), while OpenBioLink achieves the highest neighbourhood retrieval (Recall@100 = 0.523). Hetionet\u2019s limited scope constrains all three task metrics, reinforcing the observation that coverage sets an effective ceiling on downstream utility."
      ]),

      heading("2.4  Generalisation", HeadingLevel.HEADING_2),
      p([
        "Generalisation testing assesses whether inference performance holds across data-sparse diseases, therapeutic domains, and prospective (post-training-cutoff) indications. Table 3 reports key results."
      ]),
      table3,
      caption("Table 3. Generalisation performance across three evaluation settings."),
      p([
        "DRKG shows the lowest cross-domain variance (CV = 1.2%) and the strongest prospective AUROC (0.905), suggesting its structure captures emerging therapeutic signals. PrimeKG balances sparse-entity robustness with strong cross-domain consistency. Hetionet\u2019s sparse-entity performance degrades only 8.2% from well-studied to rare diseases\u2014the most equitable profile\u2014but only two post-cutoff drug\u2013disease pairs fall within its scope, limiting prospective evaluation. DRKG exhibits the steepest sparse-entity penalty (40% degradation), indicating a prevalence bias toward well-annotated diseases."
      ]),

      // ── 3. Discussion ──
      heading("3  Discussion and Recommendations", HeadingLevel.HEADING_1),
      p([
        "Three cross-cutting insights emerge from this evaluation. First, no single KG dominates all dimensions; each exhibits a distinct profile of strengths and limitations. PrimeKG offers the broadest coverage and strongest aggregate inference, but its trustworthiness lags behind KGs with per-edge provenance. DRKG balances content quality and prospective generalisability but struggles with data-sparse entities. Hetionet\u2019s narrow, curated scope yields exceptional stability and equitable cross-disease performance at the cost of coverage-limited task performance."
      ]),
      p([
        "Second, structural quality is necessary but not sufficient for inference. All KGs share strong topological properties (small-world structure, high reachability, robust stability), yet their task performance varies by more than 50% across metrics. This confirms that graph structure alone does not determine predictive utility\u2014entity coverage and relational richness set the effective ceiling."
      ]),
      p([
        "Third, trustworthiness represents the most actionable dimension for improvement. Adding per-edge confidence scores and maintaining complete provenance trails would enhance all KGs without requiring changes to their underlying content. The universal absence of uncertainty quantification (0.0 across all KGs) is a notable gap, particularly for translational applications where evidence strength must be weighed alongside prediction scores."
      ]),
      p([
        "BioKGSuite is intended as a diagnostic tool rather than a ranking system. Researchers selecting a KG for drug repurposing should match the evaluation profile to their application requirements: broad discovery campaigns may favor PrimeKG\u2019s coverage, while projects requiring auditable evidence chains may prefer DRKG\u2019s per-edge provenance. The seven-dimensional framework provides a structured basis for these decisions and a reproducible baseline for evaluating future KG releases."
      ]),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/sessions/exciting-great-albattani/mnt/BioKGBench/BioKGSuite_Report.docx", buffer);
  console.log("Report saved successfully.");
});
