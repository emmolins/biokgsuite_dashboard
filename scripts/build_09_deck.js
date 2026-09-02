// Build the 09a/09b/09c supervisor deck. Run from outputs dir (node build_09_deck.js)
const pptxgen = require("pptxgenjs");
const P = new pptxgen();
P.defineLayout({ name: "W", width: 13.333, height: 7.5 });
P.layout = "W";
P.author = "Emily Molins";
P.title = "BioKGSuite — robustness & diagnostic analyses";

const FIG = "/sessions/relaxed-festive-brown/mnt/outputs/deck_figs/";
const DARK = "1E2D2A", INK = "2C2C2A", MUTED = "6B6A64", LIGHT = "FFFFFF", TINT = "F2F4F3";
const TEAL = "2A9D8F", CORAL = "C2705A", GOLD = "B98A33", BLUE = "2E6B8A";
const HF = "Cambria", BF = "Arial";
const sh = () => ({ type: "outer", color: "000000", blur: 9, offset: 3, angle: 90, opacity: 0.12 });

const AR = { age_effect:1.60, bias_audit_composition:0.93, bias_audit_dist:2.64, bias_audit_forest:1.76,
  coverage_fair:1.60, difficulty_by_area:1.43, difficulty_drivers:1.59, hardest_pairs:1.02,
  intrinsic_difficulty:2.63, kg_by_model:1.98, lift_vs_size:1.60, mrr_vs_size:1.60 };

function fit(ar, bw, bh){ let w=bw, h=w/ar; if(h>bh){ h=bh; w=h*ar; } return {w,h}; }
// framed-figure motif: white rounded card + shadow, image centered inside
function figure(slide, key, bx, by, bw, bh){
  const {w,h} = fit(AR[key], bw, bh);
  const x = bx + (bw-w)/2, y = by + (bh-h)/2, pad = 0.1;
  slide.addShape(P.shapes.ROUNDED_RECTANGLE, { x:x-pad, y:y-pad, w:w+2*pad, h:h+2*pad,
    fill:{color:LIGHT}, line:{color:"E7E4DD", width:0.75}, rectRadius:0.06, shadow:sh() });
  slide.addImage({ path: FIG+key+".png", x, y, w, h });
}
function title(slide, t, tag, accent){
  slide.addText(t, { x:0.55, y:0.34, w:10.6, h:0.8, fontFace:HF, fontSize:27, bold:true, color:INK, margin:0, valign:"middle" });
  if(tag) slide.addText(tag.toUpperCase(), { x:10.9, y:0.46, w:1.9, h:0.45, fontFace:BF, fontSize:11, bold:true,
    color:accent||MUTED, align:"right", charSpacing:2, margin:0, valign:"middle" });
}
function takeaway(slide, x, y, w, lines, accent){
  slide.addText(lines.map((l,i)=>({ text:l, options:{ bullet:{indent:14}, color: i===0?INK:MUTED, bold:i===0,
    breakLine:true, paraSpaceAfter:8 } })), { x, y, w, h:3.4, fontFace:BF, fontSize:13.5, valign:"top" });
}

// ── S1 · title ───────────────────────────────────────────────────────────────
let s = P.addSlide(); s.background = { color: DARK };
s.addText("BIOKGSUITE", { x:0.9, y:1.5, w:8, h:0.4, fontFace:BF, fontSize:14, bold:true, color:TEAL, charSpacing:4, margin:0 });
s.addText("Robustness & diagnostic analyses", { x:0.9, y:2.0, w:11.5, h:1.5, fontFace:HF, fontSize:46, bold:true, color:LIGHT, margin:0 });
s.addText("Selection bias  ·  ranking difficulty  ·  within-family LLM scaling", { x:0.92, y:3.6, w:11, h:0.6, fontFace:BF, fontSize:19, color:"C9D6D2", margin:0 });
[["09a",CORAL],["09b",GOLD],["09c",TEAL]].forEach(([t,c],i)=>{
  s.addShape(P.shapes.OVAL,{ x:0.92+i*1.0, y:4.55, w:0.22, h:0.22, fill:{color:c} });
});
s.addText("Emily Molins  ·  DPhil Statistics, University of Oxford  ·  June 2026", { x:0.9, y:6.7, w:11, h:0.4, fontFace:BF, fontSize:12.5, color:"9DB0AB", margin:0 });

// ── S2 · roadmap ─────────────────────────────────────────────────────────────
s = P.addSlide(); s.background={color:LIGHT};
title(s, "Three questions, three checks");
const cards = [
  ["09a", CORAL, "Is the 116-pair benchmark a biased subset of what we dropped?",
   "Balanced on modality, therapeutic area and recency — only older (≤2015) drugs are under-represented, by design."],
  ["09b", GOLD, "Which drug–disease pairs are hard to rank, and why?",
   "Difficulty is intrinsic: it tracks the model's own confidence (r = 0.98), not KG coverage or drug age."],
  ["09c", TEAL, "Does KG grounding help more as the model scales?",
   "Lift over no-KG grows with size and saturates ~70B; the no-KG prior stays flat — not memorisation."],
];
cards.forEach(([n,c,q,a],i)=>{
  const x = 0.55 + i*4.15;
  s.addShape(P.shapes.ROUNDED_RECTANGLE,{ x, y:1.5, w:3.85, h:5.1, fill:{color:TINT}, line:{color:"E7E4DD",width:1}, rectRadius:0.08, shadow:sh() });
  s.addText(n.toUpperCase(),{ x:x+0.3, y:1.8, w:3.2, h:0.6, fontFace:HF, fontSize:30, bold:true, color:c, margin:0 });
  s.addText(q,{ x:x+0.3, y:2.6, w:3.25, h:1.7, fontFace:BF, fontSize:15.5, bold:true, color:INK, margin:0, valign:"top" });
  s.addText(a,{ x:x+0.3, y:4.35, w:3.25, h:2.0, fontFace:BF, fontSize:13, color:MUTED, margin:0, valign:"top" });
});

// ── divider helper ───────────────────────────────────────────────────────────
function divider(code, t, sub, accent){
  const d = P.addSlide(); d.background={color:DARK};
  d.addText(code.toUpperCase(),{ x:0.9, y:2.5, w:5, h:1.0, fontFace:HF, fontSize:54, bold:true, color:accent, margin:0 });
  d.addText(t,{ x:0.92, y:3.7, w:11.5, h:0.9, fontFace:HF, fontSize:30, bold:true, color:LIGHT, margin:0 });
  d.addText(sub,{ x:0.92, y:4.65, w:11, h:0.6, fontFace:BF, fontSize:16, color:"9DB0AB", margin:0 });
  return d;
}

// ── 09a ──────────────────────────────────────────────────────────────────────
divider("09a","Survey of dropped pairs","Why 137 of 253 candidate pairs didn't make the 116-pair benchmark", CORAL);

s = P.addSlide(); s.background={color:LIGHT};
title(s, "137 pairs dropped for integrity, not cherry-picking", "09a", CORAL);
const stats = [["81","Answer leakage","drug→disease edge already in a benchmarked KG"],
  ["37","No KG coverage","drug or disease absent from all three evaluation KGs"],
  ["19","Unresolved IDs","no DrugBank / disease ID, or placeholder indication"]];
stats.forEach(([n,l,d],i)=>{
  const x=0.55+i*4.15;
  s.addShape(P.shapes.ROUNDED_RECTANGLE,{ x, y:1.55, w:3.85, h:1.95, fill:{color:TINT}, line:{color:"E7E4DD",width:1}, rectRadius:0.08, shadow:sh() });
  s.addText(n,{ x:x+0.25, y:1.6, w:1.5, h:1.0, fontFace:HF, fontSize:44, bold:true, color:CORAL, margin:0, valign:"middle" });
  s.addText(l,{ x:x+1.7, y:1.75, w:2.0, h:0.8, fontFace:BF, fontSize:14, bold:true, color:INK, margin:0, valign:"middle" });
  s.addText(d,{ x:x+0.25, y:2.65, w:3.4, h:0.75, fontFace:BF, fontSize:11.5, color:MUTED, margin:0, valign:"top" });
});
s.addText("Both arms span 1983–2025 — neither set is concentrated in a single era (kept skews mildly newer).",
  { x:0.55, y:3.78, w:12.2, h:0.4, fontFace:BF, fontSize:13, italic:true, color:MUTED, margin:0 });
figure(s, "bias_audit_dist", 0.8, 4.2, 11.7, 3.0);

s = P.addSlide(); s.background={color:LIGHT};
title(s, "The kept set is unbiased — one by-design caveat", "09a", CORAL);
figure(s, "bias_audit_forest", 0.5, 1.7, 7.4, 4.2);
takeaway(s, 8.2, 2.1, 4.6, [
  "Fisher odds of being kept, per trait.",
  "Only “older drug (≤2015)” deviates from parity (OR ≈ 0.45, p < 0.01) — under-represented.",
  "Modality, therapeutic area, antibody/biologic and novel-modality are all n.s.",
  "The age skew is expected: the benchmark targets 2023–26 indications, bounding claims to recent drugs."], CORAL);

s = P.addSlide(); s.background={color:LIGHT};
title(s, "Kept vs dropped: matched on modality, area and year", "09a", CORAL);
figure(s, "bias_audit_composition", 0.5, 1.5, 6.6, 5.8);
takeaway(s, 7.5, 2.0, 5.3, [
  "Modality: small-molecule 48% kept vs 57% dropped; antibody/biologic 47% vs 38%.",
  "Therapeutic area: oncology 41% vs 50%; immunology 24% vs 13% — same rank order.",
  "Approval year: 2024–26 makes up 57% of kept vs 66% of dropped.",
  "No distribution diverges sharply — the selection rule didn't reshape the population."], CORAL);

// ── 09b ──────────────────────────────────────────────────────────────────────
divider("09b","Which pairs are hardest to rank","Per-pair mean reciprocal rank on the KG arm, pooled across GPT, Gemini and Llama", GOLD);

s = P.addSlide(); s.background={color:LIGHT};
title(s, "The 20 hardest-to-rank pairs", "09b", GOLD);
takeaway(s, 0.65, 1.7, 5.4, [
  "Hard pairs cluster in pediatric, rare and 2023–24 novel indications.",
  "Oncology label-expansions (dMMR endometrial, HR+/HER2 breast) recur — recent, narrow populations.",
  "All three models struggle on the same pairs — difficulty isn't a single model's blind spot."], GOLD);
figure(s, "hardest_pairs", 6.3, 1.5, 6.8, 5.8);

s = P.addSlide(); s.background={color:LIGHT};
title(s, "Difficulty tracks the model's own confidence", "09b", GOLD);
figure(s, "intrinsic_difficulty", 0.5, 1.55, 8.3, 5.5);
takeaway(s, 9.1, 1.9, 3.9, [
  "Per-pair MRR vs self-reported confidence: r = 0.98.",
  "Models “know” when they're guessing — low-confidence pairs are exactly the low-MRR ones.",
  "Colour = share of consistent KG evidence; it rises with both."], GOLD);

s = P.addSlide(); s.background={color:LIGHT};
title(s, "Real area gradient — not a coverage or recency artifact", "09b", GOLD);
const trio = [["difficulty_by_area","By therapeutic area","Immunology easiest (0.66); Infectious / Metabolic hardest (~0.40)"],
  ["coverage_fair","By # KGs covering","Covered-arm MRR is flat from 1→3 KGs — count doesn't help"],
  ["age_effect","By drug approval year","r = −0.11 (n.s.) — older drugs aren't easier"]];
trio.forEach(([k,l,d],i)=>{
  const x=0.45+i*4.25;
  figure(s, k, x, 1.6, 4.0, 3.4);
  s.addText(l,{ x, y:5.15, w:4.0, h:0.4, fontFace:BF, fontSize:14, bold:true, color:INK, align:"center", margin:0 });
  s.addText(d,{ x, y:5.55, w:4.0, h:1.2, fontFace:BF, fontSize:12, color:MUTED, align:"center", margin:0, valign:"top" });
});

// ── 09c ──────────────────────────────────────────────────────────────────────
divider("09c","Within-family LLM scaling","One architecture family, Llama 3.x · 1B → 405B · 3 seeds × 2 shuffles on the 116-set", TEAL);

s = P.addSlide(); s.background={color:LIGHT};
title(s, "KG lift grows with scale, then saturates ~70B", "09c", TEAL);
figure(s, "lift_vs_size", 0.5, 1.55, 6.1, 4.0);
figure(s, "mrr_vs_size", 6.8, 1.55, 6.1, 4.0);
takeaway(s, 0.65, 5.75, 12.2, [
  "Covered-arm lift over no-KG: −0.03 at 1B → +0.24 at 8B → +0.38 at 405B, flattening after 70B.",
  "The no-KG prior is flat across the whole ladder (~0.30) — the gain is grounding, not bigger-model memorisation."], TEAL);

s = P.addSlide(); s.background={color:LIGHT};
title(s, "Every KG beats no-KG from 8B upward", "09c", TEAL);
figure(s, "kg_by_model", 0.6, 1.55, 8.4, 5.6);
takeaway(s, 9.2, 1.95, 3.8, [
  "At 1B–3B, KG ≈ or < no-KG: too small to use the structure.",
  "Crossover at 8B; PrimeKG is strongest across the ladder, then BioKG, then DRKG.",
  "Grounding matters most at deployable model sizes."], TEAL);

// ── closing ──────────────────────────────────────────────────────────────────
s = P.addSlide(); s.background={color:DARK};
s.addText("What this de-risks", { x:0.9, y:1.0, w:11, h:0.9, fontFace:HF, fontSize:34, bold:true, color:LIGHT, margin:0 });
const out = [["09a",CORAL,"The benchmark is a defensible, unbiased subset — external validity holds (caveat: recent drugs)."],
  ["09b",GOLD,"Hard pairs are a real, model-invariant signal tied to confidence — a principled stress set, not noise."],
  ["09c",TEAL,"KG lift is a scaling phenomenon that saturates — grounding helps most at deployable sizes, and rules out memorisation."]];
out.forEach(([n,c,t],i)=>{
  const y=2.2+i*1.45;
  s.addText(n.toUpperCase(),{ x:0.95, y, w:1.4, h:0.8, fontFace:HF, fontSize:26, bold:true, color:c, margin:0, valign:"middle" });
  s.addText(t,{ x:2.5, y, w:10.0, h:1.2, fontFace:BF, fontSize:16.5, color:"E7ECEA", margin:0, valign:"middle" });
});

P.writeFile({ fileName: "/sessions/relaxed-festive-brown/mnt/outputs/BioKGSuite_09abc_analyses.pptx" })
  .then(f => console.log("wrote", f));
