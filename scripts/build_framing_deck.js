// BioKGSuite — Nature Communications framing/strategy deck. node build_framing_deck.js
const pptxgen = require("pptxgenjs");
const P = new pptxgen();
P.defineLayout({ name: "W", width: 13.333, height: 7.5 });
P.layout = "W";
P.author = "Emily Molins";
P.title = "BioKGSuite — Nature Communications framing";

const DARK="1E2D2A", INK="2C2C2A", MUTED="6B6A64", LIGHT="FFFFFF", TINT="F2F4F3";
const TEAL="2A9D8F", CORAL="C2705A", GOLD="B98A33", BLUE="2E6B8A";
const HF="Cambria", BF="Arial";
const sh=()=>({type:"outer",color:"000000",blur:9,offset:3,angle:90,opacity:0.12});

function title(s,t,tag,accent){
  s.addText(t,{x:0.55,y:0.34,w:10.4,h:0.85,fontFace:HF,fontSize:26,bold:true,color:INK,margin:0,valign:"middle"});
  if(tag) s.addText(tag.toUpperCase(),{x:10.7,y:0.46,w:2.1,h:0.45,fontFace:BF,fontSize:11,bold:true,color:accent||MUTED,align:"right",charSpacing:2,margin:0,valign:"middle"});
}

// ── S1 title ──────────────────────────────────────────────────────────────────
let s=P.addSlide(); s.background={color:DARK};
s.addText("BIOKGSUITE  ·  MANUSCRIPT STRATEGY",{x:0.9,y:1.5,w:10,h:0.4,fontFace:BF,fontSize:14,bold:true,color:TEAL,charSpacing:3,margin:0});
s.addText("Framing for Nature Communications",{x:0.9,y:2.0,w:11.6,h:1.4,fontFace:HF,fontSize:44,bold:true,color:LIGHT,margin:0});
s.addText("What I'm balancing — and what submission actually requires",{x:0.92,y:3.55,w:11,h:0.6,fontFace:BF,fontSize:19,color:"C9D6D2",margin:0});
[["09a",CORAL],["09b",GOLD],["09c",TEAL]].forEach(([t,c],i)=> s.addShape(P.shapes.OVAL,{x:0.92+i*0.95,y:4.5,w:0.2,h:0.2,fill:{color:c}}));
s.addText("Emily Molins  ·  DPhil Statistics, University of Oxford  ·  June 2026",{x:0.9,y:6.7,w:11,h:0.4,fontFace:BF,fontSize:12.5,color:"9DB0AB",margin:0});

// ── S2 the fork: what is the paper about? ─────────────────────────────────────
s=P.addSlide(); s.background={color:LIGHT};
title(s,"The core decision: what is the paper about?");
const fork=[
 ["A · The resource",MUTED,"A 7-dimension quality benchmark over 6 KGs.",
  "Solid, reproducible — but “we built a benchmark” reads as Comms Bio / Sci Data, not Nature Comms.","weaker fit"],
 ["B · The finding",BLUE,"KG grounding lifts prospective drug-repurposing, and the lift scales with model size.",
  "A real finding — but “KG helps LLMs” alone is somewhat expected.","good"],
 ["C · The bridge",TEAL,"Lead with the finding; use the benchmark as the instrument; ask whether intrinsic KG quality predicts downstream value.",
  "The surprising, broad-interest claim. My recommended spine.","recommended"],
];
fork.forEach(([h,c,one,imp,tagv],i)=>{
  const x=0.55+i*4.15, rec=i===2;
  s.addShape(P.shapes.ROUNDED_RECTANGLE,{x,y:1.55,w:3.85,h:5.0,fill:{color:rec?"EAF3F1":TINT},line:{color:rec?TEAL:"E7E4DD",width:rec?1.5:1},rectRadius:0.08,shadow:sh()});
  s.addText(h,{x:x+0.28,y:1.78,w:3.3,h:0.5,fontFace:HF,fontSize:18,bold:true,color:c,margin:0});
  s.addText(one,{x:x+0.28,y:2.45,w:3.3,h:1.6,fontFace:BF,fontSize:13.5,bold:true,color:INK,margin:0,valign:"top"});
  s.addText(imp,{x:x+0.28,y:4.15,w:3.3,h:1.7,fontFace:BF,fontSize:12.5,color:MUTED,margin:0,valign:"top"});
  s.addText(tagv.toUpperCase(),{x:x+0.28,y:6.05,w:3.3,h:0.35,fontFace:BF,fontSize:10.5,bold:true,color:rec?TEAL:MUTED,charSpacing:1,margin:0});
});

// ── S3 tensions I'm balancing ─────────────────────────────────────────────────
s=P.addSlide(); s.background={color:LIGHT};
title(s,"What I'm balancing","navigating",CORAL);
const rows=[
 [{text:"Pulls one way",options:{bold:true,color:"FFFFFF",fill:{color:INK}}},
  {text:"Pulls the other",options:{bold:true,color:"FFFFFF",fill:{color:INK}}},
  {text:"My call",options:{bold:true,color:"FFFFFF",fill:{color:TEAL}}}],
 ["Comprehensive benchmark (7 dims, 6 KGs)","One clean narrative in 5,000 words","Benchmark = instrument, not the headline"],
 ["Safe claim: “KG grounding helps LLMs”","Surprising claim: scaling law + intrinsic ≠ extrinsic","Push to the surprising claim"],
 ["Breadth of claim","Current evidence (LLM task on 3 of 6 KGs)","Close the gap before claiming breadth"],
 ["Nature Comms ambition","Acceptance probability","Aim NC; prepare Comms Bio / Patterns fallback"],
 ["Prospective rigor (n = 116, leakage-controlled)","Statistical power / scope (one task family)","Pre-register an approval-year refresh"],
];
s.addTable(rows,{x:0.55,y:1.55,w:12.25,colW:[4.0,4.25,4.0],
  border:{type:"solid",pt:0.5,color:"DDD9D0"},fontFace:BF,fontSize:12.5,color:INK,valign:"middle",
  rowH:[0.45,0.92,0.92,0.92,0.92,0.92],align:"left",fill:{color:LIGHT},margin:5});

// ── S4 the gating gap ─────────────────────────────────────────────────────────
s=P.addSlide(); s.background={color:LIGHT};
title(s,"The one decision that gates the strongest claim","decision",CORAL);
s.addShape(P.shapes.ROUNDED_RECTANGLE,{x:0.7,y:1.7,w:11.9,h:2.0,fill:{color:"F7ECE8"},line:{color:CORAL,width:1.25},rectRadius:0.08,shadow:sh()});
s.addText([
  {text:"The intrinsic-quality winner is MATRIX (overall 0.85) — but the LLM task only ran PrimeKG, DRKG, BioKG.\n",options:{bold:true,color:INK,breakLine:true,fontSize:16}},
  {text:"So “intrinsic KG quality doesn't predict downstream LLM value” — the most novel claim — can't yet be tested, because the intrinsic leader was never in the downstream task.",options:{color:"6B4A42",fontSize:14}},
],{x:1.0,y:1.95,w:11.3,h:1.5,fontFace:BF,valign:"middle",margin:0});
s.addText("Two paths",{x:0.7,y:4.1,w:6,h:0.4,fontFace:HF,fontSize:16,bold:true,color:INK,margin:0});
[["Close it",TEAL,"Run the LLM task on all 6 KGs (esp. MATRIX). Unlocks the intrinsic-vs-extrinsic claim → a genuine Nature Comms angle. Same HPC scaffolding you already have."],
 ["Leave it",MUTED,"Ship “KG helps LLMs + a benchmark.” Expected, weaker — better suited to Comms Biology than Nature Comms."]].forEach(([h,c,t],i)=>{
  const x=0.7+i*6.05;
  s.addShape(P.shapes.ROUNDED_RECTANGLE,{x,y:4.6,w:5.85,h:2.2,fill:{color:TINT},line:{color:"E7E4DD",width:1},rectRadius:0.08,shadow:sh()});
  s.addText(h.toUpperCase(),{x:x+0.3,y:4.8,w:5.2,h:0.4,fontFace:BF,fontSize:13,bold:true,color:c,charSpacing:1,margin:0});
  s.addText(t,{x:x+0.3,y:5.25,w:5.25,h:1.45,fontFace:BF,fontSize:13,color:INK,margin:0,valign:"top"});
});

// ── S5 NC hard limits ─────────────────────────────────────────────────────────
s=P.addSlide(); s.background={color:LIGHT};
title(s,"What Nature Communications requires","format",BLUE);
const lim=[["5,000","words","main text (excl. abstract, Methods, refs, legends)"],
 ["200","words","abstract — no references"],
 ["≤10","display items","figures + tables (≤4 if under 2,000 words)"],
 ["≤70","references","footnotes not used"]];
lim.forEach(([n,u,d],i)=>{
  const x=0.55+i*3.13;
  s.addShape(P.shapes.ROUNDED_RECTANGLE,{x,y:1.7,w:2.9,h:2.5,fill:{color:TINT},line:{color:"E7E4DD",width:1},rectRadius:0.08,shadow:sh()});
  s.addText(n,{x:x+0.2,y:1.95,w:2.5,h:0.95,fontFace:HF,fontSize:38,bold:true,color:BLUE,margin:0,align:"center"});
  s.addText(u,{x:x+0.2,y:2.95,w:2.5,h:0.35,fontFace:BF,fontSize:13,bold:true,color:INK,margin:0,align:"center"});
  s.addText(d,{x:x+0.2,y:3.35,w:2.5,h:0.8,fontFace:BF,fontSize:11,color:MUTED,margin:0,align:"center",valign:"top"});
});
s.addText([
  {text:"Structure:  ",options:{bold:true,color:INK}},
  {text:"Introduction → Results (topical subheads) → Discussion (succinct, no subheads) → Methods (<3,000 words).   Title ≤15 words; figure legends ≤350 words each.",options:{color:MUTED}},
],{x:0.6,y:4.7,w:12.2,h:0.9,fontFace:BF,fontSize:14,valign:"top",margin:0});

// ── S6 structure mapped ───────────────────────────────────────────────────────
s=P.addSlide(); s.background={color:LIGHT};
title(s,"Structure, mapped to my content","format",BLUE);
const st=[
 [{text:"Section",options:{bold:true,color:"FFFFFF",fill:{color:INK}}},{text:"My content",options:{bold:true,color:"FFFFFF",fill:{color:INK}}}],
 ["Introduction","Two unsolved problems: contaminated/retrospective LLM-KG eval; unprincipled KG choice"],
 ["Results R1–R2","Quality benchmark (6 KGs); leakage-controlled prospective 116-pair task + bias audit"],
 ["Results R3–R4","KG-grounding lift across 3 model families; lift vs model scale (saturates ~70B)"],
 ["Results R5–R6","Difficulty diagnostic (model-invariant, KG-specific); intrinsic vs extrinsic [needs all-6-KG run]"],
 ["Discussion","Grounding + scale ≥8B is the lever, not KG choice; contamination inflates prior results"],
 ["Methods","KGs, 18 metrics, gold-standard build, ranking protocol, scaling ladder, statistics"],
];
s.addTable(st,{x:0.55,y:1.55,w:12.25,colW:[2.9,9.35],border:{type:"solid",pt:0.5,color:"DDD9D0"},
  fontFace:BF,fontSize:12.5,color:INK,valign:"middle",rowH:[0.45,0.8,0.8,0.8,0.8,0.8,0.8],align:"left",fill:{color:LIGHT},margin:5});

// ── S7 display-item budget ────────────────────────────────────────────────────
s=P.addSlide(); s.background={color:LIGHT};
title(s,"Display-item budget  (≤10; currently 5 + 1 to run)","format",BLUE);
const fb=[
 [{text:"#",options:{bold:true,color:"FFFFFF",fill:{color:INK}}},{text:"Figure",options:{bold:true,color:"FFFFFF",fill:{color:INK}}},{text:"Status",options:{bold:true,color:"FFFFFF",fill:{color:INK}}}],
 ["1","Quality benchmark × 6 KGs (7 dimensions)","ready"],
 ["2","Prospective benchmark build + selection-bias audit","ready"],
 ["3","KG-grounding lift across 3 model families","ready"],
 ["4","Lift vs model scale — saturation","ready"],
 ["5","Difficulty diagnostic (model-invariant, KG-specific)","ready"],
 ["6","Intrinsic quality vs downstream utility","needs all-6-KG run"],
 ["S1+","Embedding validation; per-dimension detail (supplement)","ready"],
];
s.addTable(fb,{x:0.55,y:1.55,w:12.25,colW:[0.7,8.95,2.6],border:{type:"solid",pt:0.5,color:"DDD9D0"},
  fontFace:BF,fontSize:12.5,color:INK,valign:"middle",rowH:[0.45,0.66,0.66,0.66,0.66,0.66,0.66,0.66],align:"left",fill:{color:LIGHT},margin:5});

// ── S8 closing / recommendation ───────────────────────────────────────────────
s=P.addSlide(); s.background={color:DARK};
s.addText("My recommendation",{x:0.9,y:0.95,w:11,h:0.9,fontFace:HF,fontSize:32,bold:true,color:LIGHT,margin:0});
const rec=[["1",TEAL,"Lead with the finding (the scaling law + leakage-controlled prospective design); the benchmark is the instrument."],
 ["2",GOLD,"Close the all-6-KG gap to unlock the intrinsic ≠ extrinsic claim — the part with broad significance."],
 ["3",CORAL,"Aim Nature Comms with the finding frame; prepare Communications Biology / Patterns as real fallbacks."]];
rec.forEach(([n,c,t],i)=>{
  const y=2.1+i*1.25;
  s.addText(n,{x:0.95,y,w:0.7,h:0.7,fontFace:HF,fontSize:24,bold:true,color:c,margin:0,valign:"middle"});
  s.addText(t,{x:1.8,y,w:10.7,h:1.1,fontFace:BF,fontSize:16,color:"E7ECEA",margin:0,valign:"middle"});
});
s.addText("Open decisions to align on:  (1) commit to a title/spine,  (2) green-light the all-6-KG run,  (3) add a random-KG control for the contamination referee.",
  {x:0.95,y:6.05,w:11.5,h:0.9,fontFace:BF,fontSize:13.5,italic:true,color:"9DB0AB",margin:0,valign:"top"});

P.writeFile({fileName:"/sessions/relaxed-festive-brown/mnt/outputs/BioKGSuite_NatComms_framing.pptx"}).then(f=>console.log("wrote",f));
