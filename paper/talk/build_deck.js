// Goldfish Models talk deck.
// Theme: Ocean Gradient — 065A82 (deep blue) / 1C7293 (teal) / 21295C (midnight) /
//                          F5F7FA (paper) / FFFFFF (white) / FF7F50 (coral accent)
// 16:9, ~9 slides for ~9-minute talk.

const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";  // 13.3" × 7.5" — all positions assume this canvas
pres.title = "Goldfish Models";
pres.author = "Raghav Samavedam";

// Palette
const C = {
  bgDark: "0A1929",          // very dark navy for title slide
  bgPaper: "F5F7FA",         // light paper
  bgWhite: "FFFFFF",
  deep: "065A82",            // dominant primary
  teal: "1C7293",            // secondary
  midnight: "21295C",        // dark accent
  coral: "FF7F50",           // accent for stats
  ink: "1E2761",             // body text
  muted: "5A6A7F",           // caption text
  rule: "D8DEE6",            // hairlines
};

const F = {
  head: "Georgia",
  body: "Calibri",
  mono: "Consolas",
};

function pageTitle(slide, text) {
  slide.addText(text, {
    x: 0.5, y: 0.35, w: 12.3, h: 0.7,
    fontSize: 30, fontFace: F.head, color: C.midnight, bold: true,
    margin: 0,
  });
}

function pageNumberFoot(slide, n, total) {
  slide.addText(`${n}/${total}`, {
    x: 12.4, y: 7.0, w: 0.5, h: 0.3,
    fontSize: 9, fontFace: F.body, color: C.muted, align: "right", margin: 0,
  });
  slide.addText("Goldfish Models", {
    x: 0.5, y: 7.0, w: 4, h: 0.3,
    fontSize: 9, fontFace: F.body, color: C.muted, italic: true, margin: 0,
  });
}

const TOTAL = 10;

// =========================================================================
// Slide 1 — Title
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgDark };
  // Big stripe of teal on the right side
  s.addShape(pres.shapes.RECTANGLE, {
    x: 9.3, y: 0, w: 4.0, h: 7.5,
    fill: { color: C.deep }, line: { type: "none" },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 9.3, y: 4.0, w: 4.0, h: 3.5,
    fill: { color: C.teal }, line: { type: "none" },
  });

  s.addText("GOLDFISH", {
    x: 0.7, y: 1.5, w: 8.5, h: 1.2,
    fontSize: 80, fontFace: F.head, color: C.bgPaper, bold: true, charSpacing: 4, margin: 0,
  });
  s.addText("MODELS", {
    x: 0.7, y: 2.6, w: 8.5, h: 1.2,
    fontSize: 80, fontFace: F.head, color: C.coral, bold: true, charSpacing: 4, margin: 0,
  });
  s.addText("Trading per-turn context for memory and throughput", {
    x: 0.7, y: 4.0, w: 8.5, h: 0.6,
    fontSize: 22, fontFace: F.body, color: C.bgPaper, italic: true, margin: 0,
  });
  s.addText([
    { text: "Raghav Samavedam", options: { bold: true, breakLine: true } },
    { text: "CS 153 · Spring 2026", options: { color: "9FB6CC" } },
  ], {
    x: 0.7, y: 5.6, w: 8.5, h: 1.2,
    fontSize: 16, fontFace: F.body, color: C.bgPaper, margin: 0,
  });

  // Decorative tagline on the right stripe
  s.addText("a goldfish with a notebook", {
    x: 9.45, y: 4.4, w: 3.8, h: 0.5,
    fontSize: 14, fontFace: F.head, color: C.bgPaper, italic: true, margin: 0,
  });
  s.addText("(the regime)", {
    x: 9.45, y: 4.85, w: 3.8, h: 0.4,
    fontSize: 11, fontFace: F.body, color: "C9D6E2", margin: 0,
  });
}

// =========================================================================
// Slide 2 — Biological motivation
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgPaper };
  pageTitle(s, "Humans solve hard problems with ~4 items in working memory");
  pageNumberFoot(s, 2, TOTAL);

  // Left: the big stat
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.3, w: 5.8, h: 5.3,
    fill: { color: C.midnight }, line: { type: "none" },
  });
  s.addText("Cowan (2001):", {
    x: 0.75, y: 1.55, w: 5.3, h: 0.4,
    fontSize: 14, fontFace: F.body, color: "C9D6E2", italic: true, margin: 0,
  });
  s.addText("4 ± 1", {
    x: 0.75, y: 2.0, w: 5.3, h: 2.3,
    fontSize: 150, fontFace: F.head, color: C.coral, bold: true, align: "center", valign: "middle", margin: 0,
  });
  s.addText("items in active working memory at once.", {
    x: 0.75, y: 4.4, w: 5.3, h: 0.5,
    fontSize: 16, fontFace: F.body, color: C.bgPaper, align: "center", margin: 0,
  });
  s.addText("Miller's earlier 7±2 figure was for short-term capacity with chunking and rehearsal.\nFor immediate, untrained recall the number is closer to 4.", {
    x: 0.75, y: 5.0, w: 5.3, h: 1.3,
    fontSize: 12, fontFace: F.body, color: "C9D6E2", italic: true, align: "center", margin: 0,
  });

  // Right: the analogy
  s.addText("Where the actual work lives: long-term memory", {
    x: 6.7, y: 1.3, w: 6.3, h: 0.5,
    fontSize: 17, fontFace: F.head, color: C.midnight, bold: true, margin: 0,
  });

  const items = [
    {
      h: "Working memory is the bottleneck",
      b: "Only ~4 items active at once. But humans solve enormous problems by retrieving from a vastly larger long-term store — the bulk of cognition is in retrieval, not in holding state.",
    },
    {
      h: "Retrieval is precise, not fuzzy",
      b: "A surgeon recalls a specific protocol step. A programmer recalls a specific function signature. We retrieve named, structured items — not vibes-similar associations.",
    },
    {
      h: "And we structure memories as we learn",
      b: "Studying a new domain, we deliberately organize: outlines, indexes, mnemonics, hierarchies. We don't just dump — we curate the store so future retrieval is cheap.",
    },
  ];
  let py = 1.95;
  for (const it of items) {
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.7, y: py, w: 6.3, h: 1.35,
      fill: { color: C.bgWhite }, line: { color: C.rule, width: 0.5 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.7, y: py, w: 0.12, h: 1.35,
      fill: { color: C.coral }, line: { type: "none" },
    });
    s.addText(it.h, {
      x: 6.95, y: py + 0.1, w: 6.0, h: 0.4,
      fontSize: 14, fontFace: F.head, color: C.midnight, bold: true, margin: 0,
    });
    s.addText(it.b, {
      x: 6.95, y: py + 0.5, w: 6.0, h: 0.8,
      fontSize: 12, fontFace: F.body, color: C.ink, margin: 0,
    });
    py += 1.45;
  }

  // Bottom line — connects bio motivation to LLM proposal
  s.addText("Goldfish = small working set (context window) + precise, structured long-term memory (the filesystem). Not fuzzy embedding search.", {
    x: 0.5, y: 6.65, w: 12.5, h: 0.4,
    fontSize: 14, fontFace: F.body, color: C.deep, italic: true, bold: true, align: "center", margin: 0,
  });
}

// =========================================================================
// Slide 3 — The KV cache problem
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgPaper };
  pageTitle(s, "Long-context inference is bottlenecked by the KV cache");
  pageNumberFoot(s, 3, TOTAL);

  // Left column: the formula
  s.addText("Per-token KV memory:", {
    x: 0.5, y: 1.25, w: 6.0, h: 0.4,
    fontSize: 15, fontFace: F.body, color: C.muted, margin: 0,
  });

  // Big formula card
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.75, w: 6.0, h: 1.5,
    fill: { color: C.bgWhite }, line: { color: C.rule, width: 0.5 },
  });
  s.addText("M_kv  =  2 · L · n_layer · n_head · d_head · b", {
    x: 0.5, y: 1.95, w: 6.0, h: 0.7,
    fontSize: 22, fontFace: F.mono, color: C.deep, bold: true, align: "center", margin: 0,
  });
  s.addText("(K + V) · sequence × per-token weight × bytes", {
    x: 0.5, y: 2.75, w: 6.0, h: 0.4,
    fontSize: 11, fontFace: F.body, color: C.muted, italic: true, align: "center", margin: 0,
  });

  // Llama-3 70B example numbers
  s.addText("Concrete example — Llama-3 70B, FP16:", {
    x: 0.5, y: 3.5, w: 6.0, h: 0.4,
    fontSize: 14, fontFace: F.body, color: C.midnight, bold: true, margin: 0,
  });
  s.addText([
    { text: "n_layer = 80   n_head_kv = 8   d_head = 128", options: { breakLine: true } },
    { text: "→  ~320 KB per token", options: { bold: true, color: C.coral } },
  ], {
    x: 0.5, y: 3.95, w: 6.0, h: 0.9,
    fontSize: 14, fontFace: F.mono, color: C.ink, margin: 0,
  });

  s.addText("At L = 32k tokens, batch = 16:", {
    x: 0.5, y: 4.95, w: 6.0, h: 0.4,
    fontSize: 14, fontFace: F.body, color: C.midnight, bold: true, margin: 0,
  });
  s.addText([
    { text: "→ ", options: {} },
    { text: "164 GB", options: { bold: true, color: C.coral } },
    { text: " just for KV cache", options: {} },
  ], {
    x: 0.5, y: 5.4, w: 6.0, h: 0.5,
    fontSize: 18, fontFace: F.body, color: C.ink, margin: 0,
  });

  // Right column: the implication
  s.addText("The implication", {
    x: 7.0, y: 1.25, w: 5.8, h: 0.4,
    fontSize: 15, fontFace: F.body, color: C.muted, margin: 0,
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 7.0, y: 1.75, w: 5.8, h: 4.65,
    fill: { color: C.midnight }, line: { type: "none" },
  });

  s.addText("KV cache grows linearly with both sequence length AND batch size.", {
    x: 7.2, y: 1.95, w: 5.4, h: 1.0,
    fontSize: 16, fontFace: F.body, color: C.bgPaper, margin: 0,
  });

  s.addText([
    { text: "Doubling context → halving max batch.", options: { breakLine: true, bold: true, color: C.coral } },
    { text: "", options: { breakLine: true } },
    { text: "An 80 GB H100 holds ~7 concurrent 32k-token requests after weights and activations — not the hundreds you'd need to saturate the FLOPs.", options: { color: C.bgPaper } },
  ], {
    x: 7.2, y: 3.05, w: 5.4, h: 2.3,
    fontSize: 14, fontFace: F.body, color: C.bgPaper, margin: 0,
  });

  s.addText("The cache is the throttle, not compute.", {
    x: 7.2, y: 5.55, w: 5.4, h: 0.6,
    fontSize: 16, fontFace: F.head, color: C.coral, italic: true, margin: 0,
  });
}

// =========================================================================
// Slide 3 — Why this matters: batching throttle / 15x
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgPaper };
  pageTitle(s, "Eliminating the per-turn cache → batch up by ~15×");
  pageNumberFoot(s, 4, TOTAL);

  s.addText("Setup: 80 GB H100, Llama-class 70B, 32k effective context.", {
    x: 0.5, y: 1.2, w: 12.3, h: 0.5,
    fontSize: 14, fontFace: F.body, color: C.muted, italic: true, margin: 0,
  });

  // Two-column comparison
  // Left: status quo
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.9, w: 6.0, h: 4.6,
    fill: { color: C.bgWhite }, line: { color: C.rule, width: 0.5 },
  });
  s.addText("Status quo", {
    x: 0.7, y: 2.0, w: 5.6, h: 0.45,
    fontSize: 18, fontFace: F.head, color: C.deep, bold: true, margin: 0,
  });
  s.addText("Per-turn context = 32,000 tokens", {
    x: 0.7, y: 2.55, w: 5.6, h: 0.4,
    fontSize: 13, fontFace: F.body, color: C.ink, margin: 0,
  });
  s.addText("KV / request ≈ 32k × 320 KB  ≈  10.2 GB", {
    x: 0.7, y: 2.95, w: 5.6, h: 0.4,
    fontSize: 13, fontFace: F.mono, color: C.ink, margin: 0,
  });
  s.addText("Free VRAM after weights & activations ≈ 70 GB", {
    x: 0.7, y: 3.35, w: 5.6, h: 0.4,
    fontSize: 13, fontFace: F.body, color: C.ink, margin: 0,
  });
  s.addText("Max concurrent requests:", {
    x: 0.7, y: 4.0, w: 5.6, h: 0.4,
    fontSize: 13, fontFace: F.body, color: C.muted, margin: 0,
  });
  s.addText("≈ 7", {
    x: 0.7, y: 4.4, w: 5.6, h: 1.4,
    fontSize: 80, fontFace: F.head, color: C.midnight, bold: true, align: "center", margin: 0,
  });
  s.addText("concurrent users / 80 GB GPU", {
    x: 0.7, y: 5.85, w: 5.6, h: 0.4,
    fontSize: 12, fontFace: F.body, color: C.muted, italic: true, align: "center", margin: 0,
  });

  // Right: goldfish
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.8, y: 1.9, w: 6.0, h: 4.6,
    fill: { color: C.deep }, line: { type: "none" },
  });
  s.addText("Goldfish-2048", {
    x: 7.0, y: 2.0, w: 5.6, h: 0.45,
    fontSize: 18, fontFace: F.head, color: C.coral, bold: true, margin: 0,
  });
  s.addText("Per-turn context = 2,048 tokens", {
    x: 7.0, y: 2.55, w: 5.6, h: 0.4,
    fontSize: 13, fontFace: F.body, color: C.bgPaper, margin: 0,
  });
  s.addText("KV / request ≈ 2k × 320 KB  ≈  0.66 GB", {
    x: 7.0, y: 2.95, w: 5.6, h: 0.4,
    fontSize: 13, fontFace: F.mono, color: C.bgPaper, margin: 0,
  });
  s.addText("Same 70 GB of free VRAM available", {
    x: 7.0, y: 3.35, w: 5.6, h: 0.4,
    fontSize: 13, fontFace: F.body, color: C.bgPaper, margin: 0,
  });
  s.addText("Max concurrent requests:", {
    x: 7.0, y: 4.0, w: 5.6, h: 0.4,
    fontSize: 13, fontFace: F.body, color: "C9D6E2", margin: 0,
  });
  s.addText("≈ 105", {
    x: 7.0, y: 4.4, w: 5.6, h: 1.4,
    fontSize: 80, fontFace: F.head, color: C.coral, bold: true, align: "center", margin: 0,
  });
  s.addText("concurrent users / 80 GB GPU", {
    x: 7.0, y: 5.85, w: 5.6, h: 0.4,
    fontSize: 12, fontFace: F.body, color: "C9D6E2", italic: true, align: "center", margin: 0,
  });

  // Bottom: the 15x
  s.addText([
    { text: "= ", options: { color: C.muted } },
    { text: "~15× throughput", options: { bold: true, color: C.coral } },
    { text: "  on long-horizon agentic workloads — for the cost of redesigning the agent harness, not the model.", options: { color: C.ink } },
  ], {
    x: 0.5, y: 6.65, w: 12.3, h: 0.4,
    fontSize: 15, fontFace: F.body, color: C.ink, align: "center", margin: 0,
  });
}

// =========================================================================
// Slide 4 — The proposal
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgPaper };
  pageTitle(s, "The proposal: amnesia + a notebook");
  pageNumberFoot(s, 5, TOTAL);

  s.addText("Every turn, the model sees only the last L tokens of conversation history.", {
    x: 0.5, y: 1.2, w: 12.3, h: 0.45,
    fontSize: 15, fontFace: F.body, color: C.muted, italic: true, margin: 0,
  });

  // Three-card layout: cap / persist / prompt
  const cardY = 2.0, cardH = 4.5, cardW = 3.95;
  const xs = [0.5, 4.65, 8.8];
  const titles = ["1.  Hard window cap", "2.  Filesystem persistence", "3.  Light-touch protocol"];
  const bodies = [
    "Per-turn context is truncated to L ∈ {128, 512, 1024, 2048}.\n\nNothing older than L tokens reaches the model. No retrieval. No summarization. Just truncation.",
    "The agent has a private directory between turns: notes.md, state files, files it grep'd, the repo it's editing.\n\nFiles survive what the context window evicts.",
    "The system prompt tells the model the regime exists — that the window is small, that the filesystem persists, that thinking should go to disk rather than into hidden tokens.\n\nIt does NOT prescribe a state file format.",
  ];

  for (let i = 0; i < 3; i++) {
    s.addShape(pres.shapes.RECTANGLE, {
      x: xs[i], y: cardY, w: cardW, h: cardH,
      fill: { color: C.bgWhite }, line: { color: C.rule, width: 0.5 },
    });
    // Number badge
    s.addShape(pres.shapes.OVAL, {
      x: xs[i] + 0.25, y: cardY + 0.25, w: 0.55, h: 0.55,
      fill: { color: C.coral }, line: { type: "none" },
    });
    s.addText(`${i + 1}`, {
      x: xs[i] + 0.25, y: cardY + 0.27, w: 0.55, h: 0.5,
      fontSize: 18, fontFace: F.head, color: C.bgPaper, bold: true, align: "center", margin: 0,
    });
    s.addText(titles[i].replace(/^[0-9]+\.\s+/, ""), {
      x: xs[i] + 0.9, y: cardY + 0.27, w: cardW - 1.1, h: 0.55,
      fontSize: 17, fontFace: F.head, color: C.midnight, bold: true, margin: 0,
    });
    s.addText(bodies[i], {
      x: xs[i] + 0.25, y: cardY + 1.05, w: cardW - 0.5, h: cardH - 1.2,
      fontSize: 13, fontFace: F.body, color: C.ink, margin: 0,
      paraSpaceAfter: 6,
    });
  }
}

// =========================================================================
// Slide 5 — Comparison to prior methods
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgPaper };
  pageTitle(s, "Prior KV-eviction work answers the wrong question");
  pageNumberFoot(s, 6, TOTAL);

  s.addText("All of these compress the cache by deciding which tokens to drop, based on a model-internal signal:", {
    x: 0.5, y: 1.15, w: 12.3, h: 0.45,
    fontSize: 14, fontFace: F.body, color: C.muted, italic: true, margin: 0,
  });

  // Build comparison table
  const tableY = 1.85;
  const colW = [2.0, 3.4, 3.4, 3.5];
  const headers = ["Method", "What it keeps", "Compute cost", "What it implicitly assumes"];
  const rows = [
    ["H2O",        "Heavy-hitter tokens by attention",   "Online attention scoring",         "Attention magnitude ≈ semantic importance"],
    ["SnapKV",     "Cluster centers per attention head",  "Pre-fill clustering",              "Recent context predicts what matters"],
    ["NACL",       "Tokens by learned saliency",          "Auxiliary scorer per layer",       "Saliency transfers across tasks"],
    ["InfiniPot",  "Continual-summary in-place",          "Cheap summary update",             "Summary captures the residue"],
    ["Hashevict",  "Hash-bucketed approximation",         "Hash computation",                 "Hash collisions ≈ semantic similarity"],
    ["MorphKV",    "Morphing per-token retention",        "Learned policy",                   "Single policy generalizes"],
    ["RocketKV",   "Hierarchical token compression",      "Multi-stage scoring",              "Hierarchy aligns with usage"],
  ];

  // Header row
  let rowY = tableY;
  const xStarts = [];
  let x = 0.5;
  for (const w of colW) { xStarts.push(x); x += w; }

  for (let c = 0; c < headers.length; c++) {
    s.addShape(pres.shapes.RECTANGLE, {
      x: xStarts[c], y: rowY, w: colW[c], h: 0.45,
      fill: { color: C.midnight }, line: { type: "none" },
    });
    s.addText(headers[c], {
      x: xStarts[c] + 0.12, y: rowY, w: colW[c] - 0.2, h: 0.45,
      fontSize: 12, fontFace: F.body, color: C.bgPaper, bold: true, valign: "middle", margin: 0,
    });
  }

  // Data rows
  rowY += 0.45;
  const rowH = 0.42;
  for (let r = 0; r < rows.length; r++) {
    const alt = (r % 2 === 0) ? C.bgWhite : "EEF2F7";
    for (let c = 0; c < rows[r].length; c++) {
      s.addShape(pres.shapes.RECTANGLE, {
        x: xStarts[c], y: rowY, w: colW[c], h: rowH,
        fill: { color: alt }, line: { color: C.rule, width: 0.25 },
      });
      const isMethod = (c === 0);
      s.addText(rows[r][c], {
        x: xStarts[c] + 0.12, y: rowY, w: colW[c] - 0.2, h: rowH,
        fontSize: 11, fontFace: F.body,
        color: isMethod ? C.deep : C.ink, bold: isMethod, valign: "middle", margin: 0,
      });
    }
    rowY += rowH;
  }

  // Punchline
  rowY += 0.2;
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: rowY, w: 12.3, h: 1.05,
    fill: { color: C.deep }, line: { type: "none" },
  });
  s.addText([
    { text: "All of these use the model's own internal signal to decide what to keep — attention scores, learned scorers, hashes.\n", options: { color: C.bgPaper, breakLine: true } },
    { text: "Goldfish skips that question entirely. ", options: { bold: true, color: C.coral } },
    { text: "We ask the model to write what it wants to remember, in shell. The decision is explicit and inspectable.", options: { color: C.bgPaper } },
  ], {
    x: 0.7, y: rowY + 0.12, w: 11.9, h: 0.85,
    fontSize: 13, fontFace: F.body, color: C.bgPaper, valign: "middle", margin: 0,
  });
}

// =========================================================================
// Slide 6 — Setup
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgPaper };
  pageTitle(s, "Setup: gpt-4o on SWE-bench Verified");
  pageNumberFoot(s, 7, TOTAL);

  // Left: the bench
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.3, w: 6.0, h: 5.3,
    fill: { color: C.bgWhite }, line: { color: C.rule, width: 0.5 },
  });
  s.addText("Benchmark", {
    x: 0.7, y: 1.45, w: 5.6, h: 0.4,
    fontSize: 18, fontFace: F.head, color: C.deep, bold: true, margin: 0,
  });
  s.addText("SWE-bench Verified", {
    x: 0.7, y: 1.95, w: 5.6, h: 0.4,
    fontSize: 14, fontFace: F.body, color: C.ink, bold: true, margin: 0,
  });
  s.addText([
    { text: "• Real bugs in real Python repos (django, astropy, sympy)", options: { breakLine: true } },
    { text: "• Model produces a unified-diff patch", options: { breakLine: true } },
    { text: "• Official Docker scorer runs the held-out test suite", options: { breakLine: true } },
    { text: "• Patch passes iff every FAIL_TO_PASS test now passes and every PASS_TO_PASS test still passes", options: {} },
  ], {
    x: 0.7, y: 2.4, w: 5.6, h: 2.5,
    fontSize: 12, fontFace: F.body, color: C.ink, paraSpaceAfter: 6, margin: 0,
  });
  s.addText("L values swept:", {
    x: 0.7, y: 5.05, w: 5.6, h: 0.35,
    fontSize: 13, fontFace: F.body, color: C.muted, bold: true, margin: 0,
  });
  s.addText("L ∈ { 512, 1024, 2048, ∞ (native) }", {
    x: 0.7, y: 5.4, w: 5.6, h: 0.4,
    fontSize: 14, fontFace: F.mono, color: C.deep, bold: true, margin: 0,
  });
  s.addText("Per-turn output cap = L. Max turns = 20.", {
    x: 0.7, y: 5.8, w: 5.6, h: 0.4,
    fontSize: 12, fontFace: F.body, color: C.muted, italic: true, margin: 0,
  });

  // Right: model choice
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.7, y: 1.3, w: 6.1, h: 5.3,
    fill: { color: C.midnight }, line: { type: "none" },
  });
  s.addText("Why gpt-4o", {
    x: 6.9, y: 1.45, w: 5.7, h: 0.4,
    fontSize: 18, fontFace: F.head, color: C.coral, bold: true, margin: 0,
  });
  s.addText([
    { text: "Non-reasoning model. Hidden chain-of-thought does NOT compete with visible output for the same token budget.", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "When we tried gpt-5 with L = max_out, reasoning tokens consumed the entire per-turn budget. Every turn was 0 visible output, 100% LENGTH-CAP.", options: { breakLine: true, italic: true, color: "C9D6E2" } },
    { text: "", options: { breakLine: true } },
    { text: "Goldfish is the wrong regime for opaque-reasoner architectures. The honest story.", options: { color: C.bgPaper } },
  ], {
    x: 6.9, y: 1.95, w: 5.7, h: 3.0,
    fontSize: 13, fontFace: F.body, color: C.bgPaper, paraSpaceAfter: 4, margin: 0,
  });
  s.addText("Other choices that mattered:", {
    x: 6.9, y: 5.05, w: 5.7, h: 0.35,
    fontSize: 13, fontFace: F.body, color: C.coral, bold: true, margin: 0,
  });
  s.addText([
    { text: "• Auto-terminate when a valid unified-diff is exported", options: { breakLine: true } },
    { text: "• Unique scorer run_id per call (avoids Docker collisions)", options: {} },
  ], {
    x: 6.9, y: 5.4, w: 5.7, h: 0.9,
    fontSize: 11, fontFace: F.body, color: C.bgPaper, paraSpaceAfter: 2, margin: 0,
  });
}

// =========================================================================
// Slide 7 — Headline results: solve rate vs L
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgPaper };
  pageTitle(s, "Solve rate climbs monotonically with L");
  pageNumberFoot(s, 8, TOTAL);

  // Big bar chart, drawn manually as shapes
  const chartX = 0.8, chartY = 1.65, chartW = 6.5, chartH = 4.5;

  // Y-axis
  s.addShape(pres.shapes.LINE, {
    x: chartX, y: chartY, w: 0, h: chartH,
    line: { color: C.rule, width: 1 },
  });
  // X-axis
  s.addShape(pres.shapes.LINE, {
    x: chartX, y: chartY + chartH, w: chartW, h: 0,
    line: { color: C.rule, width: 1 },
  });

  // Y-axis labels at 0, 25, 50%
  const yLabels = [0, 25, 50];
  for (const v of yLabels) {
    const yPos = chartY + chartH - (v / 50) * chartH;
    s.addText(`${v}%`, {
      x: chartX - 0.6, y: yPos - 0.15, w: 0.5, h: 0.3,
      fontSize: 11, fontFace: F.body, color: C.muted, align: "right", margin: 0,
    });
    s.addShape(pres.shapes.LINE, {
      x: chartX, y: yPos, w: chartW, h: 0,
      line: { color: C.rule, width: 0.5, dashType: "dash" },
    });
  }

  // Data: L=512 → 0%, L=1024 → 18%, L=2048 → 25%, L=inf → 33%
  const bars = [
    { label: "L=512",   pct: 0,   n: "0/11" },
    { label: "L=1024",  pct: 18,  n: "2/11" },
    { label: "L=2048",  pct: 25,  n: "3/12" },
    { label: "L=∞",     pct: 33,  n: "4/12" },
  ];
  const barW = 1.0;
  const gap = (chartW - bars.length * barW) / (bars.length + 1);
  bars.forEach((b, i) => {
    const bx = chartX + gap + i * (barW + gap);
    const barH = (b.pct / 50) * chartH;
    const by = chartY + chartH - barH;
    const fillColor = (i === bars.length - 1) ? C.midnight : C.coral;
    if (b.pct > 0) {
      s.addShape(pres.shapes.RECTANGLE, {
        x: bx, y: by, w: barW, h: barH,
        fill: { color: fillColor }, line: { type: "none" },
      });
      // Percentage label on top of bar
      s.addText(`${b.pct}%`, {
        x: bx - 0.2, y: by - 0.42, w: barW + 0.4, h: 0.35,
        fontSize: 14, fontFace: F.head, color: C.midnight, bold: true, align: "center", margin: 0,
      });
    } else {
      // Empty bar — show a flat line
      s.addText("0%", {
        x: bx - 0.2, y: chartY + chartH - 0.4, w: barW + 0.4, h: 0.35,
        fontSize: 14, fontFace: F.head, color: C.muted, bold: true, align: "center", margin: 0,
      });
    }
    // X-axis label
    s.addText(b.label, {
      x: bx - 0.2, y: chartY + chartH + 0.05, w: barW + 0.4, h: 0.35,
      fontSize: 12, fontFace: F.body, color: C.ink, align: "center", bold: true, margin: 0,
    });
    s.addText(`(${b.n})`, {
      x: bx - 0.2, y: chartY + chartH + 0.4, w: barW + 0.4, h: 0.3,
      fontSize: 10, fontFace: F.body, color: C.muted, align: "center", margin: 0,
    });
  });

  // Chart caption
  s.addText("Solve rate · tinystate variant · gpt-4o", {
    x: chartX, y: chartY - 0.4, w: chartW, h: 0.3,
    fontSize: 11, fontFace: F.body, color: C.muted, italic: true, align: "center", margin: 0,
  });

  // Right side: key reads
  s.addShape(pres.shapes.RECTANGLE, {
    x: 7.7, y: 1.65, w: 5.2, h: 5.0,
    fill: { color: C.deep }, line: { type: "none" },
  });
  s.addText("Three things to notice", {
    x: 7.9, y: 1.8, w: 4.8, h: 0.45,
    fontSize: 16, fontFace: F.head, color: C.coral, bold: true, margin: 0,
  });

  const points = [
    ["Hard floor at L=512", "11 cells, zero solves. Below ~1k tokens the protocol cannot bootstrap — the model can't even fit its own notes back in."],
    ["L=2048 reaches 75% of native", "25% vs 33% native. Most of the capability survives a 16× context cut."],
    ["The gap is task-specific", "Three tasks solve at every L ≥ 1024. One task is native-only. Most tasks don't solve at any L — gpt-4o ceiling."],
  ];
  let py = 2.35;
  for (const [h, b] of points) {
    s.addText(h, {
      x: 7.9, y: py, w: 4.8, h: 0.35,
      fontSize: 12, fontFace: F.body, color: C.coral, bold: true, margin: 0,
    });
    s.addText(b, {
      x: 7.9, y: py + 0.35, w: 4.8, h: 0.9,
      fontSize: 11, fontFace: F.body, color: C.bgPaper, margin: 0,
    });
    py += 1.45;
  }
}

// =========================================================================
// Slide 8 — Emergent qualitative behaviors
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgPaper };
  pageTitle(s, "What the model invents when L is small");
  pageNumberFoot(s, 9, TOTAL);

  s.addText("Behaviors the prompt did NOT specify, observed in solving trajectories at L ≤ 2048:", {
    x: 0.5, y: 1.2, w: 12.3, h: 0.4,
    fontSize: 13, fontFace: F.body, color: C.muted, italic: true, margin: 0,
  });

  // Two columns: left = behavior cards, right = code sample
  const leftItems = [
    {
      title: "Comments as inline notes",
      body: "Instead of writing notes.md as instructed, the model prefixes commands with # comments. Bash ignores them — but they appear TWICE in history.txt (as input + as echo), surviving truncation more robustly than a separate file.",
    },
    {
      title: "Prose as scratch memory",
      body: "On 'hypothesis turns' (typically T3), the model writes 2-3 sentences of diagnosis. That prose primes its own next several turns — visible output doing double-duty as both reasoning and persistent state.",
    },
    {
      title: "sed → python3 fallback",
      body: "Every solving trajectory tries sed first, fails on escaped characters, then falls back to python3 - <<PY for the real edit. Across 3 L values and 3 tasks. The model learned this tool preference on the fly.",
    },
    {
      title: "Self-compression",
      body: "Avg response length DECREASES as L decreases. At L=1024 the model writes ~68 tokens/turn; at L=∞ it writes ~92. The constraint is a productivity feature, not just a memory limit.",
    },
  ];

  let py = 1.75;
  const cardH = 1.2;
  for (const it of leftItems) {
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: py, w: 7.5, h: cardH,
      fill: { color: C.bgWhite }, line: { color: C.rule, width: 0.5 },
    });
    // Coral side stripe
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: py, w: 0.12, h: cardH,
      fill: { color: C.coral }, line: { type: "none" },
    });
    s.addText(it.title, {
      x: 0.75, y: py + 0.08, w: 7.1, h: 0.35,
      fontSize: 13, fontFace: F.head, color: C.midnight, bold: true, margin: 0,
    });
    s.addText(it.body, {
      x: 0.75, y: py + 0.43, w: 7.1, h: cardH - 0.5,
      fontSize: 10.5, fontFace: F.body, color: C.ink, margin: 0,
    });
    py += cardH + 0.05;
  }

  // Right: a real code sample showing the comment-as-state pattern
  s.addShape(pres.shapes.RECTANGLE, {
    x: 8.2, y: 1.75, w: 4.65, h: 5.0,
    fill: { color: C.midnight }, line: { type: "none" },
  });
  s.addText("From a real T2 response", {
    x: 8.35, y: 1.85, w: 4.4, h: 0.35,
    fontSize: 12, fontFace: F.body, color: C.coral, bold: true, margin: 0,
  });
  s.addText("(L=1024, dj-11099, solved)", {
    x: 8.35, y: 2.18, w: 4.4, h: 0.3,
    fontSize: 10, fontFace: F.body, color: "9FB6CC", italic: true, margin: 0,
  });
  s.addText("```bash", {
    x: 8.35, y: 2.6, w: 4.4, h: 0.3,
    fontSize: 10, fontFace: F.mono, color: "9FB6CC", margin: 0,
  });
  s.addText([
    { text: "# The relevant classes are located", options: { color: C.coral, breakLine: true } },
    { text: "# in repo/django/contrib/auth/", options: { color: C.coral, breakLine: true } },
    { text: "# validators.py. Let's inspect this", options: { color: C.coral, breakLine: true } },
    { text: "# file to identify where the regex", options: { color: C.coral, breakLine: true } },
    { text: "# needs to be updated.", options: { color: C.coral, breakLine: true } },
    { text: "sed -n '1,40p' \\", options: { color: C.bgPaper, breakLine: true } },
    { text: "  repo/django/contrib/", options: { color: C.bgPaper, breakLine: true } },
    { text: "  auth/validators.py", options: { color: C.bgPaper } },
  ], {
    x: 8.45, y: 2.9, w: 4.3, h: 2.7,
    fontSize: 10, fontFace: F.mono, color: C.bgPaper, margin: 0,
  });
  s.addText("```", {
    x: 8.35, y: 5.65, w: 4.4, h: 0.3,
    fontSize: 10, fontFace: F.mono, color: "9FB6CC", margin: 0,
  });
  s.addText("The 5 comment lines carry the model's working memory across the turn boundary. Free storage.", {
    x: 8.35, y: 6.0, w: 4.4, h: 0.7,
    fontSize: 10.5, fontFace: F.body, color: "C9D6E2", italic: true, margin: 0,
  });
}

// =========================================================================
// Slide 9 — Takeaways
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgDark };
  pageTitle(s, "");  // no body title — slide is custom

  s.addText("Takeaways", {
    x: 0.5, y: 0.5, w: 12.3, h: 0.7,
    fontSize: 36, fontFace: F.head, color: C.bgPaper, bold: true, margin: 0,
  });

  const items = [
    {
      h: "Knowing what to keep is the hard problem.",
      b: "Attention-score, hash, and learned-saliency methods compress the cache by guessing what mattered. Goldfish externalizes the decision: the model writes what it wants to remember.",
    },
    {
      h: "Most native-context capability survives 16× compression.",
      b: "L=2048 ≈ 75% of L=∞ solve rate, with a per-turn KV footprint ~15× smaller. The protocol is doing real work.",
    },
    {
      h: "There is a floor.",
      b: "Below ~1k tokens the protocol cannot bootstrap; the act of reading your own notes consumes the window.",
    },
    {
      h: "Reasoning models are wrong for this regime.",
      b: "Hidden CoT competes with visible output. gpt-5 + L=128 emits zero visible tokens. Non-reasoning models cooperate; reasoning models do not.",
    },
  ];

  const cardY0 = 1.7, cardH = 1.15, gap = 0.15;
  for (let i = 0; i < items.length; i++) {
    const y = cardY0 + i * (cardH + gap);
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 12.3, h: cardH,
      fill: { color: "172A3A" }, line: { type: "none" },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 0.12, h: cardH,
      fill: { color: C.coral }, line: { type: "none" },
    });
    s.addText(items[i].h, {
      x: 0.85, y: y + 0.1, w: 11.7, h: 0.4,
      fontSize: 16, fontFace: F.head, color: C.coral, bold: true, margin: 0,
    });
    s.addText(items[i].b, {
      x: 0.85, y: y + 0.52, w: 11.7, h: cardH - 0.55,
      fontSize: 13, fontFace: F.body, color: C.bgPaper, margin: 0,
    });
  }

  s.addText("github.com/RSamavedam/goldfish-models", {
    x: 0.5, y: 6.9, w: 12.3, h: 0.4,
    fontSize: 12, fontFace: F.mono, color: "9FB6CC", italic: true, align: "right", margin: 0,
  });
}

// ----------------------------------------------------------------------
pres.writeFile({ fileName: "/Users/raghavsamavedam/Documents/153_project/goldfish-models/paper/talk/goldfish_models.pptx" })
  .then(fn => console.log("Wrote:", fn));
