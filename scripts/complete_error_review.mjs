/** Complete missing F/G fields in a private copy; never impersonate human review. */
import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import assert from "node:assert/strict";
import {fileURLToPath} from "node:url";
import {FileBlob, SpreadsheetFile} from "@oai/artifact-tool";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const relativeDir = "outputs/01a0fda5-3c22-72f3-99cd-53dca78016f4";
const dir = path.join(root, relativeDir);
const sourcePath = path.join(dir, "tuniscope_revue_erreurs_baseline_2026-10-17.xlsx");
const outputPath = path.join(dir, "tuniscope_analyse_erreurs_baseline_assistee_2026-10-17.xlsx");
const proposalPath = path.join(root, "data/interim/baseline_error_review_ai_proposals_2026-10-17.json");
const payloadPath = path.join(dir, "baseline_error_review_2026-10-17.json");
const ledgerPath = path.join(root, "data/interim/baseline_error_review_assisted_2026-10-17.json");
const hash = async (file) => crypto.createHash("sha256").update(await fs.readFile(file)).digest("hex");
const present = (v) => typeof v === "string" && Boolean(v.trim());

export function completeFields(row, proposal, categories) {
  assert.equal(row[0], proposal.case_id);
  assert(present(row[4]), "Human label opinion is required");
  assert(categories.includes(proposal.category), "Unknown proposal category");
  assert(present(proposal.hypothesis), "Missing AI hypothesis");
  if (present(row[5])) assert(categories.includes(row[5]), "Unknown existing category");
  const categoryAuthor = present(row[5]) ? "human" : "ai";
  const hypothesisAuthor = present(row[6]) ? "human" : "ai";
  return {category: present(row[5]) ? row[5] : proposal.category,
    hypothesis: present(row[6]) ? row[6] : `IA — ${proposal.hypothesis}`,
    authors: {label_opinion: "human", category: categoryAuthor, hypothesis: hypothesisAuthor}};
}

function selfTest() {
  const row = ["FAKE", "fiction", "POS", "NEG", "Incertain", "", "", "fiction:1"];
  const proposal = {case_id: "FAKE", category: "Incertain", hypothesis: "Lecture non résolue."};
  const categories = ["Incertain", "Autre"];
  assert.equal(completeFields(row, proposal, categories).authors.category, "ai");
  assert.equal(completeFields(row, proposal, categories).hypothesis, "IA — Lecture non résolue.");
  const existing = [...row]; existing[5] = "Autre"; existing[6] = "Note humaine";
  const kept = completeFields(existing, proposal, categories);
  assert.equal(kept.category, "Autre"); assert.equal(kept.hypothesis, "Note humaine");
  assert.deepEqual(kept.authors, {label_opinion: "human", category: "human", hypothesis: "human"});
  assert.throws(() => completeFields(row, {...proposal, case_id: "OTHER"}, categories));
  assert.throws(() => completeFields(row, {...proposal, category: "unknown"}, categories));
  assert.throws(() => completeFields(row, {...proposal, hypothesis: " "}, categories));
  assert.throws(() => completeFields(row.map((v,i) => i === 4 ? "" : v), proposal, categories));
  assert.deepEqual(row.slice(4,7), ["Incertain", "", ""]);
  console.log("Passed 10 fictitious assisted-completion assertions");
}

async function main() {
  const proposals = JSON.parse(await fs.readFile(proposalPath, "utf8"));
  const humanPath = path.join(root, proposals.human_snapshot);
  const human = JSON.parse(await fs.readFile(humanPath, "utf8"));
  const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
  const sourceHash = await hash(sourcePath);
  assert.equal(sourceHash, human.provenance.workbook_sha256, "Human workbook changed; import a fresh snapshot first");
  assert.equal(await hash(payloadPath), human.provenance.payload_sha256);
  assert.equal(await hash(path.join(root, "outputs/baselines/2026-10-17_grouped_v2_full/results.json")),
    human.provenance.source_run_results_sha256);
  assert.equal(proposals.human_validation_of_ai_additions, false);
  assert.equal(proposals.cases.length, 60);
  const byId = new Map(proposals.cases.map((r) => [r.case_id, r]));
  assert.equal(byId.size, 60, "Duplicate proposals");
  const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(sourcePath));
  if (process.argv.includes("--preview")) {
    for (const name of ["TSAC", "MADAR"]) {
      const blob = await wb.render({sheetName: name, range: "A8:H11", scale: 1, format: "png"});
      await fs.writeFile(path.join(dir, `assisted_before_${name}.png`), new Uint8Array(await blob.arrayBuffer()));
    }
    console.log("Rendered source ranges read-only; no XLSX written");
    return;
  }
  if (!process.argv.includes("--write")) throw new Error("Use --preview or --write");
  // Refuse overwrites, including completed work and previous provenance.
  for (const filename of [outputPath, ledgerPath]) {
    try {await fs.access(filename); throw new Error(`Output already exists: ${path.basename(filename)}`);}
    catch (error) {if (error.code !== "ENOENT") throw error;}
  }
  const sourceRows = {};
  const tasks = {};
  for (const corpus of ["tsac", "madar"]) {
    assert.equal(await hash(path.join(root, `data/manifests/v2/${corpus}.csv`)), payload.tasks[corpus].manifest_sha256);
    const sheet = wb.worksheets.getItem(corpus.toUpperCase());
    const rows = sheet.getRange("A9:H38").values;
    sourceRows[corpus] = rows;
    const original = new Map(human.tasks[corpus].cases.map((r) => [r.case_id, r]));
    const seen = new Set();
    const cases = rows.map((row, i) => {
      assert(!seen.has(row[0]), "Duplicate case"); seen.add(row[0]);
      const old = original.get(row[0]); assert(old, "Unknown case");
      assert.deepEqual(row.slice(4,7).map((v) => v ?? ""), old.raw_response_cells);
      assert.equal(row[2], old.truth); assert.equal(row[3], old.predicted);
      assert.equal(row[7], `${old.file}:${old.line}`);
      const fields = completeFields(row, byId.get(row[0]), payload.tasks[corpus].categories);
      sheet.getRange(`F${i + 9}:G${i + 9}`).values = [[fields.category, fields.hypothesis]];
      return {case_id: row[0], id: old.id, group_id: old.group_id, truth: old.truth, predicted: old.predicted,
        label_opinion: row[4], ...fields, human_validation_of_ai_additions: false,
        category_caveat: row[0] === "TSAC-26" ? "Human negation category retained despite no explicit negation visible" : null};
    });
    assert.equal(seen.size, 30);
    sheet.getRange("A4").values = [["E : avis humain. F : choix humains conservés ou propositions IA. G : hypothèses IA."]];
    sheet.getRange("G3").values = [["Cas documentés"]];
    sheet.getRange("G8").values = [["Hypothèse IA / explication courte"]];
    // Only rows with new wrapped text need extra height. Keep all column widths.
    sheet.getRange("A9:H38").format.rowHeight = 100;
    const categoryCounts = {};
    for (const c of cases) categoryCounts[c.category] = (categoryCounts[c.category] ?? 0) + 1;
    tasks[corpus] = {summary: {documented_cases: cases.length,
      human_label_opinions: cases.length,
      human_categories: cases.filter((c) => c.authors.category === "human").length,
      ai_categories: cases.filter((c) => c.authors.category === "ai").length,
      ai_hypotheses: cases.filter((c) => c.authors.hypothesis === "ai").length,
      category_counts: categoryCounts}, cases};
  }
  wb.recalculate();
  for (const name of ["TSAC", "MADAR"]) {
    assert.equal(wb.worksheets.getItem(name).getRange("H3").values[0][0], 30);
    const preview = await wb.render({sheetName: name, range: "A8:H11", scale: 1, format: "png"});
    await fs.writeFile(path.join(dir, `assisted_after_${name}.png`), new Uint8Array(await preview.arrayBuffer()));
    const caveat = await wb.render({sheetName: name, range: name === "TSAC" ? "E33:G35" : "E35:G38", scale: 1, format: "png"});
    await fs.writeFile(path.join(dir, `assisted_detail_${name}.png`), new Uint8Array(await caveat.arrayBuffer()));
  }
  const errors = await wb.inspect({kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!",
    options: {useRegex: true, maxResults: 10}, maxChars: 1000});
  const exported = await SpreadsheetFile.exportXlsx(wb);
  await exported.save(outputPath);
  const reopened = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
  for (const corpus of ["tsac", "madar"]) {
    const sheet = reopened.worksheets.getItem(corpus.toUpperCase());
    const rows = sheet.getRange("A9:H38").values;
    const formulas = sheet.getRange("A9:H38").formulas;
    rows.forEach((row,i) => {
      for (const col of [0,1,2,3,4,7]) assert.equal(row[col], sourceRows[corpus][i][col]);
      assert.deepEqual(row.slice(5,7), [tasks[corpus].cases[i].category, tasks[corpus].cases[i].hypothesis]);
      assert(!formulas[i].some(Boolean), "Unexpected formula in data");
    });
    assert.equal(sheet.getRange("H3").values[0][0], 30);
  }
  assert.equal(await hash(sourcePath), sourceHash, "Original workbook changed");
  const ledger = {schema_version: "1.0", recorded_at_utc: new Date().toISOString(),
    authorship: "Human label opinions; 11 human categories preserved; 49 AI categories and 60 AI hypotheses added",
    human_validation_of_ai_additions: false, all_cases_documented: true,
    causal_explanations_established: false, official_test_rows_loaded: 0,
    labels_predictions_manifests_modified: false, original_workbook_modified: false,
    provenance: {human_workbook_sha256: sourceHash, completed_workbook_sha256: await hash(outputPath),
      human_snapshot_sha256: await hash(humanPath), proposals_sha256: await hash(proposalPath),
      payload_sha256: await hash(payloadPath), script_sha256: await hash(fileURLToPath(import.meta.url)),
      source_run_results_sha256: human.provenance.source_run_results_sha256},
    verification: {source_and_human_fields_preserved: true, formula_counters: [30,30],
      saved_file_reopened: true, formula_error_scan: errors.ndjson}, tasks};
  await fs.writeFile(ledgerPath, JSON.stringify(ledger, null, 2) + "\n", {flag: "wx"});
  console.log(JSON.stringify({output: path.relative(root, outputPath), ledger: path.relative(root, ledgerPath),
    summary: Object.fromEntries(Object.entries(tasks).map(([c,t]) => [c,t.summary])),
    formula_error_scan: errors.ndjson}, null, 2));
}
if (process.argv.includes("--self-test")) selfTest();
else await main();
