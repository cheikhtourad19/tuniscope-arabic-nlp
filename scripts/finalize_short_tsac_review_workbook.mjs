/** Preserve human choices and materialize the declared blank = same-meaning rule.
 *
 * Creates a new private workbook and a text-free private decision ledger.
 * Never edits the human-filled workbook or the source corpus.
 */

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const outputDir = path.join(root, "outputs/01a0fda5-3c22-72f3-99cd-53dca78016f4");
const sourcePath = path.join(outputDir, "tuniscope_short_tsac_review_2026-10-06.xlsx");
const finalPath = path.join(outputDir, "tuniscope_short_tsac_review_decisions_2026-10-16.xlsx");
const ledgerPath = path.join(root, "data/interim/short_tsac_review_decisions_2026-10-16.json");

async function requireAbsent(filePath) {
  try {
    await fs.access(filePath);
    throw new Error(`Refusing to overwrite existing file: ${filePath}`);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
}

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

await requireAbsent(finalPath);
await requireAbsent(ledgerPath);
const sourceBytes = await fs.readFile(sourcePath);
const sourceSha256 = sha256(sourceBytes);
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(sourcePath));
const sheet = workbook.worksheets.getItem("Paires courtes");
const before = sheet.getRange("A2:K161").values;
if (before.length !== 160 || before.some((row, index) => row[0] !== index + 1)) {
  throw new Error("Expected 160 stable, sequential short-pair case IDs");
}

const allowed = new Set(["Même sens", "Sens différent", "Incertain", "Non linguistique"]);
const explicit = new Map();
const decisions = [];
let inferredCount = 0;
for (let index = 0; index < before.length; index++) {
  const row = before[index];
  const original = String(row[9] ?? "").trim();
  if (original && !allowed.has(original)) {
    throw new Error(`Unexpected decision value in case ${row[0]}`);
  }
  const inferred = original === "";
  if (inferred) {
    if (row[10] != null && String(row[10]).trim() !== "") {
      throw new Error(`Blank decision with existing note in case ${row[0]}; review manually`);
    }
    sheet.getRange(`J${index + 2}`).values = [["Même sens"]];
    sheet.getRange(`K${index + 2}`).values = [[
      "Déduit de notre consigne du 16/10/2026 : cellule vide = Même sens.",
    ]];
    inferredCount++;
  } else {
    explicit.set(row[0], original);
  }
  decisions.push({
    case_id: row[0],
    decision: inferred ? "Même sens" : original,
    origin: inferred ? "blank_interpreted_from_user_instruction" : "explicit_workbook_cell",
    test_reference: row[5],
    train_reference: row[8],
  });
}
if (inferredCount !== 77 || explicit.size !== 83) {
  throw new Error("The saved human-review counts changed; do not apply the bulk rule blindly");
}

workbook.recalculate();
const after = sheet.getRange("A2:K161").values;
for (let index = 0; index < before.length; index++) {
  const original = before[index];
  const updated = after[index];
  for (let column = 0; column < 9; column++) {
    if (original[column] !== updated[column]) {
      throw new Error(`Source or metadata changed in case ${index + 1}`);
    }
  }
  if (explicit.has(original[0]) &&
      (updated[9] !== original[9] || updated[10] !== original[10])) {
    throw new Error(`An explicit human decision changed in case ${original[0]}`);
  }
  if (!allowed.has(updated[9])) {
    throw new Error(`Missing final decision in case ${original[0]}`);
  }
}

const errorScan = await workbook.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 30 }, maxChars: 1000,
});
if (errorScan.ndjson.includes("#REF!") || errorScan.ndjson.includes("#VALUE!")) {
  throw new Error("Unexpected formula error in workbook");
}

const preview = await workbook.render({
  sheetName: "Paires courtes", range: "I78:K84", scale: 1, format: "png",
});
await fs.writeFile(path.join(outputDir, "short_tsac_final_check.png"),
  new Uint8Array(await preview.arrayBuffer()));
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(finalPath);

const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(finalPath));
const savedRows = saved.worksheets.getItem("Paires courtes").getRange("A2:K161").values;
const counts = {};
for (const row of savedRows) {
  if (!allowed.has(row[9])) throw new Error(`Saved decision missing in case ${row[0]}`);
  counts[row[9]] = (counts[row[9]] ?? 0) + 1;
}
if (savedRows.length !== 160 || counts["Même sens"] !== 152 ||
    counts["Sens différent"] !== 7 || counts["Non linguistique"] !== 1) {
  throw new Error("Saved workbook counts differ from the expected decision summary");
}

const finalSha256 = sha256(await fs.readFile(finalPath));
await fs.writeFile(ledgerPath, JSON.stringify({
  schema_version: "1.0",
  review_date: "2026-10-16",
  source_workbook_sha256: sourceSha256,
  final_workbook_sha256: finalSha256,
  rule: "Only blank decision cells were interpreted as Même sens under the user's explicit statement; explicit cells were preserved.",
  counts: { cases: 160, explicit: 83, inferred_same: 77, ...counts },
  decisions,
}, null, 2) + "\n", "utf8");

console.log(JSON.stringify({
  cases: 160, explicit: 83, inferredSame: 77, decisions: counts,
  sourceWorkbookUnchanged: sha256(await fs.readFile(sourcePath)) === sourceSha256,
}));
