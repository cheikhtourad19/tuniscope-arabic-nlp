/** Read the human-edited XLSX, preserving it, and record private decisions. */
import fs from "node:fs/promises";
import crypto from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const directory = path.join(root, "outputs/01a0fda5-3c22-72f3-99cd-53dca78016f4");
const workbookPath = path.join(directory, "tuniscope_repetition_review_2026-10-17.xlsx");
const payloadPath = path.join(directory, "repetition_review_2026-10-17.json");
const indexPath = path.join(root, "data/interim/repetition_review_index_2026-10-17.json");
const output = path.join(root, "data/interim/repetition_review_decisions_2026-10-17.json");
const hash = async (filename) => crypto.createHash("sha256").update(await fs.readFile(filename)).digest("hex");
const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
const index = JSON.parse(await fs.readFile(indexPath, "utf8"));
if (index.payload_sha256 !== await hash(payloadPath) ||
    payload.provenance.manifest_sha256 !== await hash(path.join(root, "data/manifests/v1/tsac.csv"))) {
  throw new Error("Review provenance no longer matches original inputs");
}
const before = await hash(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const sheet = workbook.worksheets.getItem("Revue");
const rows = sheet.getRange("A9:H20").values;
const formulas = sheet.getRange("A9:H20").formulas;
const allowed = new Set(["Même sens", "Sens différent", "Incertain", "Non linguistique"]);
const safe = (value) => /^[=+\-@]/u.test(value) ? `'${value}` : value;
const sourceRef = (item) => `${item.file}:${item.line}`;
if (rows.length !== 12 || payload.pairs.length !== 12) throw new Error("Expected twelve pairs");
const decisions = rows.map((row, i) => {
  const pair = payload.pairs[i];
  if (row[0] !== pair.case_id || row[1] !== pair.family ||
      row[2] !== safe(pair.text_a) || row[3] !== safe(pair.text_b) ||
      row[6] !== sourceRef(pair.a) || row[7] !== sourceRef(pair.b) ||
      formulas[i].some((value) => !!value)) {
    throw new Error(`Changed source cells or formulas at case ${pair.case_id}`);
  }
  if (!allowed.has(row[4])) throw new Error(`Missing/invalid decision at case ${pair.case_id}`);
  if (row[5] != null && typeof row[5] !== "string") throw new Error("Notes must be text");
  return {case_id: pair.case_id, family: pair.family, a: pair.a, b: pair.b,
    decision: row[4], note: row[5] ?? "", decision_origin: "explicit_human_cell"};
});
if (before !== await hash(workbookPath)) throw new Error("Workbook changed during import");
const ledger = {
  schema_version: "1.0", recorded_at_utc: new Date().toISOString(),
  provenance: {workbook_sha256: before, payload_sha256: await hash(payloadPath),
    index_sha256: await hash(indexPath), manifest_sha256: payload.provenance.manifest_sha256},
  summary: {families: 9, decisions_recorded: decisions.length,
    counts: Object.fromEntries([...allowed].map((label) => [label, decisions.filter((p) => p.decision === label).length])),
    inferred_decisions: 0, official_test_rows_loaded: 0, workbook_modified: false},
  pairs: decisions,
};
await fs.writeFile(output, JSON.stringify(ledger, null, 2) + "\n", {encoding: "utf8", flag: "wx"});
console.log(JSON.stringify(ledger.summary));
