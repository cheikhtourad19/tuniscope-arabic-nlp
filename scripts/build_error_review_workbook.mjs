/** Private, blank human-review form; source text is never evaluated as a formula. */
import fs from "node:fs/promises";
import crypto from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const directory = path.join(root, "outputs/01a0fda5-3c22-72f3-99cd-53dca78016f4");
const inputPath = path.join(directory, "baseline_error_review_2026-10-17.json");
const indexPath = path.join(root, "data/interim/baseline_error_review_index_2026-10-17.json");
const outputPath = path.join(directory, "tuniscope_revue_erreurs_baseline_2026-10-17.xlsx");
const hash = async (filename) => crypto.createHash("sha256").update(await fs.readFile(filename)).digest("hex");
const input = JSON.parse(await fs.readFile(inputPath, "utf8"));
const index = JSON.parse(await fs.readFile(indexPath, "utf8"));
if (index.payload_sha256 !== await hash(inputPath)) throw new Error("Review payload/index checksum mismatch");
try {
  await fs.access(outputPath);
  const previous = JSON.parse(await fs.readFile(path.join(directory, "baseline_error_review_workbook_verification.json"), "utf8"));
  if (!process.argv.includes("--replace-generated") || previous.workbook_sha256 !== await hash(outputPath) ||
      previous.payload_sha256 !== await hash(inputPath) || previous.human_inputs_prefilled !== 0) {
    throw new Error("Refusing to overwrite an annotated or otherwise modified workbook");
  }
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}
const safeText = (text) => /^[=+\-@]/u.test(text) ? `'${text}` : text;
// A literal OOXML escape in the corpus could be interpreted by native Excel.
// Display it reversibly as an explicit marker; retain exact raw text in payload.
const displayText = (text) => safeText(text.replace(/_x([0-9A-Fa-f]{4})_/gu, "⟦x$1⟧"));
const sourceRef = (row) => `${row.file}:${row.line}`;
const headers = ["Cas", "Texte (lecture)", "Label source", "Prédiction", "Avis sur le label source", "Difficulté principale", "Indice / explication courte", "Référence source"];
const workbook = Workbook.create();

for (const corpus of ["tsac", "madar"]) {
  const task = input.tasks[corpus];
  const rows = task.cases;
  if (rows.length !== 30 || rows.some((r) => r.label_opinion || r.category || r.hypothesis || r.truth === r.predicted)) {
    throw new Error("Expected thirty errors and entirely blank human inputs per task");
  }
  if (task.categories.join(",").length > 255) throw new Error("Inline category validation too long");
  const name = corpus.toUpperCase();
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.getRange("A1:H38").format.font = {name: "Arial", size: 11, color: "#1F2937"};
  sheet.getRange("A1:H38").format.verticalAlignment = "center";
  sheet.getRange("A2").values = [[`TuniScope — revue des erreurs ${name}`]];
  sheet.getRange("A2").format.font = {name: "Arial", size: 16, bold: true, color: "#17324D"};
  sheet.getRange("A2:H2").format.rowHeight = 32;
  sheet.getRange("A3").values = [["30 désaccords avec le label source. Baseline caractères, texte brut, validation v2."]];
  sheet.getRange("A4").values = [["À remplir : E = avis, F = difficulté supposée, G = indice ou explication courte."]];
  sheet.getRange("A5").values = [["Vide = à revoir. Incertain est accepté. ⟦x000D⟧ signale un marqueur technique présent dans la source."]];
  sheet.getRange("A6").values = [[corpus === "tsac"
    ? "POS = appréciation favorable ; NEG = défavorable. Le label source peut être discutable."
    : "MSA = arabe standard ; TN = Tunis/Sfax ; EG = Caire. Si tu ne sais pas distinguer : Incertain."]];
  sheet.getRange("A7").values = [[corpus === "tsac"
    ? "Source : TSAC, corpus local — https://github.com/fbougares/TSAC"
    : "Source : MADAR Corpus-26, corpus local — https://camel.abudhabi.nyu.edu/madar-parallel-corpus/"]];
  sheet.getRange("A3:H7").format.rowHeight = 23;
  sheet.getRange("G3").values = [["Cas renseignés"]];
  sheet.getRange("H3").formulas = [['=COUNTIFS(E9:E38,"<>",F9:F38,"<>",G9:G38,"<>")']];
  sheet.getRange("G4").values = [["Cas à examiner"]];
  sheet.getRange("H4").values = [[30]];
  sheet.getRange("H3:H4").setNumberFormat("0");
  sheet.getRange("A8:H8").values = [headers];
  sheet.getRange("A8:H8").format = {
    fill: "#17324D", font: {name: "Arial", size: 11, bold: true, color: "#FFFFFF"},
    rowHeight: 36, wrapText: true, horizontalAlignment: "center", verticalAlignment: "center",
    borders: {insideVertical: {style: "thin", color: "#FFFFFF"}},
  };
  sheet.getRange("A9:H38").values = rows.map((r) => [r.case_id, displayText(r.text_raw), r.truth,
    r.predicted, null, null, null, sourceRef(r)]);
  sheet.getRange("A9:H38").format.rowHeight = 70;
  sheet.getRange("B9:B38").format.wrapText = true;
  sheet.getRange("E9:G38").format.fill = "#FFF3CF";
  sheet.getRange("E9:G38").format.wrapText = true;
  sheet.getRange("E9:E38").dataValidation = {rule: {type: "list", values: input.label_opinions}};
  sheet.getRange("F9:F38").dataValidation = {rule: {type: "list", values: task.categories}};
  [12, 55, 12, 12, 22, 31, 48, 27].forEach((width, i) => {
    sheet.getRangeByIndexes(0, i, 38, 1).format.columnWidth = width;
  });
  rows.forEach((row, i) => {
    sheet.getRange(`B${i + 9}`).format.horizontalAlignment = /[\u0600-\u06ff]/u.test(row.text_raw) ? "right" : "left";
  });
  sheet.getRange("C9:D38").format.horizontalAlignment = "center";
  sheet.freezePanes.freezeRows(8);
  sheet.freezePanes.freezeColumns(2);

  // A partially filled case must not count as a finished review. Restore all
  // fictitious inputs before rendering/export; no AI explanation is delivered.
  sheet.getRange("E9").values = [["Incertain"]];
  sheet.getRange("F9").values = [["Incertain"]];
  workbook.recalculate();
  if (sheet.getRange("H3").values[0][0] !== 0) throw new Error("Incomplete case counted as reviewed");
  sheet.getRange("G9").values = [["Essai technique, à effacer avant livraison"]];
  workbook.recalculate();
  if (sheet.getRange("H3").values[0][0] !== 1) throw new Error("Review counter did not update");
  sheet.getRange("E9:G9").values = [[null, null, null]];
}
workbook.recalculate();
for (const name of ["TSAC", "MADAR"]) {
  const sheet = workbook.worksheets.getItem(name);
  if (sheet.getRange("H3").values[0][0] !== 0 ||
      sheet.getRange("E9:G38").values.some((row) => row.some((v) => v != null && v !== ""))) {
    throw new Error("Human-review fields must be entirely blank");
  }
  await workbook.inspect({kind: "table", range: `${name}!C8:F11`, include: "values,formulas", tableMaxRows: 4,
    tableMaxCols: 4, maxChars: 1000});
  const preview = await workbook.render({sheetName: name, range: "A1:H12", scale: 1, format: "png"});
  await fs.writeFile(path.join(directory, `baseline_error_review_${name.toLowerCase()}_preview.png`),
    new Uint8Array(await preview.arrayBuffer()));
  const cases = input.tasks[name.toLowerCase()].cases;
  const longest = cases.reduce((best, row, i) => row.text_raw.length > cases[best].text_raw.length ? i : best, 0) + 9;
  const detail = await workbook.render({sheetName: name, range: `A${longest}:H${Math.min(38, longest + 1)}`,
    scale: 1, format: "png"});
  await fs.writeFile(path.join(directory, `baseline_error_review_${name.toLowerCase()}_detail.png`),
    new Uint8Array(await detail.arrayBuffer()));
}
const errors = await workbook.inspect({kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!",
  options: {useRegex: true, maxResults: 10}, maxChars: 1000});
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
const reopened = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
for (const corpus of ["tsac", "madar"]) {
  const sheet = reopened.worksheets.getItem(corpus.toUpperCase());
  const values = sheet.getRange("A9:H38").values;
  const formulas = sheet.getRange("A9:H38").formulas;
  for (let i = 0; i < 30; i++) {
    const expected = input.tasks[corpus].cases[i];
    if (values[i][0] !== expected.case_id || values[i][1] !== displayText(expected.text_raw) ||
        values[i][2] !== expected.truth || values[i][3] !== expected.predicted ||
        values[i][7] !== sourceRef(expected) || values[i].slice(4, 7).some((v) => v) ||
        formulas[i].some((v) => v)) throw new Error("Saved source/blank-input integrity check failed");
  }
}
await fs.writeFile(path.join(directory, "baseline_error_review_workbook_verification.json"), JSON.stringify({
  workbook_sha256: await hash(outputPath), payload_sha256: await hash(inputPath),
  builder_sha256: await hash(fileURLToPath(import.meta.url)), sheets: ["TSAC", "MADAR"],
  cases_per_sheet: 30, human_inputs_prefilled: 0, partial_completion_counter_test: "passed",
  display_rule: "Literal _xHHHH_ source sequences displayed as ⟦xHHHH⟧; exact raw strings retained in payload",
  reopened_sources_and_inputs: "passed", formula_error_scan: errors.ndjson,
}, null, 2) + "\n");
console.log(JSON.stringify({sheets: ["TSAC", "MADAR"], cases: 60, human_inputs_prefilled: 0,
  counter_test: "passed", source_readback: "passed", output: path.relative(root, outputPath)}));
