/** Build a private review workbook for short TSAC train/test candidates.
 *
 * Run from the project root with the bundled @oai/artifact-tool runtime.
 * This reads authorized local corpus files and never changes them. The workbook
 * contains protected comments and must stay under Git-ignored outputs/.
 */

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const tsacDir = path.join(root, "data/raw/tsac/original/TSAC-master");
const outputDir = path.join(root, "outputs/01a0fda5-3c22-72f3-99cd-53dca78016f4");
const outputPath = path.join(outputDir, "tuniscope_short_tsac_review_2026-10-06.xlsx");

function screeningForm(text) {
  return text.normalize("NFKC").trim().replace(/\s+/gu, " ");
}

function deletionKeys(text) {
  const points = [...text];
  const keys = new Set([text]);
  for (let index = 0; index < points.length; index++) {
    keys.add([...points.slice(0, index), ...points.slice(index + 1)].join(""));
  }
  return keys;
}

function codePointLength(text) {
  return [...text].length;
}

function safeCellText(value) {
  return /^[=+\-@]/u.test(value) ? `'${value}` : value;
}

async function loadSplit(split) {
  const result = new Map();
  for (const label of ["POS", "NEG"]) {
    const filename = `${split}_${label.toLowerCase()}.txt`;
    const content = await fs.readFile(path.join(tsacDir, filename), "utf8");
    const lines = content.replace(/^\uFEFF/u, "").split(/\r?\n/u);
    for (let index = 0; index < lines.length; index++) {
      const raw = lines[index];
      if (!raw.trim()) continue;
      const key = screeningForm(raw);
      if (!result.has(key)) result.set(key, []);
      result.get(key).push({ filename, line: index + 1, label, raw });
    }
  }
  return result;
}

function labels(rows) {
  return [...new Set(rows.map((row) => row.label))].sort().join("/");
}

function hasOppositeLabels(testRows, trainRows) {
  return testRows.some((a) => trainRows.some((b) => a.label !== b.label));
}

try {
  await fs.access(outputPath);
  throw new Error("Review workbook already exists; refusing to overwrite human decisions");
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}

const train = await loadSplit("train");
const test = await loadSplit("test");
const deletionIndex = new Map();
for (const trainText of train.keys()) {
  if (codePointLength(trainText) >= 9) continue;
  for (const key of deletionKeys(trainText)) {
    if (!deletionIndex.has(key)) deletionIndex.set(key, new Set());
    deletionIndex.get(key).add(trainText);
  }
}

const shortTest = [...test.keys()].filter((text) => codePointLength(text) < 8 && !train.has(text));
const pairs = new Map();
for (const testText of shortTest) {
  for (const key of deletionKeys(testText)) {
    for (const trainText of deletionIndex.get(key) ?? []) {
      if (testText === trainText ||
          Math.abs(codePointLength(testText) - codePointLength(trainText)) > 1) continue;
      const pairKey = JSON.stringify([testText, trainText]);
      pairs.set(pairKey, { testText, trainText });
    }
  }
}

const candidates = [...pairs.values()].map((pair) => ({
  ...pair,
  opposite: hasOppositeLabels(test.get(pair.testText), train.get(pair.trainText)),
}));
const testForms = new Set(candidates.map((pair) => pair.testText));
const trainForms = new Set(candidates.map((pair) => pair.trainText));
const oppositeCount = candidates.filter((pair) => pair.opposite).length;
if (shortTest.length !== 139 || candidates.length !== 160 || testForms.size !== 80 ||
    trainForms.size !== 142 || oppositeCount !== 16) {
  throw new Error(`Short-candidate counts differ from the documented read-only audit: ${JSON.stringify({
    shortTest: shortTest.length, candidates: candidates.length, testForms: testForms.size,
    trainForms: trainForms.size, oppositeCount,
  })}`);
}

const testGroupIds = new Map([...testForms].sort().map((text, index) => [text, index + 1]));
candidates.sort((a, b) => Number(b.opposite) - Number(a.opposite) ||
  testGroupIds.get(a.testText) - testGroupIds.get(b.testText) ||
  a.trainText.localeCompare(b.trainText));

const workbook = Workbook.create();
const guide = workbook.worksheets.add("Guide");
guide.showGridLines = false;
guide.getRange("A2").values = [["TuniScope — revue des textes TSAC courts"]];
guide.getRange("A2").format.font = { name: "Arial", size: 15, bold: true, color: "#17324D" };
guide.getRange("A4:B11").values = [
  ["Point", "À savoir"],
  ["Objectif", "Décider si chaque paire exprime le même sens malgré un caractère différent."],
  ["Priorité", "Les 16 paires avec des étiquettes source potentiellement opposées sont placées en premier."],
  ["Famille test", "Plusieurs paires peuvent partager le même texte test. Décider séparément pour chaque texte train."],
  ["Champs jaunes", "Choisir une décision et ajouter une note si le cas est ambigu ou non linguistique."],
  ["Décision", "Même sens / Sens différent / Incertain / Non linguistique. Ne pas deviner."],
  ["Jeu de test", "Le test officiel reste intact. Ce classeur ne supprime ni ne réétiquette rien."],
  ["Confidentialité", "Contient des commentaires de corpus ; ne pas ajouter à Git ni partager publiquement."],
];
guide.getRange("A4:B4").format = {
  fill: "#17324D", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  rowHeight: 28,
};
guide.getRange("A5:B11").format.font = { name: "Arial", size: 10, color: "#1F2937" };
guide.getRange("A5:B11").format.rowHeight = 35;
guide.getRange("A4:A11").format.columnWidth = 22;
guide.getRange("B4:B11").format.columnWidth = 103;

const sheet = workbook.worksheets.add("Paires courtes");
sheet.showGridLines = false;
sheet.freezePanes.freezeRows(1);
const headers = [
  "Cas", "Priorité", "Famille test", "Texte test", "Étiquette test", "Référence test",
  "Texte train", "Étiquette train", "Référence train", "Décision", "Note / justification",
];
sheet.getRange("A1:K1").values = [headers];
sheet.getRange("A1:K1").format = {
  fill: "#17324D", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  rowHeight: 38, verticalAlignment: "center",
};
sheet.getRange("A2:K161").format.font = { name: "Arial", size: 10, color: "#1F2937" };
sheet.getRange("A2:K161").format.rowHeight = 48;
sheet.getRange("A2:K161").format.verticalAlignment = "center";
const widths = [9, 15, 14, 42, 15, 24, 42, 16, 24, 21, 48];
for (let index = 0; index < widths.length; index++) {
  sheet.getRangeByIndexes(0, index, 161, 1).format.columnWidth = widths[index];
}

const rows = candidates.map((pair, index) => {
  const testRows = test.get(pair.testText);
  const trainRows = train.get(pair.trainText);
  return [
    index + 1, pair.opposite ? "À vérifier" : "Ordinaire", testGroupIds.get(pair.testText),
    safeCellText(testRows[0].raw), labels(testRows),
    `${testRows[0].filename}:${testRows[0].line} (${testRows.length} ligne(s))`,
    safeCellText(trainRows[0].raw), labels(trainRows),
    `${trainRows[0].filename}:${trainRows[0].line} (${trainRows.length} ligne(s))`, "", "",
  ];
});
sheet.getRange("A2:K161").values = rows;
sheet.getRange("D2:D161").format.wrapText = true;
sheet.getRange("G2:G161").format.wrapText = true;
sheet.getRange("J2:K161").format.fill = "#FFF3CF";
sheet.getRange("J2:J161").dataValidation = {
  rule: { type: "list", values: ["Même sens", "Sens différent", "Incertain", "Non linguistique"] },
};
sheet.getRange("B2:B17").format.fill = "#FDE2D8";

workbook.recalculate();
const check = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 1200 });
if (!check.ndjson.includes("Guide") || !check.ndjson.includes("Paires courtes")) {
  throw new Error("Workbook sheet verification failed");
}
const rowCheck = await workbook.inspect({
  kind: "table", range: "Paires courtes!A1:C18", include: "values,formulas",
  tableMaxRows: 18, tableMaxCols: 3, maxChars: 3000,
});
if (!rowCheck.ndjson.includes("À vérifier")) {
  throw new Error("Priority row verification failed");
}
await fs.mkdir(outputDir, { recursive: true });
const preview = await workbook.render({ sheetName: "Guide", range: "A2:B11", scale: 1, format: "png" });
await fs.writeFile(path.join(outputDir, "short_tsac_review_guide_preview.png"),
  new Uint8Array(await preview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({
  workbook: path.relative(root, outputPath), shortTestForms: shortTest.length,
  candidatePairs: candidates.length, testForms: testForms.size,
  trainForms: trainForms.size, priorityPairs: oppositeCount,
}));
