/** Build a private, Git-ignored workbook for human audit decisions.
 *
 * Run from the project root with a Node environment containing
 * @oai/artifact-tool. The output is deliberately kept under outputs/.
 * This script never changes the source corpora or applies review decisions.
 */

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const outputDir = path.join(root, "outputs", "01a0fda5-3c22-72f3-99cd-53dca78016f4");
const outputPath = path.join(outputDir, "tuniscope_human_review_2026-10-04.xlsx");
const madarDir = path.join(root, "data/raw/madar/original/MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/MADAR_Corpus");
const tsacDir = path.join(root, "data/raw/tsac/original/TSAC-master");
const exactIndex = path.join(root, "data/interim/exact_review_2026-10-04.tsv");
const nearIndex = path.join(root, "data/interim/tsac_near_review_2026-10-04.tsv");

function parseTsv(content) {
  const [header, ...rows] = content.trimEnd().split(/\r?\n/);
  const names = header.split("\t");
  return rows.map((line) => Object.fromEntries(names.map((name, i) => [name, line.split("\t")[i] ?? ""])));
}

const sourceCache = new Map();
async function sourceText(source, filename, lineNumber) {
  if (path.basename(filename) !== filename) throw new Error("Invalid source filename");
  const allowed = source === "TSAC"
    ? /^(train|test)_(pos|neg)\.txt$/.test(filename)
    : /^MADAR\.corpus\.(MSA|Tunis|Sfax|Cairo)\.tsv$/.test(filename);
  if (!allowed) throw new Error("Unexpected source filename");
  const fullPath = path.join(source === "TSAC" ? tsacDir : madarDir, filename);
  if (!sourceCache.has(fullPath)) {
    sourceCache.set(fullPath, (await fs.readFile(fullPath, "utf8")).replace(/^\uFEFF/, "").split(/\r?\n/));
  }
  const line = sourceCache.get(fullPath)[Number(lineNumber) - 1];
  if (line === undefined) throw new Error("Indexed source line is missing");
  if (source === "TSAC") return line;
  const fields = line.split("\t");
  if (fields.length !== 4) throw new Error("Malformed indexed MADAR line");
  return fields[3];
}

// Keep untrusted corpus strings as literal cell values, never formulas.
function cellText(value) {
  return /^[=+\-@]/.test(value) ? `'${value}` : value;
}

function styleSheet(sheet, headers, rowCount, widths) {
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.getRangeByIndexes(0, 0, 1, headers.length).values = [headers];
  sheet.getRangeByIndexes(0, 0, 1, headers.length).format = {
    fill: "#17324D",
    font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
    rowHeight: 34,
    verticalAlignment: "center",
  };
  const body = sheet.getRangeByIndexes(1, 0, rowCount, headers.length);
  body.format.font = { name: "Arial", size: 10, color: "#1F2937" };
  body.format.rowHeight = 54;
  body.format.verticalAlignment = "center";
  for (let i = 0; i < widths.length; i++) {
    sheet.getRangeByIndexes(0, i, rowCount + 1, 1).format.columnWidth = widths[i];
  }
}

await fs.access(exactIndex);
await fs.access(nearIndex);
try {
  await fs.access(outputPath);
  throw new Error("Review workbook already exists; refusing to overwrite human decisions");
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}

const exactRows = parseTsv(await fs.readFile(exactIndex, "utf8"));
const nearRows = parseTsv(await fs.readFile(nearIndex, "utf8"));
const conflicts = exactRows.filter((row) => row.category === "exact_pos_neg_conflict");
if (conflicts.length !== 13 || nearRows.length !== 124) {
  throw new Error("Review index counts differ from the verified audit");
}

const workbook = Workbook.create();
const guide = workbook.worksheets.add("Guide");
guide.showGridLines = false;
guide.getRange("A2").values = [["TuniScope — revue humaine des données"]];
guide.getRange("A2").format.font = { name: "Arial", size: 15, bold: true, color: "#17324D" };
guide.getRange("A4:B12").values = [
  ["Onglet", "Action"],
  ["Conflits TSAC", "Lire le commentaire et choisir POS, NEG, Ambigu ou À revoir."],
  ["Quasi-doublons", "Comparer les deux commentaires puis choisir Même sens, Sens différent ou Incertain."],
  ["Champs jaunes", "Seules les décisions et notes jaunes sont à remplir. Les étiquettes source restent inchangées."],
  ["Cas TSAC", "13 groupes avec étiquettes POS et NEG contradictoires. Une ligne représente un texte exact."],
  ["Candidats proches", "124 paires proposées par un filtre automatique ; elles ne sont pas toutes des doublons."],
  ["Autres doublons", "316 groupes train/test TSAC et 4 croisements MADAR : politique commune à fixer, sans revue ligne par ligne."],
  ["Jeu de test", "Le test officiel reste intact. Ce fichier ne change aucune donnée et ne prépare aucun entraînement."],
  ["Confidentialité", "Le classeur contient des textes de corpus et ne doit pas être ajouté à Git ni partagé publiquement."],
];
guide.getRange("A4:B4").format = {
  fill: "#17324D", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, rowHeight: 28,
};
guide.getRange("A5:B12").format.font = { name: "Arial", size: 10, color: "#1F2937" };
guide.getRange("A5:B12").format.rowHeight = 34;
guide.getRange("A4:A12").format.columnWidth = 21;
guide.getRange("B4:B12").format.columnWidth = 100;

const conflictSheet = workbook.worksheets.add("Conflits TSAC");
const conflictHeaders = ["Cas", "Commentaire exact", "Source POS", "Source NEG", "Lignes du groupe", "Étiquettes source", "Décision", "Note / justification"];
styleSheet(conflictSheet, conflictHeaders, conflicts.length, [9, 75, 27, 27, 18, 20, 18, 56]);
const conflictValues = [];
for (const row of conflicts) {
  const textA = await sourceText("TSAC", row.file_a, row.line_a);
  const textB = await sourceText("TSAC", row.file_b, row.line_b);
  if (textA !== textB) throw new Error("Conflict index does not point to identical text");
  conflictValues.push([
    Number(row.case_id), cellText(textA), `${row.file_a}:${row.line_a}`,
    `${row.file_b}:${row.line_b}`, Number(row.group_occurrences),
    `${row.label_a} / ${row.label_b}`, "", "",
  ]);
}
conflictSheet.getRange(`A2:H${conflicts.length + 1}`).values = conflictValues;
conflictSheet.getRange(`B2:B${conflicts.length + 1}`).format.wrapText = true;
conflictSheet.getRange(`G2:H${conflicts.length + 1}`).format.fill = "#FFF3CF";
conflictSheet.getRange(`G2:G${conflicts.length + 1}`).dataValidation = {
  rule: { type: "list", values: ["POS", "NEG", "Ambigu", "À revoir"] },
};

const nearSheet = workbook.worksheets.add("Quasi-doublons");
const nearHeaders = ["Cas", "Similarité", "Texte train", "Référence train", "Texte test", "Référence test", "Labels source", "Décision", "Note / différence"];
styleSheet(nearSheet, nearHeaders, nearRows.length, [9, 13, 64, 26, 64, 26, 19, 19, 52]);
const sortedNear = [...nearRows].sort((a, b) => {
  const aOpposite = a.train_labels !== a.test_labels ? 1 : 0;
  const bOpposite = b.train_labels !== b.test_labels ? 1 : 0;
  return bOpposite - aOpposite || Number(b.similarity) - Number(a.similarity);
});
const nearValues = [];
for (const row of sortedNear) {
  nearValues.push([
    Number(row.case_id), Number(row.similarity),
    cellText(await sourceText("TSAC", row.train_file, row.train_line)),
    `${row.train_file}:${row.train_line}`,
    cellText(await sourceText("TSAC", row.test_file, row.test_line)),
    `${row.test_file}:${row.test_line}`,
    `${row.train_labels} / ${row.test_labels}`, "", "",
  ]);
}
nearSheet.getRange(`A2:I${nearRows.length + 1}`).values = nearValues;
nearSheet.getRange(`B2:B${nearRows.length + 1}`).setNumberFormat("0.0000");
nearSheet.getRange(`C2:C${nearRows.length + 1}`).format.wrapText = true;
nearSheet.getRange(`E2:E${nearRows.length + 1}`).format.wrapText = true;
nearSheet.getRange(`H2:I${nearRows.length + 1}`).format.fill = "#FFF3CF";
nearSheet.getRange(`H2:H${nearRows.length + 1}`).dataValidation = {
  rule: { type: "list", values: ["Même sens", "Sens différent", "Incertain"] },
};

workbook.recalculate();
const check = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 1200 });
if (!check.ndjson.includes("Conflits TSAC") || !check.ndjson.includes("Quasi-doublons")) {
  throw new Error("Workbook sheet verification failed");
}
await fs.mkdir(outputDir, { recursive: true });
const preview = await workbook.render({ sheetName: "Guide", range: "A2:B12", scale: 1, format: "png" });
await fs.writeFile(path.join(outputDir, "review_guide_preview.png"), new Uint8Array(await preview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`Created Git-ignored review workbook with ${conflicts.length} conflicts and ${nearRows.length} near-duplicate candidates.`);
