/** Build a private, literal-text TSAC review workbook from fixed candidate references.
 * Run with the bundled artifact-tool Node runtime, never on a public data service.
 */
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const directory = path.join(root, "outputs/01a0fda5-3c22-72f3-99cd-53dca78016f4");
const outputPath = path.join(directory, "tuniscope_repetition_review_2026-10-17.xlsx");
const input = JSON.parse(await fs.readFile(path.join(directory, "repetition_review_2026-10-17.json"), "utf8"));
try {
  await fs.access(outputPath);
  throw new Error("Refusing to overwrite an existing review workbook");
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}
if (input.summary.families !== 9 || input.pairs.length !== 12 ||
    input.pairs.some((pair) => pair.decision || pair.note)) {
  throw new Error("Unexpected or prefilled review payload");
}

const safeText = (value) => /^[=+\-@]/u.test(value) ? `'${value}` : value;
const sourceRef = (item) => `${item.file}:${item.line}`;
const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Revue");
sheet.showGridLines = false;
sheet.getRange("A1:H20").format.font = { name: "Arial", size: 11, color: "#1F2937" };
sheet.getRange("A1:H20").format.verticalAlignment = "center";
sheet.getRange("A2").values = [["TuniScope — variantes avec lettres répétées"]];
sheet.getRange("A2").format.font = { name: "Arial", size: 16, bold: true, color: "#17324D" };
sheet.getRange("A2:H2").format.rowHeight = 32;
sheet.getRange("A3").values = [["9 familles, 12 comparaisons train/validation"]];
sheet.getRange("A4").values = [["Compare les textes A et B, puis choisis une décision dans la colonne E."]];
sheet.getRange("A5").values = [["Remplis les 12 décisions. Une cellule vide reste à revoir. En cas de doute : Incertain."]];
sheet.getRange("A6").values = [["Les prédictions et labels du modèle sont masqués. Les textes sources restent inchangés."]];
sheet.getRange("A7").values = [["Source : TSAC local, manifestes v1. Ce classeur contient des textes privés."]];
sheet.getRange("A4:H7").format.rowHeight = 23;
sheet.getRange("G3").values = [["Décisions remplies"]];
sheet.getRange("H3").formulas = [[
  '=COUNTIFS(E9:E20,"Même sens")+COUNTIFS(E9:E20,"Sens différent")+COUNTIFS(E9:E20,"Incertain")+COUNTIFS(E9:E20,"Non linguistique")'
]];
sheet.getRange("G4").values = [["Comparaisons"]];
sheet.getRange("H4").values = [[12]];
sheet.getRange("H3:H4").setNumberFormat("0");

const headers = ["Cas", "Famille", "Texte A (train)", "Texte B (validation)", "Décision", "Note / différence", "Référence A", "Référence B"];
sheet.getRange("A8:H8").values = [headers];
sheet.getRange("A8:H8").format = {
  fill: "#17324D", font: { name: "Arial", size: 11, bold: true, color: "#FFFFFF" },
  rowHeight: 32, horizontalAlignment: "center", verticalAlignment: "center",
};
sheet.getRange("A9:H20").values = input.pairs.map((pair) => [
  pair.case_id, pair.family, safeText(pair.text_a), safeText(pair.text_b), null, null,
  sourceRef(pair.a), sourceRef(pair.b),
]);
sheet.getRange("A9:H20").format.rowHeight = 48;
sheet.getRange("A9:B20").setNumberFormat("0");
sheet.getRange("C9:D20").format.wrapText = true;
sheet.getRange("E9:F20").format.fill = "#FFF3CF";
sheet.getRange("F9:F20").format.wrapText = true;
sheet.getRange("E9:E20").dataValidation = {
  rule: { type: "list", values: ["Même sens", "Sens différent", "Incertain", "Non linguistique"] },
};
const widths = [7, 9, 29, 29, 22, 31, 27, 27];
widths.forEach((width, index) => {
  sheet.getRangeByIndexes(0, index, 20, 1).format.columnWidth = width;
});
input.pairs.forEach((pair, index) => {
  for (const [column, value] of [["C", pair.text_a], ["D", pair.text_b]]) {
    sheet.getRange(`${column}${index + 9}`).format.horizontalAlignment = /[\u0600-\u06ff]/u.test(value) ? "right" : "left";
  }
  if (index && pair.family !== input.pairs[index - 1].family) {
    sheet.getRange(`A${index + 9}:H${index + 9}`).format.borders = {
      top: { style: "thin", color: "#C7D2DF" },
    };
  }
});
sheet.freezePanes.freezeRows(8);

// Check that the single progress formula responds, then restore the empty form.
sheet.getRange("E9").values = [["Incertain"]];
workbook.recalculate();
if (sheet.getRange("H3").values[0][0] !== 1) throw new Error("Review progress counter failed");
sheet.getRange("E9").values = [[null]];
workbook.recalculate();
if (sheet.getRange("H3").values[0][0] !== 0 ||
    sheet.getRange("E9:F20").values.some((row) => row.some((value) => value))) {
  throw new Error("Human decisions must remain empty");
}
await workbook.inspect({ kind: "table", range: "Revue!A8:B20", include: "values,formulas", tableMaxRows: 13, tableMaxCols: 2, maxChars: 1800 });
await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!", options: { useRegex: true, maxResults: 10 } });
const preview = await workbook.render({ sheetName: "Revue", range: "A1:H20", scale: 1, format: "png" });
await fs.writeFile(path.join(directory, "repetition_review_preview.png"), new Uint8Array(await preview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const reopened = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const saved = reopened.worksheets.getItem("Revue");
const cells = saved.getRange("A9:H20").values;
if (cells.length !== 12 || cells.some((row, index) =>
  row[0] !== input.pairs[index].case_id || row[1] !== input.pairs[index].family || row[4] || row[5])) {
  throw new Error("Saved workbook integrity check failed");
}
console.log(JSON.stringify({ workbook: path.relative(root, outputPath), families: 9, pairs: 12, decisions_prefilled: 0, saved_readback: "passed" }));
