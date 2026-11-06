/** Build a private review form from checked saved predictions, with blank human fields. */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile, FileBlob } from '@oai/artifact-tool';

const root = path.resolve(process.argv[2] ?? process.cwd());
const dir = path.join(root, 'outputs/01a0fda5-3c22-72f3-99cd-53dca78016f4/encoder_error_review_2026-11-06');
const payloadPath = path.join(dir, 'review_payload.json');
const destination = path.join(dir, 'TuniScope_revue_erreurs_encodeur.xlsx');
const digest = async p => crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');
try { await fs.access(destination); throw new Error('Preserve existing workbook and human work'); }
catch (e) { if (e.code !== 'ENOENT') throw e; }
const payload = JSON.parse(await fs.readFile(payloadPath, 'utf8'));
if (payload.reference_seed !== 42 || payload.human_fields_prefilled !== 0 || payload.official_test_rows_selected !== 0)
  throw new Error('Unexpected review protocol');
const display = raw => {
  const text = raw.replace(/_x([0-9A-Fa-f]{4})_/gu, '⟦x$1⟧');
  return /^[=+\-@]/u.test(text) ? `'${text}` : text;
};
const headers = ['Cas', 'Texte', 'Label source', 'CAMeLBERT 42', 'TF-IDF',
  'Avis sur le label', 'Difficulté supposée', 'Indice / explication courte',
  'Comparaison', 'Référence source', 'Autres graines CAMeLBERT'];
const wb = Workbook.create();
const complete = s => s.getRange('K3').values[0][0];
for (const corpus of ['tsac', 'madar']) {
  const task = payload.tasks[corpus], rows = task.cases, name = corpus.toUpperCase();
  if (rows.length !== 30 || rows.some(r => r.truth === r.predicted || r.label_opinion || r.category || r.hypothesis))
    throw new Error('Expected thirty errors with blank human fields');
  const s = wb.worksheets.add(name);
  s.showGridLines = false;
  s.getRange('A1:K38').format.font = {name: 'Arial', size: 11, color: '#243247'};
  s.getRange('A1:K38').format.verticalAlignment = 'center';
  s.getRange('A2').values = [[`TuniScope — erreurs CAMeLBERT ${name}`]];
  s.getRange('A2').format.font = {name: 'Arial', size: 16, bold: true, color: '#17324D'};
  s.getRange('A2:K2').format.rowHeight = 32;
  s.getRange('A3').values = [['30 erreurs du modèle complet, graine 42. Validation v2, texte brut.']];
  s.getRange('A4').values = [['À remplir : F = avis sur le label ; G = difficulté ; H = indice ou explication. Incertain est accepté.']];
  s.getRange('A5').values = [['Erreur commune : les deux modèles se trompent. Erreur introduite : TF-IDF a le label source, CAMeLBERT se trompe.']];
  s.getRange('A6').values = [[corpus === 'tsac'
    ? 'POS = favorable ; NEG = défavorable. Source : TSAC — https://github.com/fbougares/TSAC'
    : 'MSA = arabe standard ; TN = Tunis/Sfax ; EG = Caire. Source : https://camel.abudhabi.nyu.edu/madar-parallel-corpus/']];
  s.getRange('A7').values = [['Les colonnes I–K donnent le contexte. Vide = à revoir ; ne modifie pas les textes, labels ou prédictions.']];
  s.getRange('A3:K7').format.rowHeight = 23;
  s.getRange('J3').values = [['Cas renseignés']];
  s.getRange('K3').formulas = [['=COUNTIFS(F9:F38,"<>",G9:G38,"<>",H9:H38,"<>")']];
  s.getRange('J4').values = [['Cas à examiner']];
  s.getRange('K4').values = [[30]];
  s.getRange('K3:K4').setNumberFormat('0');
  s.getRange('A8:K8').values = [headers];
  s.getRange('A8:K8').format = {
    fill: '#17324D', font: {name: 'Arial', size: 11, bold: true, color: '#FFFFFF'},
    wrapText: true, rowHeight: 40, horizontalAlignment: 'center', verticalAlignment: 'center',
    borders: {insideVertical: {style: 'thin', color: '#FFFFFF'}},
  };
  s.getRange('A9:K38').values = rows.map(r => [r.case_id, display(r.text_raw), r.truth,
    r.predicted, r.baseline_predicted, null, null, null, r.comparison,
    `${r.source_file}:${r.source_line}`, `13 : ${r.predictions_by_seed['13']} / 2026 : ${r.predictions_by_seed['2026']}`]);
  s.getRange('B9:B38').format.wrapText = true;
  s.getRange('F9:K38').format.wrapText = true;
  s.getRange('F9:H38').format.fill = '#FFF3CF';
  s.getRange('F9:F38').dataValidation = {rule: {type: 'list', values: payload.label_opinions}};
  s.getRange('G9:G38').dataValidation = {rule: {type: 'list', values: task.categories}};
  [16, 62, 13, 15, 12, 23, 30, 43, 24, 29, 27].forEach((width, i) => {
    s.getRangeByIndexes(0, i, 38, 1).format.columnWidth = width;
  });
  s.getRange('C9:E38').format.horizontalAlignment = 'center';
  rows.forEach((r, i) => {
    const lines = display(r.text_raw).split('\n').reduce((n, line) => n + Math.max(1, Math.ceil(line.length / 45)), 0);
    s.getRange(`A${i+9}:K${i+9}`).format.rowHeight = Math.max(84, lines * 18 + 16);
    s.getRange(`B${i+9}`).format.horizontalAlignment = /[\u0600-\u06ff]/u.test(r.text_raw) ? 'right' : 'left';
  });
  s.freezePanes.freezeRows(8);
  s.freezePanes.freezeColumns(2);
  // Test that partial input is not silently counted as a completed review.
  s.getRange('F9:G9').values = [['Incertain', 'Incertain']];
  wb.recalculate();
  if (complete(s) !== 0) throw new Error('Partial review counted as complete');
  s.getRange('H9').values = [['Contrôle technique à effacer']];
  wb.recalculate();
  if (complete(s) !== 1) throw new Error('Completed review not counted');
  s.getRange('F9:H9').values = [[null, null, null]];
}
wb.recalculate();
for (const name of ['TSAC', 'MADAR']) {
  const s = wb.worksheets.getItem(name);
  if (complete(s) !== 0 || s.getRange('F9:H38').values.some(r => r.some(v => v != null && v !== '')))
    throw new Error('Human fields must remain blank');
  const preview = await wb.render({sheetName: name, range: 'A1:K11', scale: 1, format: 'png'});
  await fs.writeFile(path.join(dir, `${name.toLowerCase()}_preview.png`), new Uint8Array(await preview.arrayBuffer()));
  const rows = payload.tasks[name.toLowerCase()].cases;
  const longest = rows.reduce((best, row, i) => row.text_raw.length > rows[best].text_raw.length ? i : best, 0) + 9;
  const detail = await wb.render({sheetName: name, range: `A${longest}:H${longest}`, scale: 1, format: 'png'});
  await fs.writeFile(path.join(dir, `${name.toLowerCase()}_longest.png`), new Uint8Array(await detail.arrayBuffer()));
}
const scan = await wb.inspect({kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',
  options: {useRegex: true, maxResults: 20}, maxChars: 1500});
const file = await SpreadsheetFile.exportXlsx(wb);
await file.save(destination);
const reopened = await SpreadsheetFile.importXlsx(await FileBlob.load(destination));
for (const corpus of ['tsac', 'madar']) {
  const s = reopened.worksheets.getItem(corpus.toUpperCase());
  const actual = s.getRange('A9:K38').values, formulas = s.getRange('A9:K38').formulas;
  payload.tasks[corpus].cases.forEach((r, i) => {
    const wanted = [r.case_id, display(r.text_raw), r.truth, r.predicted, r.baseline_predicted,
      null, null, null, r.comparison, `${r.source_file}:${r.source_line}`,
      `13 : ${r.predictions_by_seed['13']} / 2026 : ${r.predictions_by_seed['2026']}`];
    if (wanted.some((v,j) => (actual[i][j] ?? null) !== v) || formulas[i].some(v => v))
      throw new Error('Source preservation or blank-input readback failed');
  });
}
await fs.writeFile(path.join(dir, 'workbook_verification.json'), JSON.stringify({
  workbook_sha256: await digest(destination), payload_sha256: await digest(payloadPath),
  builder_sha256: await digest(fileURLToPath(import.meta.url)), sheets: ['TSAC', 'MADAR'],
  cases_per_sheet: 30, human_fields_prefilled: 0, source_and_blank_fields_readback: 'passed',
  partial_and_complete_counter_test: 'passed', formula_error_scan: scan.ndjson,
  display_rule: 'Literal _xHHHH_ shown as ⟦xHHHH⟧; formula-like text escaped; exact raw text preserved in private payload',
}, null, 2) + '\n');
console.log(JSON.stringify({output: destination, cases: 60, human_fields_prefilled: 0, source_readback: 'passed', counter_test: 'passed'}));
