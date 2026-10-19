/** Preserve the human XLSX and record an immutable, possibly partial review. */
import fs from "node:fs/promises";
import crypto from "node:crypto";
import path from "node:path";
import assert from "node:assert/strict";
import {fileURLToPath} from "node:url";
import {FileBlob, SpreadsheetFile} from "@oai/artifact-tool";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const directory = path.join(root, "outputs/01a0fda5-3c22-72f3-99cd-53dca78016f4");
const workbookPath = path.join(directory, "tuniscope_revue_erreurs_baseline_2026-10-17.xlsx");
const payloadPath = path.join(directory, "baseline_error_review_2026-10-17.json");
const indexPath = path.join(root, "data/interim/baseline_error_review_index_2026-10-17.json");
const sha = (value) => crypto.createHash("sha256").update(value).digest("hex");
const hash = async (filename) => sha(await fs.readFile(filename));
const displayText = (raw) => {
  const text = raw.replace(/_x([0-9A-Fa-f]{4})_/gu, "⟦x$1⟧");
  return /^[=+\-@]/u.test(text) ? `'${text}` : text;
};
const nonempty = (value) => typeof value === "string" && value.trim().length > 0;

export function readCase(row, expected, opinions, categories) {
  if (row[0] !== expected.case_id || row[2] !== expected.truth || row[3] !== expected.predicted ||
      row[7] !== `${expected.file}:${expected.line}`) {
    throw new Error(`Changed case, label, prediction or reference: ${expected.case_id}`);
  }
  const original = displayText(expected.text_raw);
  let textIntegrity = "exact";
  if (row[1] !== original) {
    if (typeof row[1] !== "string" || row[1].trimEnd() !== original.trimEnd()) {
      throw new Error(`Changed source text: ${expected.case_id}`);
    }
    textIntegrity = "trailing_whitespace_only";
  }
  const raw = row.slice(4, 7).map((value) => value ?? "");
  if (raw.some((v) => typeof v !== "string")) throw new Error("Human fields must be text");
  const opinion = raw[0].trim();
  const category = raw[1].trim();
  if ((opinion && !opinions.includes(opinion)) || (category && !categories.includes(category))) {
    throw new Error(`Unknown review choice: ${expected.case_id}`);
  }
  const present = raw.filter(nonempty).length;
  return {
    case_id: expected.case_id, id: expected.id, group_id: expected.group_id,
    file: expected.file, line: expected.line, truth: expected.truth, predicted: expected.predicted,
    source_text_verification: textIntegrity, observed_display_sha256: sha(row[1]),
    label_opinion: opinion, category, hypothesis: raw[2], raw_response_cells: raw,
    status: present === 3 ? "complete" : present ? "partial" : "pending",
    inferred_fields: 0,
  };
}

export function summarize(cases) {
  const countBy = (field) => cases.reduce((out, row) => {
    const value = row[field];
    if (nonempty(value)) out[value] = (out[value] ?? 0) + 1;
    return out;
  }, {});
  return {
    total_cases: cases.length,
    label_opinions_filled: cases.filter((c) => nonempty(c.label_opinion)).length,
    categories_filled: cases.filter((c) => nonempty(c.category)).length,
    hypotheses_filled: cases.filter((c) => nonempty(c.hypothesis)).length,
    complete_cases: cases.filter((c) => c.status === "complete").length,
    partial_cases: cases.filter((c) => c.status === "partial").length,
    pending_cases: cases.filter((c) => c.status === "pending").length,
    label_opinion_counts: countBy("label_opinion"), category_counts: countBy("category"),
    trailing_whitespace_differences: cases.filter((c) => c.source_text_verification !== "exact").length,
  };
}

async function writeProgressReport(ledger, snapshotPath) {
  const suffix = ledger.provenance.workbook_sha256.slice(0, 12);
  const reportDirectory = path.join(root, `reports/error_review_progress_2026-10-17_${suffix}`);
  const summaries = Object.fromEntries(Object.entries(ledger.tasks).map(([c, task]) => [c, task.summary]));
  const data = {recorded_at_utc: ledger.recorded_at_utc, analysis_complete: ledger.analysis_complete,
    all_label_opinions_recorded: Object.values(summaries).every((s) => s.label_opinions_filled === s.total_cases),
    inferred_fields: 0, workbook_modified: false, labels_or_predictions_changed: false,
    provenance: {...ledger.provenance, snapshot_sha256: await hash(snapshotPath),
      report_generator_sha256: await hash(fileURLToPath(import.meta.url))}, summary: summaries};
  const lines = ["# TuniScope — État de la revue humaine des erreurs", "",
    "**Livrables :** L07 et L12, contributions partielles. Instantané des réponses explicites, sans correction des labels.", "",
    "## Champs renseignés", "",
    "| Corpus | Cas | Avis sur le label (E) | Catégorie (F) | Explication (G) | Analyses complètes |",
    "| --- | ---: | ---: | ---: | ---: | ---: |"];
  for (const [corpus, s] of Object.entries(summaries)) {
    lines.push(`| ${corpus.toUpperCase()} | ${s.total_cases} | ${s.label_opinions_filled} | ${s.categories_filled} | ${s.hypotheses_filled} | ${s.complete_cases} |`);
  }
  lines.push("", "## Avis sur les labels d'origine", "",
    "| Avis humain | TSAC | MADAR |", "| --- | ---: | ---: |");
  for (const opinion of ["Label plausible", "Label discutable", "Incertain", "Non interprétable"]) {
    lines.push(`| ${opinion} | ${summaries.tsac.label_opinion_counts[opinion] ?? 0} | ${summaries.madar.label_opinion_counts[opinion] ?? 0} |`);
  }
  lines.push("", "## Hypothèses de difficulté déjà choisies", "",
    "| Corpus | Catégorie choisie | Nombre |", "| --- | --- | ---: |");
  for (const [corpus, s] of Object.entries(summaries)) {
    for (const [category, count] of Object.entries(s.category_counts)) lines.push(`| ${corpus.toUpperCase()} | ${category} | ${count} |`);
  }
  lines.push("", "## Portée et travail restant", "",
    "Un avis sur le label ne fournit pas à lui seul une explication de l'erreur. Une catégorie est une hypothèse " +
    "du relecteur, pas une cause démontrée du comportement du modèle. Les cellules vides restent en attente, " +
    "sans propagation d'une convention d'un ancien lot de doublons.", "",
    "La sélection couvre des strates de confusion et d'écriture sans être proportionnelle. Les comptes ne " +
    "mesurent donc ni la fréquence des causes dans le corpus ni un taux réel de labels erronés. La revue " +
    "individuelle, avec labels et prédictions visibles, n'est pas une annotation indépendante ni un accord interannotateurs.", "",
    "Un écart d'espace final dans une cellule de lecture a été enregistré comme tel, sans changer le corpus. " +
    "Les labels, prédictions, références et contenus de lecture ont été contrôlés. Les réponses et l'état " +
    "de complétude sont conservés dans un instantané privé immuable ; le classeur n'est pas réécrit.", "",
    "L07 reste incomplet : il faut encore compléter les catégories et explications manquantes, puis " +
    "discuter les observations. Les analyses des futurs modèles et l'évaluation finale restent distinctes. " +
    "Ce rapport ne contient ni commentaires du corpus, ni références source individuelles, ni notes libres.");
  const contents = {"summary.json": JSON.stringify(data, null, 2) + "\n", "README.md": lines.join("\n") + "\n"};
  await fs.mkdir(reportDirectory, {recursive: true});
  for (const [name, content] of Object.entries(contents)) {
    const target = path.join(reportDirectory, name);
    try {await fs.writeFile(target, content, {encoding: "utf8", flag: "wx"});}
    catch (error) {
      if (error.code !== "EEXIST" || await fs.readFile(target, "utf8") !== content) throw error;
    }
  }
  return path.relative(root, reportDirectory);
}

async function main() {
  const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
  const index = JSON.parse(await fs.readFile(indexPath, "utf8"));
  const payloadHash = await hash(payloadPath);
  if (index.payload_sha256 !== payloadHash) throw new Error("Frozen review payload changed");
  const runPath = path.join(root, "outputs/baselines/2026-10-17_grouped_v2_full/results.json");
  if (await hash(runPath) !== payload.provenance.run_results_sha256) throw new Error("Reference run changed");
  const before = await hash(workbookPath);
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
  const tasks = {};
  for (const corpus of ["tsac", "madar"]) {
    if (await hash(path.join(root, `data/manifests/v2/${corpus}.csv`)) !== payload.tasks[corpus].manifest_sha256) {
      throw new Error("Manifest changed since review preparation");
    }
    const expected = new Map(payload.tasks[corpus].cases.map((row) => [row.case_id, row]));
    const sheet = workbook.worksheets.getItem(corpus.toUpperCase());
    const rows = sheet.getRange("A9:H38").values;
    const formulas = sheet.getRange("A9:H38").formulas;
    const seen = new Set();
    const cases = rows.map((row, i) => {
      if (!expected.has(row[0]) || seen.has(row[0]) || formulas[i].some((value) => value)) {
        throw new Error("Unknown/duplicate case or formula in review data");
      }
      seen.add(row[0]);
      return readCase(row, expected.get(row[0]), payload.label_opinions, payload.tasks[corpus].categories);
    });
    if (seen.size !== 30 || expected.size !== 30) throw new Error("Expected thirty cases per task");
    tasks[corpus] = {summary: summarize(cases), cases};
  }
  if (before !== await hash(workbookPath)) throw new Error("Workbook changed during read");
  const ledger = {
    schema_version: "1.0", recorded_at_utc: new Date().toISOString(),
    provenance: {workbook_sha256: before, payload_sha256: payloadHash, index_sha256: await hash(indexPath),
      source_run_results_sha256: await hash(runPath), importer_sha256: await hash(fileURLToPath(import.meta.url))},
    workbook_modified: false, raw_data_or_labels_modified: false, test_rows_loaded: 0,
    inferred_fields: 0, analysis_complete: Object.values(tasks).every((t) => t.summary.complete_cases === 30),
    tasks,
  };
  const output = path.join(root, `data/interim/baseline_error_review_decisions_2026-10-17_${before.slice(0, 12)}.json`);
  try {
    await fs.writeFile(output, JSON.stringify(ledger, null, 2) + "\n", {encoding: "utf8", flag: "wx"});
  } catch (error) {
    if (error.code !== "EEXIST") throw error;
    const old = JSON.parse(await fs.readFile(output, "utf8"));
    if (old.provenance.workbook_sha256 !== before || old.provenance.payload_sha256 !== payloadHash ||
        JSON.stringify(old.tasks) !== JSON.stringify(tasks)) throw new Error("Existing snapshot differs");
  }
  const saved = JSON.parse(await fs.readFile(output, "utf8"));
  const report = await writeProgressReport(saved, output);
  console.log(JSON.stringify({snapshot: path.relative(root, output), report, analysis_complete: ledger.analysis_complete,
    summary: Object.fromEntries(Object.entries(tasks).map(([c, t]) => [c, t.summary]))}, null, 2));
}

function selfTest() {
  const e = {case_id: "FAKE-01", id: "fake", group_id: "g", file: "fiction.txt", line: 1,
    truth: "POS", predicted: "NEG", text_raw: "fictional text "};
  const blank = [e.case_id, e.text_raw, e.truth, e.predicted, "", "", "", "fiction.txt:1"];
  const read = (row) => readCase(row, e, ["Label plausible", "Incertain"], ["Négation", "Incertain"]);
  const tests = [
    () => assert.equal(read(blank).status, "pending"),
    () => { const r = read(blank.map((v, i) => i === 4 ? "Label plausible" : v));
      assert.equal(r.status, "partial"); assert.equal(r.category, ""); assert.equal(r.hypothesis, ""); },
    () => assert.equal(read([...blank.slice(0, 4), "Incertain", "Incertain", "I do not know", blank[7]]).status, "complete"),
    () => assert.equal(read(blank.map((v, i) => i === 1 ? v.trimEnd() : v)).source_text_verification, "trailing_whitespace_only"),
    () => assert.throws(() => read(blank.map((v, i) => i === 2 ? "NEG" : v))),
    () => assert.throws(() => read(blank.map((v, i) => i === 1 ? "changed meaning" : v))),
    () => assert.throws(() => read(blank.map((v, i) => i === 4 ? "unexpected" : v))),
    () => { const s = summarize([read(blank)]); assert.equal(s.complete_cases, 0);
      assert.equal(s.label_opinions_filled, 0); assert.deepEqual(s.category_counts, {}); },
  ];
  tests.forEach((test) => test());
  console.log(`Passed ${tests.length} fictitious review-import checks`);
}

if (process.argv.includes("--self-test")) selfTest();
else await main();
