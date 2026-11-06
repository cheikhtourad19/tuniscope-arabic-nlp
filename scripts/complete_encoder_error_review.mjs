/** Complete a separate private workbook, preserving source cells and per-field authorship. */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';

const hash = async p => crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');
const filled = value => typeof value === 'string' && value.trim().length > 0;
const display = raw => {
  const text=raw.replace(/_x([0-9A-Fa-f]{4})_/gu,'⟦x$1⟧');
  return /^[=+\-@]/u.test(text) ? `'${text}` : text;
};

export function completeFields(existing, proposal, opinions, categories) {
  assert.equal(existing.length,3);
  assert(opinions.includes(proposal.label_opinion),'Unknown proposed opinion');
  assert(categories.includes(proposal.category),'Unknown proposed category');
  assert(filled(proposal.hypothesis),'Missing proposed hypothesis');
  existing.forEach(v=>assert(v==null || typeof v==='string','Existing field must be text'));
  if(filled(existing[0]))assert(opinions.includes(existing[0]),'Unknown existing opinion');
  if(filled(existing[1]))assert(categories.includes(existing[1]),'Unknown existing category');
  const proposals=[proposal.label_opinion,proposal.category,`IA — ${proposal.hypothesis}`];
  return {values:existing.map((v,i)=>filled(v)?v:proposals[i]),
          authors:existing.map(v=>filled(v)?'human_from_source':'ai')};
}

function selfTest() {
  const p={label_opinion:'Incertain',category:'Incertain',hypothesis:'Contexte insuffisant.'};
  const opinions=['Incertain','Label plausible'],categories=['Incertain','Autre'];
  const blank=completeFields(['','',''],p,opinions,categories);
  assert.deepEqual(blank.authors,['ai','ai','ai']);
  assert.equal(blank.values[2],'IA — Contexte insuffisant.');
  const existing=['Label plausible','Autre','Notre explication'];
  const complete=completeFields(existing,p,opinions,categories);
  assert.deepEqual(complete.values,existing);
  assert.deepEqual(complete.authors,['human_from_source','human_from_source','human_from_source']);
  assert.deepEqual(completeFields(['Incertain','','Note'],p,opinions,categories).authors,['human_from_source','ai','human_from_source']);
  assert.throws(()=>completeFields(['','',''],{...p,category:'unknown'},opinions,categories));
  assert.throws(()=>completeFields(['','',''],{...p,hypothesis:' '},opinions,categories));
  assert.deepEqual(existing,['Label plausible','Autre','Notre explication']);
  console.log('Passed assisted-completion preservation and validation checks');
}

async function main() {
  const root=path.resolve(process.argv[2]??process.cwd());
  const dir=path.join(root,'outputs/01a0fda5-3c22-72f3-99cd-53dca78016f4/encoder_error_review_2026-11-06');
  const source=path.join(dir,'TuniScope_revue_erreurs_encodeur.xlsx');
  const destination=path.join(dir,'TuniScope_analyse_erreurs_encodeur_assistee.xlsx');
  const ledgerPath=path.join(dir,'assisted_review_ledger.json');
  const reportDir=path.join(root,'reports/encoder_errors_assisted_2026-11-06');
  for(const filename of [destination,ledgerPath,reportDir]) {
    try {await fs.access(filename);throw new Error(`Preserve existing output: ${filename}`);}
    catch(e){if(e.code!=='ENOENT')throw e;}
  }
  const proposalPath=path.join(dir,'ai_proposals.json'),payloadPath=path.join(dir,'review_payload.json');
  const proposals=JSON.parse(await fs.readFile(proposalPath,'utf8'));
  const payload=JSON.parse(await fs.readFile(payloadPath,'utf8'));
  const sourceHash=await hash(source);
  assert.equal(sourceHash,proposals.source_workbook_sha256,'Source changed: refresh snapshot before editing');
  assert.equal(proposals.human_validation_of_ai_additions,false);
  assert.equal(proposals.causal_explanations_established,false);
  assert.equal(await hash(path.join(root,'outputs/encoder_finetuning/2026-11-05_cls_paired_mps_full/results.json')),payload.encoder_results_sha256);
  assert.equal(await hash(path.join(root,'outputs/baselines/2026-10-17_grouped_v2_full/results.json')),payload.baseline_results_sha256);
  const byId=new Map(proposals.cases.map(r=>[r.case_id,r]));
  assert.equal(byId.size,60);assert.equal(proposals.cases.length,60);
  const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(source));
  const sourceRows={},tasks={};
  for(const corpus of ['tsac','madar']) {
    assert.equal(await hash(path.join(root,`data/manifests/v2/${corpus}.csv`)),payload.tasks[corpus].manifest_sha256);
    const s=wb.worksheets.getItem(corpus.toUpperCase()),rows=s.getRange('A9:K38').values;
    sourceRows[corpus]=rows;
    const cases=payload.tasks[corpus].cases.map((expected,i)=>{
      const row=rows[i],p=byId.get(expected.case_id);assert(p,'Missing proposal');
      assert.equal(p.human_validation,false);assert.equal(p.authorship,'ai');
      const wanted=[expected.case_id,display(expected.text_raw),expected.truth,expected.predicted,expected.baseline_predicted];
      assert.deepEqual(row.slice(0,5),wanted,'Source fields changed');
      assert.equal(row[8],expected.comparison);
      assert.equal(row[9],`${expected.source_file}:${expected.source_line}`);
      assert.equal(row[10],`13 : ${expected.predictions_by_seed['13']} / 2026 : ${expected.predictions_by_seed['2026']}`);
      const before=row.slice(5,8).map(v=>v??'');
      const result=completeFields(before,p,payload.label_opinions,payload.tasks[corpus].categories);
      s.getRange(`F${i+9}:H${i+9}`).values=[result.values];
      const originalHeight=s.getRange(`A${i+9}:K${i+9}`).format.rowHeight;
      const lines=Math.max(...result.values.map(v=>v.split('\n').reduce((n,line)=>n+Math.max(1,Math.ceil(line.length/40)),0)));
      s.getRange(`A${i+9}:K${i+9}`).format.rowHeight=Math.max(originalHeight,lines*18+16);
      return {case_id:expected.case_id,id:expected.id,group_id:expected.group_id,truth:expected.truth,predicted:expected.predicted,
        comparison:expected.comparison,label_opinion:result.values[0],category:result.values[1],hypothesis:result.values[2],
        source_response_cells:before,authors:{label_opinion:result.authors[0],category:result.authors[1],hypothesis:result.authors[2]},
        priority_reservation:result.values[0]==='Label discutable',human_validation_of_ai_additions:false};
    });
    s.getRange('A2').values=[[`TuniScope — analyse assistée des erreurs ${corpus.toUpperCase()}`]];
    s.getRange('A4').values=[['F–H : propositions IA, sans validation humaine indépendante. Labels du corpus et prédictions inchangés.']];
    s.getRange('A7').values=[['F/G : avis et catégorie proposés. H : indice et hypothèse. Incertain ne signifie pas label faux.']];
    s.getRange('J3').values=[['Cas documentés']];
    s.getRange('H8').values=[['Hypothèse IA / indice']];
    const countBy=field=>cases.reduce((out,c)=>(out[c[field]]=(out[c[field]]??0)+1,out),{});
    const authorship={};
    for(const field of ['label_opinion','category','hypothesis']) {
      authorship[field]={human:cases.filter(c=>c.authors[field]==='human_from_source').length,
                        ai:cases.filter(c=>c.authors[field]==='ai').length};
    }
    tasks[corpus]={summary:{cases:cases.length,label_opinions:countBy('label_opinion'),categories:countBy('category'),
      priority_reservations:cases.filter(c=>c.priority_reservation).map(c=>c.case_id),authorship},cases};
  }
  wb.recalculate();
  for(const corpus of ['tsac','madar']) {
    const s=wb.worksheets.getItem(corpus.toUpperCase());assert.equal(s.getRange('K3').values[0][0],30);
    const preview=await wb.render({sheetName:corpus.toUpperCase(),range:'A1:K11',scale:1,format:'png'});
    await fs.writeFile(path.join(dir,`assisted_after_${corpus}.png`),new Uint8Array(await preview.arrayBuffer()));
    const longest=tasks[corpus].cases.reduce((best,c,i)=>c.hypothesis.length>tasks[corpus].cases[best].hypothesis.length?i:best,0)+9;
    const detail=await wb.render({sheetName:corpus.toUpperCase(),range:`F${longest}:H${longest}`,scale:1,format:'png'});
    await fs.writeFile(path.join(dir,`assisted_detail_${corpus}.png`),new Uint8Array(await detail.arrayBuffer()));
  }
  const scan=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10},maxChars:1200});
  const output=await SpreadsheetFile.exportXlsx(wb);await output.save(destination);
  const reopened=await SpreadsheetFile.importXlsx(await FileBlob.load(destination));
  for(const corpus of ['tsac','madar']) {
    const s=reopened.worksheets.getItem(corpus.toUpperCase()),rows=s.getRange('A9:K38').values;
    rows.forEach((r,i)=>{
      for(const col of [0,1,2,3,4,8,9,10])assert.equal(r[col],sourceRows[corpus][i][col]);
      const c=tasks[corpus].cases[i];assert.deepEqual(r.slice(5,8),[c.label_opinion,c.category,c.hypothesis]);
      assert(!s.getRange(`A${i+9}:K${i+9}`).formulas[0].some(Boolean));
    });
    assert.equal(s.getRange('K3').values[0][0],30);
  }
  assert.equal(await hash(source),sourceHash,'Original workbook changed during completion');
  const ledger={schema:'encoder-error-assisted-review-v1',human_validation_of_ai_additions:false,
    causal_explanations_established:false,original_workbook_modified:false,labels_predictions_or_manifests_modified:false,
    official_test_rows_selected:0,reference_seed:42,
    provenance:{source_workbook_sha256:sourceHash,assisted_workbook_sha256:await hash(destination),
      payload_sha256:await hash(payloadPath),proposals_sha256:await hash(proposalPath),script_sha256:await hash(fileURLToPath(import.meta.url)),
      encoder_results_sha256:payload.encoder_results_sha256,baseline_results_sha256:payload.baseline_results_sha256},
    verification:{source_and_existing_inputs_preserved:true,saved_file_reopened:true,counters:[30,30],formula_error_scan:scan.ndjson},tasks};
  await fs.writeFile(ledgerPath,JSON.stringify(ledger,null,2)+'\n',{flag:'wx'});
  const summary={schema:ledger.schema,human_validation_of_ai_additions:false,causal_explanations_established:false,
    official_test_rows_selected:0,reference_seed:42,source_workbook_sha256:sourceHash,
    assisted_workbook_sha256:ledger.provenance.assisted_workbook_sha256,payload_sha256:ledger.provenance.payload_sha256,
    proposals_sha256:ledger.provenance.proposals_sha256,script_sha256:ledger.provenance.script_sha256,
    tasks:Object.fromEntries(Object.entries(tasks).map(([c,t])=>[c,t.summary]))};
  await fs.mkdir(reportDir);
  await fs.writeFile(path.join(reportDir,'summary.json'),JSON.stringify(summary,null,2)+'\n');
  const lines=['# Revue assistée des erreurs de CAMeLBERT','',
    '6 novembre 2026. Analyse qualitative assistée par IA à notre demande, sur la sélection de validation documentée en doc50. Les propositions ajoutées ne sont pas une annotation humaine indépendante. Aucun entraînement ni changement des labels ou scores.','',
    '| Corpus | Cas documentés | Labels plausibles | Labels discutables | Incertains | Avis humains conservés | Avis IA |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'];
  for(const [corpus,t]of Object.entries(tasks))lines.push(`| ${corpus.toUpperCase()} | ${t.summary.cases} | ${t.summary.label_opinions['Label plausible']??0} | ${t.summary.label_opinions['Label discutable']??0} | ${t.summary.label_opinions.Incertain??0} | ${t.summary.authorship.label_opinion.human} | ${t.summary.authorship.label_opinion.ai} |`);
  lines.push('','## Difficultés proposées','','| Corpus | Catégorie proposée | Cas |','| --- | --- | ---: |');
  for(const [corpus,t]of Object.entries(tasks))for(const [category,n]of Object.entries(t.summary.categories))lines.push(`| ${corpus.toUpperCase()} | ${category} | ${n} |`);
  lines.push('','Ces catégories sont des hypothèses IA sur un échantillon qualitatif stratifié, pas la fréquence des causes dans le corpus ni des explications causales du réseau. Un mot présent dans une phrase ne prouve pas que le modèle l’a utilisé.','',
    'Les trois réserves prioritaires TSAC portent sur une lecture littérale favorable ou dépréciative qui semble contredire le label, avec ambiguïté de transcription ou d’ironie. Il ne s’agit pas de trois erreurs d’annotation établies. Les phrases MADAR jugées incertaines manquent souvent de marqueur exclusif : la provenance peut être correcte même si la phrase ressemble au standard ou à une autre variété.','',
    'Le classeur original reste intact. La copie assistée, les textes, propositions par cas et le registre d’origine de chaque champ restent Git-ignorés. Seuls ces agrégats sont publiables dans le dépôt. Le compteur indique des cas documentés et ne mesure pas des validations humaines.','',
    'Contributions partielles à L07/L10 : 60 cas documentés avec assistance déclarée, vérification technique du classeur et limites explicites. Relecture humaine des propositions, analyse des composants finaux et autres exigences globales restent distinctes.');
  await fs.writeFile(path.join(reportDir,'README.md'),lines.join('\n')+'\n');
  console.log(JSON.stringify({output:destination,original_preserved:true,summaries:summary.tasks,formula_error_scan:scan.ndjson},null,2));
}

if(process.argv.includes('--self-test'))selfTest();else await main();
