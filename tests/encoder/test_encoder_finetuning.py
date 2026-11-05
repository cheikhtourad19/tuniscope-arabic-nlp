"""Tiny random BERT and invented texts: gradients, accumulation and checkpoint safety."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from transformers import BertConfig,BertForSequenceClassification,BertTokenizerFast
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from encoder_finetuning import set_regime,train_epoch,evaluate,tokens,transformer_probe
from frozen_encoder import state_digest
from run_encoder_finetuning import run,validate_config
from run_frozen_encoder import save
from encoder_finetuning_inference import load_checkpoint
from verify_encoder_finetuning import verify_epoch_metrics
from baseline_metrics import classification_metrics


class EncoderFineTuningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);torch.set_num_threads(2)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.directory=Path(self.temp.name)
        v=self.directory/'vocab.txt';v.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\nc\n')
        self.tokenizer=BertTokenizerFast(vocab_file=str(v),do_lower_case=False)
        self.model=BertForSequenceClassification(BertConfig(vocab_size=8,hidden_size=8,intermediate_size=16,
             num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=128,num_labels=2,
             id2label={0:'NEG',1:'POS'},label2id={'NEG':0,'POS':1},classifier_dropout=0.))
        self.config=json.loads((ROOT/'configs/camelbert_finetuning_v1.json').read_text())
        self.config.update(batch_size=4,gradient_accumulation_steps=2)
        self.rows=[{'text_raw':'a b' if i%2 else 'c a'} for i in range(11)]
        self.x=tokens(self.rows,self.tokenizer,self.config);self.y=[i%2 for i in range(11)]

    def epoch(self,model,regime,config=None):
        set_regime(model,regime);optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
        return train_epoch(model,self.x,self.y,self.tokenizer,optimizer,config or self.config,regime,torch.device('cpu'),torch.Generator().manual_seed(42))

    def test_full_gradients_update_transformer_and_head_despite_eval_backbone(self):
        before=transformer_probe(self.model);head=state_digest(self.model.classifier)
        result=self.epoch(self.model,'finetuned')
        self.assertFalse(self.model.bert.training)
        self.assertNotEqual(before,transformer_probe(self.model));self.assertNotEqual(head,state_digest(self.model.classifier))
        self.assertGreater(result['gradients']['transformer_gradient_norm'],0)
        self.assertEqual(result['optimizer_steps'],2)

    def test_frozen_keeps_entire_backbone_and_only_updates_head(self):
        before=state_digest(self.model.bert);head=state_digest(self.model.classifier)
        result=self.epoch(self.model,'frozen')
        self.assertEqual(before,state_digest(self.model.bert));self.assertNotEqual(head,state_digest(self.model.classifier))
        self.assertFalse(result['gradients']['backbone_any_gradient'])

    def test_accumulation_matches_large_batch_including_incomplete_final_window(self):
        a=copy.deepcopy(self.model);b=copy.deepcopy(self.model)
        self.epoch(a,'finetuned')
        self.epoch(b,'finetuned',{**self.config,'batch_size':8,'gradient_accumulation_steps':1})
        for x,y in zip(a.parameters(),b.parameters()):
            self.assertTrue(torch.allclose(x,y,atol=1e-6,rtol=1e-5))

    def test_validation_no_update_or_gradient_and_valid_fixed_labels(self):
        set_regime(self.model,'finetuned');before=state_digest(self.model)
        out=evaluate(self.model,self.x,self.y,self.tokenizer,['NEG','POS'],self.config,torch.device('cpu'))
        self.assertEqual(before,state_digest(self.model));self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertEqual(out['metrics']['n'],11)
        self.assertTrue(set(out['predicted'])<= {'NEG','POS'})

    def test_native_safetensors_reload_preserves_all_predictions(self):
        initial=state_digest(self.model.bert);initial_probe=transformer_probe(self.model)
        self.epoch(self.model,'finetuned');target=self.directory/'checkpoint';self.model.save_pretrained(target,safe_serialization=True)
        self.tokenizer.save_pretrained(target)
        save(target/'tuniscope.json',{'corpus':'tsac','regime':'finetuned','seed':42,'config':self.config,'labels':['NEG','POS'],
             'initial_backbone_sha256':initial,'initial_transformer_probe_sha256':initial_probe,
             'selected_backbone_sha256':state_digest(self.model.bert),'selected_head_sha256':state_digest(self.model.classifier)})
        loaded,tokenizer,_=load_checkpoint(target,torch.device('cpu'))
        a=evaluate(self.model,self.x,self.y,self.tokenizer,['NEG','POS'],self.config,torch.device('cpu'))
        b=evaluate(loaded,self.x,self.y,tokenizer,['NEG','POS'],self.config,torch.device('cpu'))
        self.assertEqual(a['probabilities'],b['probabilities'])
        # Metadata forgery cannot silently substitute different weights.
        metadata=json.loads((target/'tuniscope.json').read_text());metadata['selected_head_sha256']='wrong';save(target/'tuniscope.json',metadata)
        with self.assertRaises(ValueError):load_checkpoint(target,torch.device('cpu'))

    def test_full_guard_before_data_or_checkpoint_load(self):
        with self.assertRaisesRegex(ValueError,'verified smoke'):
            run('full','paired',ROOT/'outputs/encoder_finetuning/test-not-created','cpu')
        self.assertFalse((ROOT/'outputs/encoder_finetuning/test-not-created').exists())

    def test_wrong_config_and_regime_rejected(self):
        with self.assertRaises(ValueError):validate_config({**self.config,'manifest_version':'v1'})
        with self.assertRaises(ValueError):set_regime(self.model,'anything')

    def test_epoch_class_metrics_recalculated_and_corruption_rejected(self):
        rows=[{'id':str(i),'group_id':str(i),'label':label} for i,label in enumerate(['NEG','NEG','POS','POS'])]
        probabilities=[[.8,.2],[.4,.6],[.2,.8],[.1,.9]]
        predicted=['NEG','POS','POS','POS'];labels=['NEG','POS']
        records=[{'id':r['id'],'group_id':r['group_id'],'truth':r['label'],'predicted':p,'probabilities':scores}
                 for r,p,scores in zip(rows,predicted,probabilities)]
        metrics=classification_metrics([r['label'] for r in rows],predicted,labels)
        self.assertAlmostEqual(metrics['macro_f1'],11/15)
        save(self.directory/'epoch_predictions.json',records)
        epoch={'validation_metrics':metrics,'validation_predictions':'epoch_predictions.json','validation_macro_f1':metrics['macro_f1']}
        self.assertTrue(verify_epoch_metrics(self.directory,epoch,rows,labels))
        epoch['validation_metrics']['per_class']['NEG']['f1-score']=0.
        with self.assertRaises(ValueError):verify_epoch_metrics(self.directory,epoch,rows,labels)

    def test_missing_legacy_epoch_metrics_not_fabricated(self):
        self.assertFalse(verify_epoch_metrics(self.directory,{'epoch':1},[],['NEG','POS']))
        with self.assertRaises(ValueError):verify_epoch_metrics(self.directory,{'validation_metrics':{}},[],['NEG','POS'])


if __name__=='__main__':unittest.main()
