"""Read-only inventory of training artifacts. Run from the repository root."""
import csv,json,re,subprocess
from pathlib import Path
ROOT=Path.cwd();OUT=ROOT/'reports/versuchsanalyse'
RUNS={};flat=[]
def readcsv(p):return list(csv.DictReader(p.open())) if p.exists() else []
def numeric(x):
 try:return float(x)
 except (ValueError,TypeError):return None
def phase(rid):
 d=rid.split('_')[1][:8]
 if rid.startswith('static'):return 'Statisch'
 if rid.startswith('pinn'):return 'PINN'
 for end,label in [('20260210','A: Langzeit-Sekante'),('20260220','B: normierte Quelle'),('20260223','C: zwei Vorhersagen, dt=0.1'),('20260226','D: Quelle/Laplace/dt-Umbau'),('20260305','E: V0.2/V0.3/V0.4'),('20260313','F: dt=0.01/Metadaten'),('20260323','G: Stahl-Debug'),('20260327','H: Stahl-Medium'),('20260330','I: Gewicht/Warmstart/Soft-Fehlstart')]:
  if d<end:return label
 return 'J: Soft / Physics-only'
for p in sorted((ROOT/'runs').rglob('config.json'),key=lambda p:p.parent.name):
 c=json.load(p.open());rid=c['run_id'];m=readcsv(p.with_name('metrics.csv'));parts=readcsv(p.with_name('loss_components.csv'));by={int(x['epoch']):x.copy() for x in m}
 for x in parts:by.setdefault(int(x['epoch']),{}).update(x)
 rs=[]
 for ep,x in sorted(by.items()):
  y={'epoch':ep};y.update({k:numeric(v) for k,v in x.items() if k!='epoch' and numeric(v) is not None});y['mode']=x.get('physics_residual_mode',c.get('physics_residual_mode',''))
  y['lp']=numeric(x.get('lp_weight',x.get('a',c.get('a'))));y['mw']=numeric(x.get('mse_weight',c.get('mse_weight',1)))
  y['vm']=numeric(x.get('val_mse_loss',x.get('val_data')));y['vp']=numeric(x.get('val_physics_loss',x.get('val_pde')));y['vt']=numeric(x.get('val_loss',x.get('val_total')))
  if y['vm'] is None and y['lp']==0 and y['mw']==1:y['vm']=y['vt']
  if y['mw']==0:y['vm']=None
  if y['lp']==0:y['vp']=None
  rs.append(y)
 changes=[]
 for y in rs:
  w=(y['lp'],y['mw'])
  if not changes or changes[-1]['weights']!=w:changes.append({'start':y['epoch'],'end':y['epoch'],'weights':w})
  else:changes[-1]['end']=y['epoch']
 run={'id':rid,'path':str(p.parent.relative_to(ROOT)),'config':c,'phase':phase(rid),'rows':rs,'weight_phases':changes,'metric_rows':len(m),'duplicate_epochs':len(m)-len({x['epoch'] for x in m})}
 run['raw_metrics']=m
 run['raw_components']=parts
 run['metric_seeds']=sorted({x.get('seed','') for x in m if x.get('seed')})
 RUNS[rid[-6:]]=run
 last=rs[-1] if rs else {};valid=[x for x in rs if x['vm'] is not None];best=min(valid,key=lambda x:x['vm']) if valid else {}
 row={'run_id':rid,'phase':run['phase'],'name':c.get('name',p.parent.parent.name),'model':c['model_class'],'path':run['path'],'seed':c.get('seed'),'lr':c.get('lr'),'batch':c.get('batch'),'dtype_recorded':c.get('training_dtype'),'data_path_recorded':c.get('data_path',c.get('base_path')),'experiments':c.get('selected_experiment_folders'),'samples':c.get('dataset_samples'),'epochs_requested':c.get('epochs'),'epochs_logged':len(rs),'metric_rows_raw':len(m),'duplicate_epoch_rows':run['duplicate_epochs'],'metric_seeds':','.join(run['metric_seeds']),'first_epoch':rs[0]['epoch'] if rs else None,'last_epoch':last.get('epoch'),'val_mse_last':last.get('vm'),'val_physics_last':last.get('vp'),'val_total_last':last.get('vt'),'val_mse_best':best.get('vm') if not run['duplicate_epochs'] else None,'val_mse_best_epoch':best.get('epoch') if not run['duplicate_epochs'] else None,'logged_weight_phases':json.dumps(changes),'comment':c.get('comment'),'git_commit_recorded':None,'status':('Mehrere Seeds je Epochenindex; letzte CSV-Zeile je Epoche als Anzeige' if run['duplicate_epochs'] else 'Metriken vorhanden') if rs else 'Keine protokollierte Epoche'}
 flat.append(row)
OUT.mkdir(exist_ok=True)
(OUT/'run_details.json').write_text(json.dumps(RUNS,ensure_ascii=False,indent=2))
with (OUT/'run_inventar.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=flat[0].keys());w.writeheader();w.writerows(flat)
print('Runs',len(flat),'mit Metriken',sum(x['epochs_logged']>0 for x in flat),'ohne',sum(x['epochs_logged']==0 for x in flat),'Epochen',sum(x['epochs_logged'] for x in flat))
print('HEAD',subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
