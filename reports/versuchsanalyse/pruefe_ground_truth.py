from pathlib import Path
import numpy as np,zarr,json,csv
results={}
for dirname in ['steel_debug_small','steel_train_medium']:
 base=Path('data')/dirname;info=json.load((base/'info.json').open()); R=info['temp_range'];dt=info['dt']; alpha=info['alpha']; spacing=[info[x] for x in ['dx','dy','dz']]
 vals=[]
 for folder in sorted(base.glob('experiment*'))[:3]:
  store=next(folder.glob('*.zarr'));g=zarr.open_group(str(store),mode='r');print(dirname,folder.name,'keys',list(g.array_keys()),'shape',g['temperature'].shape)
  q=np.asarray(g['source_field'])/(info['rho']*info['cp']*R)
  def norm(x):return (np.asarray(x,dtype=np.float64)-info['min_temp'])/R
  def lap(u):return sum((np.take(u,range(2,u.shape[d]),axis=d)[tuple(slice(1,-1) if j!=d else slice(None) for j in range(3))]-2*u[1:-1,1:-1,1:-1]+np.take(u,range(0,u.shape[d]-2),axis=d)[tuple(slice(1,-1) if j!=d else slice(None) for j in range(3))])/(spacing[d]**2) for d in range(3))
  u0=norm(g['temperature'][0]);
  for n in [1,100,600,1200]:
   a=norm(g['temperature'][n-1]);b=norm(g['temperature'][n]);c=norm(g['temperature'][n+1]);inn=(slice(1,-1),)*3
   rp=(b-a)[inn]/dt-alpha*lap(a)-q[inn];rf=(c-b)[inn]/dt-alpha*lap(b)-q[inn];soft=(b-a)[inn]/dt-alpha*lap(u0)
   row={'experiment':folder.name,'n':n,'past_mse':float(np.mean(rp**2)),'future_mse':float(np.mean(rf**2)),'soft_on_truth_mse':float(np.mean(soft**2))}
   if n>=120:
    wrong=(b-norm(g['temperature'][n-120]))[inn]/dt-alpha*lap(norm(g['temperature'][n-120]))-q[inn];row['wrong_time_gap_mse']=float(np.mean(wrong**2))
   vals.append(row)
 results[dirname]=vals
print(json.dumps(results,indent=2))
Path('reports/versuchsanalyse/ground_truth_residual_checks.json').write_text(json.dumps(results,indent=2))
vol=64*32*16;interior=62*30*14;fac=vol/interior/.1**2
print('FACTOR',fac)
for p in Path('runs').rglob('loss_components.csv'):
 rows=list(csv.DictReader(p.open()))
 if rows and rows[-1].get('physics_residual_mode')=='gt_past':
  r=rows[-1];m=float(r['val_mse_loss']);ph=float(r['val_physics_loss']);print(p.parent.name,'ratio',ph/m if m else None,'inferred_mse',ph/fac)
