from pathlib import Path
import numpy as np,json,zarr
base=Path('data/old/new_detailed_heat_sim_f64');d=json.load((base/'normalization_values.json').open());R=d['max_temp']-d['min_temp'];rows=[]
for p in sorted(base.glob('experiment*')):
 stores=list(p.glob('*.zarr'))
 if not stores:continue
 g=zarr.open_group(str(stores[0]),mode='r');a=np.asarray(g['temperature'][0]);b=np.asarray(g['temperature'][1]);mask=np.ones_like(a,dtype=bool);mask[1:-1,1:-1,1:-1]=False
 rows.append({'experiment':p.name,'boundary_mse_floor':float(np.sum(((a-b)/R)[mask]**2)/a.size),'hot_boundary_cells':int(np.sum((a>500)&mask))})
print('OLD',len(rows),'mean boundary floor',np.mean([x['boundary_mse_floor'] for x in rows]),'min',min(x['boundary_mse_floor'] for x in rows),'max',max(x['boundary_mse_floor'] for x in rows));print('dates',sorted(set(x['experiment'].split('_')[2] for x in rows)))
Path('reports/versuchsanalyse/old_dataset_boundary_checks.json').write_text(json.dumps(rows,indent=2))
