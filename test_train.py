import json,tempfile,unittest
from pathlib import Path
import pandas as pd
from train import train,POLLUTANTS
class TemporalTrainingTests(unittest.TestCase):
 def test_temporal_holdout_and_artifact_contract(self):
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory);rows=[]
   for year in range(2010,2015):
    for station in ['A','B']:
     for month in [1,2]:rows.append({'id':station,'date':f'01/{month:02d}/{year}',**{k:float(year-2000+month) for k in POLLUTANTS}})
   csv=folder/'measurements.csv';pd.DataFrame(rows).to_csv(csv,sep=';',index=False)
   output=folder/'model.joblib';artifact=train(csv,output);metrics=json.loads(output.with_suffix('.metrics.json').read_text())
   self.assertEqual(artifact['training_years'],[2010,2013]);self.assertEqual(metrics['temporal_test_start'],2014);self.assertEqual(metrics['test_rows'],4);self.assertEqual(set(metrics['pollutants']),set(POLLUTANTS));self.assertTrue(all('mean_baseline_mse' in result for result in metrics['pollutants'].values()))
 def test_invalid_schema_is_rejected(self):
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'invalid.csv';path.write_text('year;station\n2010;A\n')
   with self.assertRaises(ValueError):train(path,Path(directory)/'model.joblib')
