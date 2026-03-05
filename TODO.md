# TODO / Bugfixes

- PECNN-Ordner unter `runs/` in PITCNN umbenennen und in den zugehoerigen `config.json`-Dateien die Strings `name` und `model_class` entsprechend von PECNN auf PITCNN anpassen.
- Dtype fuer Training/Evaluation config-gesteuert machen: `train_dtype` in der jeweiligen Run-Config definieren, in `main.py` zentral validieren und dann konsistent an Dataset, Modelle, Loss, Training und spaetere Evaluation/Prediction weitergeben; harte Dtype-Stellen (z. B. in `training/loss.py`) dabei entfernen und den gewaehlten Dtype in `config.json` des Runs abspeichern.
