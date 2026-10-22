# Aktueller Trainingsstatus

Stand: 2026-03-27

Nachtrag: 2026-03-30

Diese Datei dokumentiert, welche Slurm-Jobs aktuell eingereicht wurden, was sie trainieren und wo ihre Ergebnisse bzw. Logs gespeichert werden.

## Slurm-Logs

Alle Slurm-Logs werden durch `slurm/main.slurm` hier gespeichert:

- `logs/%x_%j.out`
- `logs/%x_%j.err`

Dabei ist:

- `%x` = Job-Name
- `%j` = Slurm-Job-ID

## Aktuell eingereichte Jobs

### 1. 14 Epochen, `lp_weight = 0.01`, `physics_residual_mode = gt_future`

- Slurm-Job-ID: `949151`
- Job-Name: `tf_gfut14`
- Config: `configs/pitcnn_timefirst_steel_medium_gt_future_100ep.py`
- Training:
  - Datensatz: `./data/steel_train_medium/`
  - Modellklasse: `PITCNN_dynamic`
  - Epochen: `14`
  - Loss-Schedule: 14 Epochen mit `lp_weight = 0.01`, `mse_weight = 1.0`
  - Physics-Residual: `gt_future`
- Modellname / Zielordner:
  - Modellwurzel: `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_BOTH_gt_future_14ep_lp001/`
  - Neuer Run-Ordner: `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_BOTH_gt_future_14ep_lp001/<run_id>/`
- Erwartete Artefakte im Run-Ordner:
  - `config.json`
  - `metrics.csv`
  - `loss_components.csv`
  - `<run_id>.pth`
  - `epoch_*.pth`
  - `losses_plot_<run_id>.png`
  - `proc_time_<run_id>.txt`

### 2. 14 Epochen, `lp_weight = 0.01`, `physics_residual_mode = gt_past`

- Slurm-Job-ID: `949149`
- Job-Name: `tf_gpast14`
- Config: `configs/pitcnn_timefirst_steel_medium_gt_past_100ep.py`
- Training:
  - Datensatz: `./data/steel_train_medium/`
  - Modellklasse: `PITCNN_dynamic`
  - Epochen: `14`
  - Loss-Schedule: 14 Epochen mit `lp_weight = 0.01`, `mse_weight = 1.0`
  - Physics-Residual: `gt_past`
- Modellname / Zielordner:
  - Modellwurzel: `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_BOTH_gt_past_14ep_lp001/`
  - Neuer Run-Ordner: `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_BOTH_gt_past_14ep_lp001/<run_id>/`
- Erwartete Artefakte im Run-Ordner:
  - `config.json`
  - `metrics.csv`
  - `loss_components.csv`
  - `<run_id>.pth`
  - `epoch_*.pth`
  - `losses_plot_<run_id>.png`
  - `proc_time_<run_id>.txt`

### 3. 14 Epochen, `lp_weight = 0.01`, `physics_residual_mode = model_past`

- Slurm-Job-ID: `949150`
- Job-Name: `tf_mpast14`
- Config: `configs/pitcnn_timefirst_steel_medium_model_past_100ep.py`
- Training:
  - Datensatz: `./data/steel_train_medium/`
  - Modellklasse: `PITCNN_dynamic`
  - Epochen: `14`
  - Loss-Schedule: 14 Epochen mit `lp_weight = 0.01`, `mse_weight = 1.0`
  - Physics-Residual: `model_past`
- Modellname / Zielordner:
  - Modellwurzel: `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_BOTH_model_past_14ep_lp001/`
  - Neuer Run-Ordner: `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_BOTH_model_past_14ep_lp001/<run_id>/`
- Erwartete Artefakte im Run-Ordner:
  - `config.json`
  - `metrics.csv`
  - `loss_components.csv`
  - `<run_id>.pth`
  - `epoch_*.pth`
  - `losses_plot_<run_id>.png`
  - `proc_time_<run_id>.txt`

### 4. MSE-only-Run von 14 auf 20 Gesamtepochen fortsetzen

- Slurm-Job-ID: `949146`
- Job-Name: `tf_mse20r`
- Config: `configs/pitcnn_timefirst_steel_medium_mse_only_100ep.py`
- Training:
  - Datensatz: `./data/steel_train_medium/`
  - Modellklasse: `PITCNN_dynamic`
  - Resume-Quelle: bestehender Run `dynamic_20260323-104107_7e2ea6`
  - Ziel: von bereits trainierten 14 Epochen auf insgesamt 20 Epochen fortsetzen
  - Loss-Schedule: 20 Epochen mit `lp_weight = 0.0`, `mse_weight = 1.0`
  - Physics-Residual: `gt_future` (formal gesetzt, aber effektiv irrelevant, da `lp_weight = 0.0`)
- Speicherort:
  - Dieser Job schreibt **nicht** in einen neuen Ordner.
  - Er trainiert im bestehenden Run weiter unter:
  - `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_MSE_ONLY_100ep/dynamic_20260323-104107_7e2ea6/`
- Erwartete aktualisierte Artefakte:
  - `config.json`
  - `metrics.csv`
  - `loss_components.csv`
  - `dynamic_20260323-104107_7e2ea6.pth`
  - zusätzliche `epoch_*.pth`
  - aktualisierte Loss-Plots / Laufzeitdateien

### 5. Follow-up ab MSE-Checkpoint: 10 weitere Epochen mit `gt_future`, `lp_weight = 1.0`

- Slurm-Job-ID: `949148`
- Job-Name: `tf_gfut30b`
- Config: `configs/pitcnn_timefirst_steel_medium_from_mse20_to_gt_future_10ep.py`
- Abhängigkeit:
  - Startet nur, wenn Job `949146` erfolgreich beendet wurde (`afterok:949146`).
- Training:
  - Datensatz: `./data/steel_train_medium/`
  - Modellklasse: `PITCNN_dynamic`
  - Initialisierung aus Checkpoint des Runs `dynamic_20260323-104107_7e2ea6`
  - Gesamtschedule: 30 Epochen
  - Davon:
    - Epochen 1-20: `lp_weight = 0.0`, `mse_weight = 1.0`
    - Epochen 21-30: `lp_weight = 1.0`, `mse_weight = 1.0`
  - Physics-Residual: `gt_future`
- Speicherort:
  - Dieser Job schreibt in einen **neuen** Run-Ordner.
  - Modellwurzel:
    `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_FROM_MSE20_TO_gt_future_10ep_lp1/`
  - Neuer Zielordner beim Start:
    `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_FROM_MSE20_TO_gt_future_10ep_lp1/<neuer_run_id>/`
- Hinweis:
  - `<neuer_run_id>` wird erst beim tatsächlichen Start erzeugt und hat das Format `dynamic_YYYYMMDD-HHMMSS_xxxxxx`.
  - In `config.json` des neuen Runs wird zusätzlich gespeichert, dass der Lauf aus `dynamic_20260323-104107_7e2ea6` initialisiert wurde.

### 7. 14 Epochen, `lp_weight = 1.0`, `physics_residual_mode = soft_physics`

- Slurm-Job-ID: `949207`
- Job-Name: `tf_soft14`
- Config: `configs/pitcnn_timefirst_steel_medium_soft_physics_14ep.py`
- Training:
  - Datensatz: `./data/steel_train_medium/`
  - Modellklasse: `PITCNN_dynamic`
  - Epochen: `14`
  - Loss-Schedule: 14 Epochen mit `lp_weight = 1.0`, `mse_weight = 1.0`
  - Physics-Residual: `soft_physics`
  - Residualform: `(output_n - output_{n-1}) / dt - alpha * laplace(input_t0)`
  - Besonderheit: `input_t0` ist die initiale Temperatur `U0`, der Source-Term wird in diesem Modus nicht verwendet.
- Modellname / Zielordner:
  - Modellwurzel: `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_BOTH_soft_physics_14ep_lp1/`
  - Neuer Run-Ordner: `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_BOTH_soft_physics_14ep_lp1/<run_id>/`
- Erwartete Artefakte im Run-Ordner:
  - `config.json`
  - `metrics.csv`
  - `loss_components.csv`
  - `<run_id>.pth`
  - `epoch_*.pth`
  - `losses_plot_<run_id>.png`
  - `proc_time_<run_id>.txt`
- Hinweis:
  - Ein zuvor falsch eingereichter Soft-Physics-Job mit `lp_weight = 0.01` wurde als Slurm-Job `949203` abgebrochen und ersetzt.

### 6. Follow-up ab MSE-Checkpoint: 10 weitere Epochen mit `gt_past`, `lp_weight = 1.0`

- Slurm-Job-ID: `949147`
- Job-Name: `tf_gpast30b`
- Config: `configs/pitcnn_timefirst_steel_medium_from_mse20_to_gt_past_10ep.py`
- Abhängigkeit:
  - Startet nur, wenn Job `949146` erfolgreich beendet wurde (`afterok:949146`).
- Training:
  - Datensatz: `./data/steel_train_medium/`
  - Modellklasse: `PITCNN_dynamic`
  - Initialisierung aus Checkpoint des Runs `dynamic_20260323-104107_7e2ea6`
  - Gesamtschedule: 30 Epochen
  - Davon:
    - Epochen 1-20: `lp_weight = 0.0`, `mse_weight = 1.0`
    - Epochen 21-30: `lp_weight = 1.0`, `mse_weight = 1.0`
  - Physics-Residual: `gt_past`
- Speicherort:
  - Dieser Job schreibt in einen **neuen** Run-Ordner.
  - Modellwurzel:
    `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_FROM_MSE20_TO_gt_past_10ep_lp1/`
  - Neuer Zielordner beim Start:
    `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_FROM_MSE20_TO_gt_past_10ep_lp1/<neuer_run_id>/`
- Hinweis:
  - `<neuer_run_id>` wird erst beim tatsächlichen Start erzeugt und hat das Format `dynamic_YYYYMMDD-HHMMSS_xxxxxx`.
  - In `config.json` des neuen Runs wird zusätzlich gespeichert, dass der Lauf aus `dynamic_20260323-104107_7e2ea6` initialisiert wurde.

## Nachtrag 2026-03-30

Die folgenden Jobs wurden nach der ursprünglichen Statusdatei zusätzlich eingereicht.

### 8. Soft-Physics-Run erneut gestartet nach Fix des Channel-Mismatch

- Slurm-Job-ID: `953818`
- Job-Name: `tf_soft14`
- Aktueller Status beim Eintrag: `RUNNING`
- Config: `configs/pitcnn_timefirst_steel_medium_soft_physics_14ep.py`
- Training:
  - Datensatz: `./data/steel_train_medium/`
  - Modellklasse: `PITCNN_dynamic`
  - Epochen: `14`
  - Loss-Schedule: 14 Epochen mit `lp_weight = 1.0`, `mse_weight = 1.0`
  - Physics-Residual: `soft_physics`
  - Residualform: `(output_n - output_{n-1}) / dt - alpha * laplace(U0)`
  - Besonderheit: In `soft_physics` wird nach dem Fix nur noch Kanal 0 von `input_t0`, also die initiale Temperaturverteilung `U0`, an den Laplace-Term übergeben. Zusätzliche Kanäle wie das Heat-Source-Feld werden dort ignoriert.
- Modellname / Zielordner:
  - Modellwurzel: `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_BOTH_soft_physics_14ep_lp1/`
  - Neuer Run-Ordner: `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_BOTH_soft_physics_14ep_lp1/<run_id>/`

### 9. Physics-only-Run, 14 Epochen, `physics_residual_mode = gt_past`

- Slurm-Job-ID: `954005`
- Job-Name: `tf_gpast14p`
- Aktueller Status beim Eintrag: `RUNNING`
- Config: `configs/pitcnn_timefirst_steel_medium_gt_past_phys_only_14ep.py`
- Training:
  - Datensatz: `./data/steel_train_medium/`
  - Modellklasse: `PITCNN_dynamic`
  - Epochen: `14`
  - Loss-Schedule: 14 Epochen mit `lp_weight = 1.0`, `mse_weight = 0.0`
  - Physics-Residual: `gt_past`
- Modellname / Zielordner:
  - Modellwurzel: `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_PHYS_ONLY_gt_past_14ep_lp1/`
  - Neuer Run-Ordner: `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_PHYS_ONLY_gt_past_14ep_lp1/<run_id>/`
- Erwartete Artefakte im Run-Ordner:
  - `config.json`
  - `metrics.csv`
  - `loss_components.csv`
  - `<run_id>.pth`
  - `epoch_*.pth`
  - `losses_plot_<run_id>.png`
  - `proc_time_<run_id>.txt`

### 10. Physics-only-Run, 14 Epochen, `physics_residual_mode = gt_future`

- Slurm-Job-ID: `954004`
- Job-Name: `tf_gfut14p`
- Aktueller Status beim Eintrag: `RUNNING`
- Config: `configs/pitcnn_timefirst_steel_medium_gt_future_phys_only_14ep.py`
- Training:
  - Datensatz: `./data/steel_train_medium/`
  - Modellklasse: `PITCNN_dynamic`
  - Epochen: `14`
  - Loss-Schedule: 14 Epochen mit `lp_weight = 1.0`, `mse_weight = 0.0`
  - Physics-Residual: `gt_future`
- Modellname / Zielordner:
  - Modellwurzel: `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_PHYS_ONLY_gt_future_14ep_lp1/`
  - Neuer Run-Ordner: `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_PHYS_ONLY_gt_future_14ep_lp1/<run_id>/`
- Erwartete Artefakte im Run-Ordner:
  - `config.json`
  - `metrics.csv`
  - `loss_components.csv`
  - `<run_id>.pth`
  - `epoch_*.pth`
  - `losses_plot_<run_id>.png`
  - `proc_time_<run_id>.txt`

## Zusammenfassung der Speicherorte

Bereits existierender Ordner, der weitertrainiert wird:

- `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_MSE_ONLY_100ep/dynamic_20260323-104107_7e2ea6/`

Neue Modellwurzeln für frisch erzeugte Runs:

- `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_BOTH_gt_future_14ep_lp001/`
- `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_BOTH_gt_past_14ep_lp001/`
- `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_BOTH_model_past_14ep_lp001/`
- `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_FROM_MSE20_TO_gt_future_10ep_lp1/`
- `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_FROM_MSE20_TO_gt_past_10ep_lp1/`
- `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_BOTH_soft_physics_14ep_lp1/`
- `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_PHYS_ONLY_gt_past_14ep_lp1/`
- `runs/dynamic/steel_train_medium/PITCNN_dynamic_STEEL_MEDIUM_PHYS_ONLY_gt_future_14ep_lp1/`

## Konfigurationsdateien

Die zugehörigen Configs liegen hier:

- `configs/pitcnn_timefirst_steel_medium_gt_future_100ep.py`
- `configs/pitcnn_timefirst_steel_medium_gt_past_100ep.py`
- `configs/pitcnn_timefirst_steel_medium_model_past_100ep.py`
- `configs/pitcnn_timefirst_steel_medium_mse_only_100ep.py`
- `configs/pitcnn_timefirst_steel_medium_from_mse20_to_gt_future_10ep.py`
- `configs/pitcnn_timefirst_steel_medium_from_mse20_to_gt_past_10ep.py`
- `configs/pitcnn_timefirst_steel_medium_soft_physics_14ep.py`
- `configs/pitcnn_timefirst_steel_medium_gt_past_phys_only_14ep.py`
- `configs/pitcnn_timefirst_steel_medium_gt_future_phys_only_14ep.py`
