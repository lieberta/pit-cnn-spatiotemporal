# Experimentprotokoll für neue dynamische Trainingsläufe

## Trennung und Skalierung

`prepare_experiment_split` trennt vollständige Experimentordner, niemals einzelne
Zeitpunkte. Der Split-Seed ist unabhängig vom Modell-Seed (Standard jeweils 42).
70 % der geeigneten Experimente gehen ins Training, 15 % in die Validation,
der Rest in den Test; gerundet wird auf ganze Experimente, mit mindestens einem
Experiment je Gruppe. Weniger als drei geeignete Experimente werden abgewiesen.

| Datensatz | Training | Validation | Test |
|---|---:|---:|---:|
| steel_debug_small | 4 | 1 | 1 |
| steel_train_medium | 22 | 4 | 6 |

Die versionierten Listen liegen unter `splits/`. Standardmäßig wird dieselbe
Liste als `experiment_split.json` im Dataset angelegt und anschließend wiederverwendet.
Alternativ setzt `split_manifest_path` explizit den Pfad zur versionierten Liste.
Jeder Run erhält eine Kopie. Änderungen an Experimentauswahl, Metadaten, Split-Seed
oder Quellzahlpolitik werden bei Wiederverwendung abgewiesen. Der SHA-256 erfasst
Metadaten, nicht sämtliche Temperaturarrays.

Min/Max stammen ausschließlich aus den Metadaten der Trainingssimulationen.
Für steel_train_medium: 19.7373091693751 bis 119.81875303121228 °C.
Validation und Test verwenden genau diese Skalierung, ohne Clipping; Werte
außerhalb [0,1] sind zulässig. Temperatur, Anfangszustand, Randtemperatur und
Quellterm verwenden dieselbe Skala R = Tmax − Tmin.

`training_source_counts = [3,4,5,6,7]` reserviert andere Quellzahlen vollständig
für den Test. Diese Einstellung erfordert einen passenden Datensatz: die
vorliegenden Stahlversuche haben 18–24 Quellen. Für das Paper müssen unabhängige
Simulationen mit 1–2 und 8–10 Quellen ergänzt und getrennt ausgewertet werden.
Die Experiment-ID ist der Ordnername; umbenannte Kopien sind keine unabhängigen Fälle.

## Zeit, Anfangsbedingung und Residual

Das Dataset gibt den ganzzahligen `step_idx` zurück. Der Modelleingang ist
τ = step_idx·Δt/H, der vorherige Eingang τpast = (step_idx−1)·Δt/H.
Die Ableitung wird weiterhin durch Δt in Sekunden geteilt.
Für step_idx=1 wird die vergangene Vorhersage durch den Temperaturkanal X0
ersetzt. Zusätzlich wird die unveränderte Modellausgabe bei τ=0 kontrolliert;
dies ist keine harte Erzwingung der Anfangsbedingung und kein zusätzlicher IC-Loss.

Der gemeinsame explizite Euler-Residual lautet auf inneren Gitterpunkten:

    r = (u_now − u_past)/Δt − α·Laplacian(u_past) − q/(ρ·cp·R)

Er wird für jede Variante einschließlich MSE-only berechnet. Für die gemeinsame
Diagnostik sind beide Zustände Modellvorhersagen, abgesehen vom bekannten ersten
Anfangszustand. Die gewählte Trainingsvariante (model_past, gt_past, gt_future,
soft_physics) bestimmt weiterhin den optimierten Loss.

`train_physics_loss` und `val_physics_loss` bezeichnen den gemeinsamen
ungewichteten Residual-MSE, auch für MSE-only.
`train_objective_physics_loss` und `val_objective_physics_loss` enthalten den tatsächlich verwendeten Physikterm;
bei deaktiviertem Physiktraining ist dieser leer. Deshalb lässt sich insbesondere
bei Target-Varianten `val_total_loss` nicht aus `val_physics_loss` rekonstruieren.
Der Daten-MSE wird auch bei MSE-Gewicht null gemessen. Batchmittelwerte werden
nach tatsächlicher Samplezahl gewichtet. IC-Diagnostik steht zusätzlich in
`initial_condition_metrics.csv`.

## Reproduzierbarkeit und Fortsetzungskontrolle

Die Konfiguration `configs/reproducibility.py` definiert Modell- und Split-Seed.
Vor jeder Modellinitialisierung werden Python, NumPy und Torch gesetzt;
deterministische Torch-Algorithmen sind aktiviert. Der Train-Sampler erhält pro
Epoche seed+epoch, Validation und Test sind nicht gemischt. Identische Seeds,
Architektur und Datenkonfiguration liefern dieselbe Initialisierung und
Batchreihenfolge. Hardware-/Versionswechsel sind damit nicht automatisch bitgleich.

Jeder Run speichert `initial_state.pth`, dessen Hash, Git-Commit/Dirty-Status und
Diff-Hash, Metadatenhash, Normierung, Einheiten, Split und effektiven Lossplan.
Checkpoints enthalten Optimizer, Python-/NumPy-/Torch-Zufallszustände, Seed und Protokoll.
`best.pth` wird ausschließlich nach dem kleinsten Validation-MSE ausgewählt.
Die Testmenge wird im Training nicht ausgewertet.

Alte Checkpoints ohne dieses Protokoll werden beim Fortsetzen abgewiesen:
Deren Training hat möglicherweise die neuen Testexperimente gesehen, und die
Normierung kann abweichen. Ein unverbrauchter Test lässt sich nicht nachträglich
herstellen. Historische Berichte dokumentieren weiterhin den damaligen Stand.

1. Neue Baseline mit `configs/pitcnn_timefirst_steel_medium_mse20_protocol.py`
   für 20 Epochen trainieren.
2. Ihre Run-ID als `MSE20_PROTOCOL_RUN_ID` setzen und mit
   `configs/pitcnn_timefirst_steel_medium_from_mse20_to_mse_10ep.py` in einem neuen
   Run auf 30 Epochen verlängern.
3. Für die Physics-Fortsetzungen ebenfalls diese neue Baseline als
   `initialize_run_ids_dynamic` verwenden. Die gt_past-/gt_future-Fortsetzungen
   lesen ebenfalls `MSE20_PROTOCOL_RUN_ID`; historische IDs sind dort entfernt.
   Gleiche Seeds und einen 20+10-Epochenplan wählen.
   Verzweigt wird vom letzten 20-Epochen-Checkpoint, nicht von einem früheren Best.
4. Für mehrere Seeds jeweils eine eigene Baseline und gepaarte Fortsetzungen
   erzeugen; die Splitlisten bleiben unverändert.

## Unabhängige Auswertung

    python -m evaluation.evaluate_physics \
      --runs runs/dynamic/MODELL/RUN_A runs/dynamic/MODELL/RUN_B \
      --output-dir reports/test_evaluation --device cpu

Das Tool verwendet `best.pth` und die gespeicherten Testexperimente. Es prüft
Protokoll, Split und Metadaten. `--extrapolation-data-root PFAD` verwendet einen
separaten Datensatz mit der Trainingsnormierung und dem Trainingshorizont H;
`--source-counts 1 2` beziehungsweise `8 9 10` filtert die Quellzahl. Material,
Gitterabstände und Randtemperatur müssen übereinstimmen.

Pro Experiment: MAE/RMSE in °C, PDE-RMSE in K/s, IC-RMSE/Maximalfehler und
Rand-RMSE/Maximalfehler in °C. Der PDE-Residual wird mit R rückskaliert und nur
im Inneren gemessen; die Randdiagnostik betrachtet alle äußeren Gitterflächen
ohne doppelt gezählte Ecken. Zeitpunkte und Quellzahlen erhalten eigene Tabellen.

`experiments.csv` und `times.csv` enthalten Einzelwerte; `seed_summary.csv`,
`source_count_summary.csv` und `time_summary.csv` enthalten Gruppenwerte.
`experiment_across_seeds.csv` mittelt zuerst Seeds desselben Experiments;
`variant_summary.csv` mittelt anschließend diese unabhängigen Experimente.
`across_seed_summary.csv` enthält Mittelwert und Streuung der pro Seed über
Testexperimente gemittelten Fehler. Doppelte Variante/Seed-Kombinationen werden
abgewiesen; gepaarte Runs benötigen denselben Hash der Anfangsgewichte.
Die Streuungen sind deskriptive Standardabweichungen, keine Konfidenzintervalle.
Ein Scatterplot stellt Temperatur-RMSE dem PDE-RMSE gegenüber. `provenance.json`
zeichnet Checkpoint und verwendete Fälle auf.

## Prüfung

    python -m unittest discover -s tests -v

Die Tests prüfen Zeit-/Einheitenkonsistenz, Temperaturkanal und ersten Schritt,
alle Loss-Modi, Split-Disjunktheit/Wiederverwendung, Train-only-Normierung,
Initialisierungsseed sowie physikalische Auswertung unabhängig von Batchgrößen.
Ein zusätzlicher Test verwendet echte gespeicherte Float64-Stahlsimulationen:
Der konsistente Euler-Residual muss numerisch null sein; falsches Δt und ein
fehlender Quellterm werden erkannt. Ohne lokale Simulationsdaten wird nur dieser
Datencheck übersprungen. Die vorliegenden Änderungen starten keine neuen Trainings.

## Kontrollierter Vergleich A–E

    python scripts/physics_comparison.py

Dies schreibt ausschließlich einen Plan für Seeds 42, 43 und 44. Mit
`--execute` werden zunächst der echte Stahl-Residualtest und die Kalibrierungen,
danach die 15 Trainings nacheinander ausgeführt. Auf einem GPU-Rechenknoten
ausführen; die gesamte Kampagne ist kein kurzer Funktionstest.

Alle Varianten verwenden `configs/pitcnn_steel_comparison.py`, dieselben
versionierten Splits, Modellklasse PITCNN_dynamic, 16 Kanäle, Batchgröße 128,
Lernrate 0.001 und 30 Epochen ab identischen Anfangsgewichten je Seed:

| Variante | Regularisierung | Gewicht |
|---|---|---:|
| A | MSE-only | 0 |
| B | vollständiges model_past | 0.001 |
| C | vollständiges model_past | 0.01 |
| D | vollständiges model_past | 0.1 |
| E | soft_physics / diffusion-inspired | aus Trainingsgradienten |

`evaluation.calibrate_physics` misst auf dem ersten geshuffelten Trainingsbatch
(Standard 4 Samples) die ungewichteten Losses und Gradienten-L2-Normen beider
Regularisierungen. Es führt keine Optimizer-Schritte aus und liest keine
Validation-/Testtargets. Pro Seed wird als Kandidat für E das Gewicht gewählt,
das an der Initialisierung ein Verhältnis von 0.1 zwischen gewichtetem
Physikgradienten und Datengradienten ergibt. Der Median dieser drei Kandidaten
wird für alle Seeds verwendet; Wahl und Einzelmessungen werden gespeichert.
Diese lokale Skalenkalibrierung ist keine Hyperparameteroptimierung und keine
Aussage über spätere Trainingsgradienten.

Der Trainer speichert zusätzlich `loss_gradient_scales.csv`: erster
Trainingsbatch jeder Epoche, Datenloss, tatsächlich gewählter Physikterm,
Gradientennormen, Verhältnisse und Gewichte. Das Messen überschreibt keine
Parametergradienten. `measure_physics=True` aktiviert in
`CombinedLoss_dynamic.compute_components` die Messung bei Gewicht null.
Der gemeinsame Modell-PDE-Residual bleibt für alle Varianten identisch.

`PITCNN_SEED` setzt den Modellseed; der Splitseed bleibt 42. Der Run speichert
neben der effektiven JSON-Konfiguration eine Kopie der aufgerufenen Python-
Konfiguration und deren SHA-256. Protokollversion 2 verhindert eine Fortsetzung
mit inkompatiblem CSV-/Diagnostikstand aus Version 1.

`across_seed_summary.csv` und die Initialisierungskontrolle sind für den
Vergleich frischer Läufe gedacht; Warmstarts mit unterschiedlichen Ausgangs-
checkpoints müssen separat ausgewertet werden.

gt_past bleibt Implementierungskontrolle, gt_future eine zusätzliche
supervidierte Kontrolle; beide gehören nicht zum A–E-Vergleich. Historische
Paper-Reproduktion erfordert separat eingefrorene Gewichte, Daten und Code;
sie wird durch die neue Kampagne nicht ersetzt.

## Slurm-Start

`slurm/physics_comparison.slurm OUTPUT calibration` führt den Residualcheck und
alle drei Kalibrierungen auf einer GPU aus. Danach lässt sich dieselbe Datei
mit `OUTPUT training`, `--array=0-14%3` und `--dependency=afterok:KALIBRIERUNGSJOB`
starten. Die Array-Indizes bilden zuerst die Varianten A–E für Seed 42, dann
43 und 44 ab. Jeder Lauf bekommt seine eigene Zeitgrenze von 23:59 Stunden.
Die Python-Umgebung bringt ihre CUDA-Laufzeit mit; ein separates altes CUDA-
Modul wird nicht geladen.

PyTorch 2.0.1 unterstützt natives Reflection-Padding auf CUDA im Rückwärtslauf
nicht deterministisch. `ReflectionConv3d` behält Parameter, Zustandsnamen und
Vorwärtswerte unverändert, summiert im Rückwärtslauf jedoch die reflektierten
Randbeiträge durch geordnete Slice-Additionen. Der Test vergleicht Werte und
Gradienten mit dem nativen CPU-Padding, einschließlich asymmetrischer Ränder.
Nach dieser Startkorrektur bestanden alle 13 lokalen Tests.

Die GPU-Kalibrierung zeigte danach denselben PyTorch-2.0.1-Fehler auch bei
`avg_pool3d_backward_cuda`. Für das ausschließlich verwendete nichtüberlappende
2×2×2-Pooling bleibt der native Vorwärtslauf erhalten; der Rückwärtslauf verteilt
jeden Gradienten deterministisch mit Faktor 1/8 auf die acht Eingabezellen.
Der Vergleich mit nativen CPU-Werten und -Gradienten besteht. Zusammen bestehen
alle 14 lokalen Tests; die GPU-Kalibrierung prüft zusätzlich den vollständigen
Modellgradienten vor dem Training.

Auf Rechenknoten ohne Git muss `PITCNN_GIT_PROVENANCE_FILE` auf einen vor der
Einreichung erzeugten Snapshot zeigen. Dieser enthält Repositorypfad, Commit,
Dirty-Status und Diff-Hash vom Einreichungsrechner. Die Herkunftsmethode wird
als `submission_host_snapshot` markiert; ohne Snapshot bricht die Erfassung ab.
Während der Kampagne den Quellstand nicht verändern. Der Slurm-Starter erwartet
die Datei `source_provenance.json` im Ausgabeordner der Kampagne.
