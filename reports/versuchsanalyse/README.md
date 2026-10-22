# Detaillierte Versuchsanalyse

- `Detaillierte_Versuchsanalyse.pdf`: vollständiger Bericht mit mathematischen Ableitungen, historischen Vergleichstabellen und Register aller 93 Run-Ordner.
- `Detaillierte_Versuchsanalyse.md`: editierbare Textfassung.
- `run_inventar.csv`: filterbare Run-Übersicht; fehlende/deaktivierte Messgrößen bleiben leer.
- `run_details.json`: Konfigurationen, zusammengeführte Epochen, Rohmetriken und tatsächliche Gewichtsphasen.
- `ground_truth_residual_checks.json`: 24 Stichproben der Euler-Konsistenz auf Stahl-Daten; außerdem Soft-Physics und falscher Zeitabstand.
- `old_dataset_boundary_checks.json`: Fehleruntergrenzen durch feste Anfangsränder für 29 vorhandene alte Experimente.

## Reproduktion

Alle Befehle im Repository-Hauptverzeichnis ausführen. Inventar und Textgenerator benötigen nur die Python-Standardbibliothek:

```bash
python reports/versuchsanalyse/inventar.py
python reports/versuchsanalyse/erstelle_analyse.py
python reports/versuchsanalyse/render_pdf.py
```

Der PDF-Renderer benötigt ReportLab und pypdf sowie die DejaVu-Schriftarten. Verwendet wurden ReportLab 3.6.12 und pypdf 3.17.4 unter Python 3.9; temporär installierte Pakete werden zusätzlich unter `/tmp/repo-summary-pdf` gesucht. Normale installierte Pakete funktionieren ebenfalls. Die Prüfscripte für Simulationsfelder benötigen NumPy und Zarr 2:

```bash
/beegfs/home/l/lieberta/venv/penn-venv/bin/python reports/versuchsanalyse/pruefe_ground_truth.py
/beegfs/home/l/lieberta/venv/penn-venv/bin/python reports/versuchsanalyse/pruefe_alte_randwerte.py
```

Es werden nur Berichtsartefakte geschrieben. Training, Modelle, Daten und bestehende Änderungen an Trainingsdateien bleiben unverändert. Der Text enthält manuell geprüfte historische Interpretationen; eine spätere erneute Ausführung auf veränderten Daten ersetzt keine erneute historische Prüfung.

## Zählung und Provenienz

93 Konfigurationen, 83 Ordner mit Metriken, 10 ohne vollständige protokollierte Epoche. 2.000 rohe Metrikzeilen ergeben 1.942 unterschiedliche Run/Epochen-Paare. Zwei frühe Run-Ordner enthalten mehrere Seeds pro Epochenindex; zur kompakten Anzeige wird die letzte Zeile je Epochenindex verwendet, die Rohzeilen bleiben vollständig erhalten. Für diese beiden Ordner wird im CSV kein globales Best-MSE angegeben. Die PDF führt die verschiedenen Endpunkte separat auf.

Die Runs speichern keine Git-SHAs. Die Zuordnung zu Formel-Familien stützt sich auf historische Code-Snapshots, Run-Kommentare, Logs und Datenmetadaten; die PDF unterscheidet bestätigte Beobachtung und unsichere historische Zuordnung.
