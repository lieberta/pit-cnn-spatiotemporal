# Prüfstand zur unterbrochenen Überarbeitung (22.10.2026)

Dieser Stand dokumentiert die vorhandene Implementierung; die geplante
wissenschaftliche Evaluation ist noch nicht abgeschlossen.

1. **Historie:** Git-Rückblick und detaillierte Analyse von 93 Run-Konfigurationen
   liegen unter `reports/`. Historische Runs enthalten keine Git-SHAs; ihre
   Zuordnung ist daher teilweise unsicher. Ein vollständiges, separat geprüftes
   Archiv von Paper-Code, Gewichten und Simulationsarrays ist nicht nachgewiesen.
2. **Zeit:** Benachbarte Modelleingaben verwenden ganzzahlige Simulationsschritte
   und die normierte Schrittweite Δt/H; die Ableitung verwendet Sekunden.
3. **Anfangszustand und PDE:** Beim ersten Residualschritt wird X0 verwendet.
   Der rohe Modellausgang bei t=0 wird diagnostiziert, aber X̂(0)=X0 wird noch
   nicht hart erzwungen. Der gemeinsame Euler-Residual berücksichtigt Quelle
   und innere Gitterzellen. Der Test auf gespeicherten Float64-Stahldaten besteht.
4. **Datentrennung:** Experimentweise, persistierte Train/Validation/Test-Splits
   und ausschließlich aus Trainingsdaten bestimmte Temperaturnormierung sind
   implementiert. Splitlisten für beide Stahldatensätze liegen unter `splits/`.
5. **Unabhängige Messgrößen:** Datenfehler und gemeinsamer Modell-PDE-Residual
   werden unabhängig von den Trainingsgewichten protokolliert; der tatsächlich
   optimierte Physikterm bleibt gesondert ausgewiesen.
6. **Reproduzierbarkeit:** Seeds, Initialgewichte, Run-Provenienz und Auswahl
   von best.pth nach Validation-MSE sind implementiert. Neue Baseline- und
   MSE-Fortsetzungskonfigurationen liegen vor. Historische Physics-Fortsetzungen
   müssen noch auf eine neu trainierte Protokoll-Baseline umgestellt werden.
7. **Evaluation:** Das Werkzeug für Testdaten, Temperaturfehler in °C,
   PDE-Fehler in K/s sowie Zeit-, Quellzahl- und Seed-Auswertungen ist vorhanden.
   Neue kontrollierte Trainings und Mehrseed-Vergleiche wurden im Rahmen dieser
   Prüfung nicht durchgeführt. Zusätzliche Simulationen für die vorgesehenen
   Quellzahlbereiche fehlen; vorhandene Stahldaten enthalten 18–24 Quellen.

## Erneute Prüfung

Mit `/beegfs/home/l/lieberta/venv/penn-venv/bin/python -m unittest discover -s tests -v`
bestanden alle 10 Tests einschließlich des Tests auf gespeicherten Simulationen.
`git diff --check` meldete keine Fehler. Die Tests enthalten kurze synthetische
Trainerdurchläufe; daraus folgt keine abgeschlossene experimentelle Evaluation.

Die drei neuen Protokoll-Konfigurationen werden ausdrücklich versioniert,
obwohl die bestehende `.gitignore` neue Dateien unter `configs/` ausschließt.
