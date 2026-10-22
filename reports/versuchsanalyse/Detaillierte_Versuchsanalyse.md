# Mathematische Analyse deines Versuchsverlaufs
## Loss-Ableitungen, Datengenerierung und Ergebnisse im Zusammenhang
**Untersuchter Stand:** Git bis `4867499` (27.03.2026), vorhandene Run-Artefakte und Logs bis 31.03.2026. Auswertung erstellt am 22.10.2026. 93 Run-Konfigurationen, davon 83 mit Epochenmetriken und 10 ohne protokollierte Epoche; insgesamt 2.000 Metrikzeilen mit 1.942 eindeutigen Kombinationen aus Run-ID und Epochenindex. 58 zusätzliche Zeilen stammen aus mehrfach belegten Epochen zweier früher Run-Ordner. 74 dynamische CNN-Runs, 18 statische CNN-Runs und ein PINN-Run. Frühere Resultate von 2024 werden aus Git ergänzt.

**Kernaussage:** Du hast über die Zeit nicht nur die Genauigkeit einer Ableitung verbessert. Du hast mehrfach die mathematische Nebenbedingung selbst verändert: von einer Sekante über den gesamten Horizont über eine Kopplung zweier Modellzustände bis zu einem mit Ground Truth verankerten Ein-Schritt-Fehler. Gleichzeitig änderten sich Quelle, Randbedingungen, Temperatur- und Zeitskalierung sowie das physikalische Problem. Die Loss-Werte sind deshalb nur innerhalb klar abgegrenzter Versuchsgruppen sinnvoll vergleichbar.

Drei besonders belastbare Befunde: (1) Das alte Datenset erzeugt zusammen mit dem späteren Kopieren der Anfangsränder eine unvermeidbare MSE-Untergrenze. (2) `gt_past` entspricht auf den aktuellen Euler-Daten praktisch einem skalierten MSE; das lässt sich direkt in deinen Logs nachweisen. (3) Der letzte Git-Stand enthält einen Einheitenfehler bei der zweiten Vorhersage nach Einführung normierter Modellzeit. Dieser begrenzt insbesondere die Interpretation von `model_past` und `soft_physics`.

## 1. Wie die Zuordnung vorgenommen wurde
**Messwerte:** `runs/**/metrics.csv`, `loss_components.csv` und `config.json`. Die Tabellen verwenden tatsächlich protokollierte Epochen und Gewichte. Ein Name wie „100ep“ bedeutet nicht, dass 100 Epochen gelaufen sind. „—“ bedeutet nicht verfügbar oder im Training nicht berechnet; deaktivierte Komponenten werden nicht als fehlerfreie Messung interpretiert.

**Mathematik:** Historische Git-Blobs von Loss, Trainer, Dataloader und Simulator, ergänzt durch Run-Kommentare. Kein Run speichert einen Git-SHA oder einen unveränderlichen Snapshot von Code und Daten. `dataset_version` ist meist „unknown“. Daher unterscheidet der Bericht: **belegt im Code**, **belegt im Run**, **plausible zeitliche Zuordnung**. Ein Commit nach Run-Start kann bereits zuvor uncommittete Änderungen enthalten. V0.2/V0.3 vom 26.02. sind ein konkretes Beispiel: Ihre Änderungen wurden erst am 27.02. gemeinsam committed.

**Vergleichsregeln:** Standardmäßig wird die letzte vollständige protokollierte Epoche gezeigt; ausdrücklich bezeichnete Tabellen vergleichen gemeinsame Epochen. „Min. MSE“ ist das Minimum der protokollierten Kurve, nicht automatisch ein gespeicherter Best-Checkpoint. Best-MSE und Best-Physik können aus verschiedenen Epochen stammen. Bei den zwei gemischten frühen Run-Ordnern werden im Register die jeweils letzten CSV-Einträge je Epochenindex angezeigt; alle Rohzeilen und Seeds bleiben in run_details.json erhalten, und ein globales Best-MSE wird im CSV-Inventar dafür nicht ausgewiesen. Die Größen sind normierte Validierungsverluste, keine Temperaturfehler in Kelvin und kein unabhängiger Generalisierungstest.

**Zusätzliche Prüfung:** Die Formeln wurden read-only an gespeicherten Temperaturfeldern geprüft. Es wurde kein Training gestartet und kein Modell oder Trainingscode verändert. Die bestehende lokale Änderung in `training/loss.py` wurde lediglich gelesen.

## 2. Gemeinsame Notation und physikalische Referenz
Es bezeichnen U die physikalische Temperatur, u=(U−Umin)/R die normierte Temperatur mit R=Umax−Umin, yₙ den normierten Simulationszustand und vₙ=fθ(u₀,s,tₙ) die Modellvorhersage. P bezeichnet das Einschränken auf innere Gitterzellen. Die Wärmequelle s hat nach Normalisierung die Einheit „normierte Temperatur pro Sekunde“.

```math
∂U/∂t = α ΔU + S,         α = k/(ρ cp),     S = Q/(ρ cp)
∂u/∂t = α Δu + s,         s = Q/(ρ cp R)
Lh u = (u[i+1]−2u[i]+u[i−1])/dx² + analoge Terme in y,z
A = I + dt α Lh
P yₙ = P(yₙ₋₁ + dt α Lh yₙ₋₁ + dt s)
LD = mean_all((vₙ−yₙ)²),   LP = mean_inner(r²)
L = λD LD + λP LP
```

Die ältere Simulation setzt S=100000 direkt als Temperaturanstiegsrate ein; sie teilt diesen Wert nicht durch ρ·cp. Die spätere Simulation interpretiert Q dagegen explizit als volumetrische Heizleistung. Eine bloß numerische Gleichheit von Quellparametern würde deshalb nicht dieselbe Physik bedeuten.

**Drei verschiedene Schrittweiten:** dt_sim ist der interne Simulationsschritt; dt_store der Abstand gespeicherter Frames; h_loss der Abstand der beiden für den Loss verwendeten Zustände. Dazu kommt ab Ende März die dimensionslose Modellzeit τ=t/H. Diese vier Größen müssen auseinandergehalten werden.

### 2.1 Mathematischer Fahrplan: Zuordnung der Loss-Familien

| Familie / Zeitpunkt | Zeitdifferenz | Laplace / Quelle | Beleg und Grenze |
| --- | --- | --- | --- |
| Früh dynamisch bis 09.02. | (v(t)−u₀)/t | L_unscaled(u₀); rohe Schwelle >1000 | Git: 7f9d778, 6a1f0ac; kein Run-Snapshot |
| Normierung ab 13.02. | Weiterhin (v(t)−u₀)/t | L_unscaled(u₀); s=S/R | fcaa462; normsource-Kommentare |
| Zwei Vorhersagen ab 20.02. | [v(t)−v(t−0,1)]/0,1 | L_unscaled(u₀); Quelle teils normiert, teils zurückgenommen | cd47e19, 13cf5d8; lokale Zwischenstände unsicher |
| L(output), 23.–25.02. | Übergang 0,1 → 0,001 s | L_unscaled(vₙ); normierte Quelle im Commit | 2237a9c; mehrere Änderungen gleichzeitig |
| V0.2, 26.02. | Zwei Ausgaben, h≈0,001 s | Lh(vₙ); nur Innenraum | Run-Kommentare; kein isolierter V0.2-Commit |
| V0.3/V0.4 ab 26.02. | Zwei Ausgaben, h=0,001 s | Lh(vₙ₋₁); Quelle aus u₀ | 1ee33d4; alte Daten oft weiterbenutzt |
| 05.–09.03. | h noch 0,001; Daten teils dt=0,01 | Weiter Lh(vₙ₋₁) | 5408f5d; Übergang von Loader und Trainer |
| Ab 09.03. | h aus dataset.dt | Weiter Lh(vₙ₋₁) | fe8e545; Starts am 10.03. ohne Messwerte |
| Ab 13.03., model_past | Zwei Ausgaben mit dt=0,1 | Lh(vₙ₋₁), explizites s | 59472c7; Anfangsverankerung entfernt |
| Ab 13.03., GT-Modi | vₙ−yₙ₋₁ oder yₙ₊₁−vₙ, geteilt durch dt | Lh(yₙ₋₁) bzw. Lh(vₙ), explizites s | 59472c7; Modi in Run-CSV belegt |
| Ab 27.03., Soft-Physics | Zwei Ausgaben | Lh(u₀), kein s | 0790762; erfolgreicher Run erst 30.03. |
| Letzter Git-Stand | τ=t/H, aber vergangene Abfrage τ−dt | Zweiter Netzaufruf zeitlich inkonsistent | 4867499 + unveränderter Trainer; historische Betroffenheit nicht vollständig beweisbar |

Diese Zuordnung benennt die rekonstruierbare Formel-Familie. Wo nur Datum oder Name vorliegen, wird daraus kein exakt belegter Runtime-Snapshot gemacht. Der spätere Text trennt innerhalb jeder Familie die tatsächlich gemessenen Vergleichsläufe.

## 3. Datengenerierung: Was sich physikalisch geändert hat

| Stand / Daten | Raumgitter / Geometrie | Zeit | Quelle / Material |
| --- | --- | --- | --- |
| 2024, alte Raumdaten | 64×32×16; 6,3×3,1×1,5; dx=dy=dz=0,1 | dt_sim=0,001; Speicherung jedes 100. Frames → 0,1; H=10 s | α=0,0257; S=100000; T₀-Quellen 1100, Umgebung 20 |
| 23.02., new_detailed_heat_sim_f64 | Gleiche Geometrie und Gitterzahl | Alle 10001 Frames; dt_store=0,001; H=10 s | Gleiche alte Quelle; float64-Simulation/-Speicherung |
| 26./27.02., überarbeiteter Generator | Quellen weg von x/y-Rändern; z=1:3 statt :2 | Parametrisierbares T und DT | Änderung betrifft neu erzeugte Daten; alte Dateien bleiben unverändert |
| 05.–09.03., TRAIN_DATASET_10s_dt0.01 | Alte Raumgeometrie; laut Generator | dt=0,01; 1001 Frames; 10 s | Info: R=57343,328; 400 Experimente in späteren Run-Configs |
| 13.03., steel_debug_small | 64×32×16; 0,5×0,5×0,15 | dt=0,1; H=120 s; heute 1202 Frames | k=45, ρ=7850, cp=470; Q=3e7 |
| 23.03., steel_train_medium | Gleiche Stahl-Physik; dx≈0,00794; dy≈0,01613; dz=0,01 | dt=0,1; 1200 Targets/Experiment | 32 Experimente; 18/20/22/24 Quellen; R=101,25644 |

Die „high-res“-Umstellung im Februar ist in den untersuchten Generatoren vor allem eine **zeitliche** Verfeinerung der gespeicherten Daten um Faktor 100, nicht eine Erhöhung von Nx, Ny, Nz. Im März bleibt die Gitterzahl ebenfalls gleich, aber die physikalische Domäne schrumpft. Damit ändern sich die räumlichen Ableitungskoeffizienten drastisch und anisotrop.

Für Stahl ist α=1,21967746e-5, S=Q/(ρcp)≈8,1312 und s≈0,08030 im Medium-Datensatz. Anfangstemperatur und Quelle sind getrennte Felder; der Anfangszustand enthält geglättete kleine Störungen und 50 statt 1100 Grad in den Quellzellen. Zwei Eingangskanäle liefern normiertes T₀ und den normierten Quellterm. Die Quellmaske bleibt zusätzlich für den Loss verfügbar. Der Trainer setzt `source_region_weight=1.0`; ein tatsächlich trainierter Sweep dieser Gewichtung ist nicht belegt.

Der Beton-Zweig wurde bereits im Februar eingeführt: k=1,7, ρ=2300, cp=880 ergibt α≈8,40e-7. Die dokumentierten März-CNN-Vergleiche laufen jedoch auf Stahl. Eine aktivierte realistischere Materialbeschreibung ist nicht mit einer Verbesserung desselben alten Benchmarks gleichzusetzen.

Die aktuelle Speicherung enthält einen zusätzlichen Zustand bei H+dt, damit `gt_future` auch für t=H definiert ist. Alle Modi verwenden n=1,…,1200. Die älteren committeden Dataloader beschränkten Targets auf Zustände mit vorhandenem Vor- und Nachfolger. Der genaue historische Zeitpunkt der Datenneuerzeugung ist nicht durch unveränderliche Datensatzversionen abgesichert; die heute vorhandenen Metadaten belegen den aktuellen Zustand.

### 3.1 Nachgewiesener Konflikt zwischen altem Datensatz und neuer Randbehandlung
Im alten Generator liegen Heizquellen in z=:2, also auch auf dem Bodenrand z=0. T₀ ist dort heiß; nach jedem Simulationsschritt werden die Außenflächen auf 20 gesetzt. Ab V0.2 kopiert das dynamische Modell dagegen wieder die vollständigen Anfangsränder in die Ausgabe. Auf solchen alten Daten können Vorhersage und Target an diesen Stellen niemals übereinstimmen.

```math
MSE_min ≥ sum_boundary((u₀ − yₙ)²) / N_all
```

An allen **29 heute vorhandenen** alten Zarr-Experimenten vom 23.02. ergibt diese Rand-Untergrenze im Mittel **7,019e-6**, je Experiment 4,424e-6 bis 9,966e-6. Sie wurde aus T₀ und dem ersten Folgezustand berechnet; der Simulator hält die Folgeränder konstant. Die Quelle wurde in diesen vorhandenen Daten nicht nachträglich nach innen versetzt.

Das erklärt einen großen Teil des Plateaus von ungefähr 7e-6 bei späterem MSE-only-Training auf diesem Datensatz: Run `4054e1` endet bei 7,273e-6. Das ist ein struktureller Fehlerboden, kein überzeugender Beleg für fehlende Netzkapazität. Der Mittelwert über 29 Experimente ist kein exakt rekonstruiertes historisches Validierungsmittel; einige frühere Runs nennen noch 30 Experimente, und die Splits sind nicht gespeichert. Quelle: `old_dataset_boundary_checks.json` und Modell-Randzuweisungen.

## 4. 2024 bis Anfang Februar 2026: Sekante über den gesamten Horizont

Der frühe dynamische Loss verwendet nur eine Vorhersage. Die zeitliche „Ableitung“ ist die mittlere Änderung vom Anfang bis zum Zielzeitpunkt, während der Laplace-Term am Anfangszustand festgehalten wird. Der räumliche Kernel enthält 1 an den sechs Nachbarn und −6 in der Mitte, aber keinen Faktor 1/dx².

```math
r_old(t) = (v(t) − u₀)/t − α L_unscaled(u₀) − s(u₀)
L_unscaled = sum_six_neighbors(u_neighbor) − 6u_center
```

Damit verlangt LP≈0 näherungsweise v(t)=u₀+t[αL_unscaled(u₀)+s(u₀)]: eine lineare Fortschreibung mit eingefrorener Anfangsdiffusion. Das ist keine lokale Ableitung an t. Je länger der Horizont, desto weniger entspricht diese Nebenbedingung der iterierten Simulation. Bei dx=0,1 müsste der Laplace-Term 100-mal so groß sein.

Hinzu kommt die Quellerkennung `input > 1000` auf normalisierten Daten. Für Werte im üblichen Bereich [0,1] bleibt die Maske leer: Die frühe „Physik“ kann effektiv eine Gleichung ohne Wärmequelle erzwingen. Die Quelle ist dann nicht nur falsch skaliert, sondern verschwindet vollständig.

Der Autograd-Entwurf von 2024 wurde auskommentiert und im Februar 2026 entfernt. Der damalige Versuch differenziert nach einem nachträglich expandierten Zeit-Tensor und verwendet `grad_outputs=ones`. Das ist kein belegter erfolgreicher voxelweiser Zeitableitungsweg. Der Codekommentar beschreibt Probleme, aber ein grundsätzliches mathematisches Verbot, automatische Zeitableitung mit räumlichen Differenzen zu kombinieren, folgt daraus nicht.

Die archivierten Auswertungen von August 2024 zeigen bei zehn Fällen mittlere absolute Temperaturabweichungen: frühes dynamisches Modell 47,33→133,53 von 1→10 s, „latenttimeforrealthistime“ 26,62→13,82, „smalltest“ 29,37→28,33. Unterschiedliche Trainingsbedingungen und keine vollständigen Run-Metadaten erlauben daraus keinen isolierten Architekturvergleich. Quelle: `310ebe5:plots/*/deviations.txt` und `evaluate.py`.

| Run / Variante | Ep. | λP / λD | Val.-MSE | Val.-Physik | Val.-Gesamt |
| --- | --- | --- | --- | --- | --- |
| d056d1 · latent + Physik | 10 | 1.0/1.0 | — | — | 5.646e-05 |
| ca1b44 · latent, nur Daten | 10 | 0.0/1.0 | 7.803e-07 | — | 7.803e-07 |
| 27e545 · timefirst + Physik | 25 | 1.0/1.0 | — | — | 6.052e-05 |
| 62167a · timefirst, nur Daten | 24 | 0.0/1.0 | 7.362e-06 | — | 7.362e-06 |

| Run-Ordner | Seed der CSV-Zeile | Epoche | Val.-Gesamt | Bedeutung |
| --- | --- | --- | --- | --- |
| 27e545 | 60553 | 25 | 1,691e-4 | MSE + Physik |
| 27e545 | 16912 | 25 | 5,092e-4 | MSE + Physik |
| 27e545 | 94993 | 25 | 6,052e-5 | MSE + Physik |
| 62167a | 60553 | 24 | 1,353e-6 | nur MSE |
| 62167a | 16912 | 24 | 5,113e-6 | nur MSE |
| 62167a | 94993 | 24 | 7,362e-6 | nur MSE |

Die frühen Gesamt-Loss-Werte sind nicht mit einem MSE gleichzusetzen; Komponenten fehlen. Die timefirst-Ordner enthalten 55 bzw. 52 CSV-Zeilen, aber nur 25 bzw. 24 unterschiedliche Epochenindizes. Ab Epoche 11 sind jeweils drei Seeds (60553, 16912, 94993) unter derselben Run-ID vermischt. Das sind keine eindeutigen 55-/52-Epochen-Lernkurven. Die Übersicht zeigt lediglich den letzten CSV-Eintrag des letzten Epochenindex, hier Seed 94993; die anderen Fortsetzungen stehen unten. Welcher Zweig dem zuletzt überschriebenen Checkpoint entspricht, ist damit nicht gesichert. Der MSE-only-Run `ca1b44` zeigt aber, dass das Netz grundsätzlich sehr kleine Datenfehler erreichen konnte.

### 4.1 Statische Versuche: zusätzlich vertauschte Argumente
In den historischen `CombinedLoss`-Versionen, auch noch im Snapshot `4a7f6de`, wird `physics_loss(current_input, output)` aufgerufen, obwohl `HeatEquationLoss.forward(model_output, current_input)` die umgekehrte Bedeutung erwartet. Daraus folgt für diese Version:

```math
r_static_actual = (u₀ − v(t))/t − α L_unscaled(v(t)) − s(v(t))
```

Neben der falschen Richtung der Zeitableitung wird die Quelle damit aus der Vorhersage statt aus dem Anfangsfeld abgeleitet. Nach Aktivierung der normierten Quellschwelle kann die Vorhersage die Quellmaske selbst verändern. Das ist ein gewichtiger Störfaktor für die statischen Ergebnisse. Am 13.03. wird die statische Schnittstelle im Rahmen der expliziten Quelle umgebaut; die hier vorhandenen statischen Runs stammen davor. Die exakte lokale Version einzelner früher Läufe bleibt ungesichert.

| Serie | Horizont s | Run +Physik / Daten | Gesamt +Physik, Ep.10 | MSE nur Daten, Ep.10 |
| --- | --- | --- | --- | --- |
| 09.02. | 0.5 | 952339 / d56387 | 2.203e-04 | 2.507e-04 |
| 09.02. | 3 | 5f5ad2 / 4624df | 1.038e-04 | 4.672e-04 |
| 09.02. | 10 | eb9d45 / 287556 | 7.568e-05 | 9.032e-05 |
| 16.02. f32 | 0.5 | 6f3462 / 704464 | 1.338e-02 | 5.645e-05 |
| 16.02. f32 | 3 | 778736 / e4eb74 | 2.204e-02 | 6.735e-05 |
| 16.02. f32 | 10 | 34bf39 / 087ac2 | 5.198e+00 | 1.231e-04 |
| 19.02. f64 | 0.5 | 6064a1 / 59f80b | 1.346e-02 | 8.071e-05 |
| 19.02. f64 | 3 | 6ada61 / 795aad | 2.290e-02 | 3.462e-04 |
| 19.02. f64 | 10 | feabe2 / fbae34 | 2.642e+00 | 5.471e-04 |

**Ableitbar:** Der starke Anstieg des statischen Physik-Loss nach Aktivierung der Quelle und die problematischen langen Horizonte passen zu einer inkonsistenten Nebenbedingung. Die Tabelle beweist keine Überlegenheit oder Unterlegenheit „physikalischer CNNs“ als Methode. float64 beseitigt die strukturellen Fehler nicht.

## 5. 13.–20. Februar: Quelle normieren, dann zwei Modellvorhersagen
Mit `fcaa462` wird der dynamische Quellterm mathematisch passend zur Temperaturtransformation skaliert:

```math
R = 27373,34765625 − 20 = 27353,34765625
s = 100000/R ≈ 3,65586
Schwelle_norm = (1000−20)/R ≈ 0,035828
```

Die frühere leere Maske wird dadurch aktiv. Die Sekante und der unskalierte Laplace-Kernel bleiben zunächst bestehen. Ein höherer Loss nach dieser Änderung bedeutet daher nicht automatisch schlechteres Lernen: Jetzt wird überhaupt erst eine nichttriviale Quelle berücksichtigt. `d977bec` verändert den dynamischen Loss selbst nicht erneut; der Commit betrifft unter anderem Run-Metadaten. Die relevante Codeänderung ist `fcaa462`.

Ab `cd47e19` ersetzt du die Langzeit-Sekante durch zwei Auswertungen desselben Netzes, beide mit demselben Anfangsfeld:

```math
v(t) = fθ(u₀,t),   v(t−h) = fθ(u₀,t−h),   h=0,1 s
r = [v(t)−v(t−h)]/h − α L_unscaled(u₀) − s(u₀)
```

Das ist **kein autoregressiver Rollout**: v(t−h) wird nicht als Eingang der zweiten Vorhersage genutzt. Beide Ausgaben sind direkt an u₀ und Zeit konditioniert, und Gradienten laufen durch beide Zweige. Die Zeitableitung wird lokal, aber die räumliche Diffusion ist weiterhin am Anfang eingefroren. Zudem entspricht h=0,1 hundert internen Simulationsschritten von je 0,001 s; der ein-Schritt-Euler-Residual ist damit nicht exakt der Datenoperator.

| Run / Variante | Ep. | λP / λD | Val.-MSE | Val.-Physik | Val.-Gesamt |
| --- | --- | --- | --- | --- | --- |
| 606d45 · timefirst, normierte Quelle | 25 | 1.0/1.0 | — | — | 2.330e-02 |
| 388723 · Wiederholung normierte Quelle | 25 | 1.0/1.0 | — | — | 2.329e-02 |
| 8b5b13 · latent nur Daten | 10 | 0.0/1.0 | 4.444e-06 | — | 4.444e-06 |
| 88ed97 · latent + normierte Quelle | 10 | 1.0/1.0 | — | — | 2.328e-02 |
| 75c372 · timefirst nur Daten | 10 | 0.0/1.0 | 4.193e-06 | — | 4.193e-06 |
| 7051f0 · timefirst + Quelle | 10 | 1.0/1.0 | — | — | 2.317e-02 |
| eb07d7 · f64 laut Name | 10 | 1.0/1.0 | — | — | 2.321e-02 |
| 58287a · timefirst f64 laut Name | 10 | 1.0/1.0 | — | — | 2.320e-02 |
| 26b526 · zwei Vorhersagen, Quelle | 10 | 1.0/1.0 | — | — | 2.312e-02 |
| 8475da · timefirst, zwei Vorhersagen | 10 | 1.0/1.0 | — | — | 2.336e-02 |

**Ergebnis:** Mehrere Physik-Runs bleiben bei etwa 0,023 Gesamt-Loss, während Daten-Baselines etwa 4e-6 MSE erreichen. Die neue Zeitableitung und höhere Präzision reichen allein nicht. Mangels Komponenten in diesen frühen Runs lässt sich das Plateau nicht quantitativ in Daten- und PDE-Fehler zerlegen. Die float-Angaben sind teilweise nur Run-Namen/-Kommentare, nicht aufgezeichnete Runtime-Dtypes.

### 5.1 Sonderzweig: separates PINN
Das PINN nutzt automatische erste und zweite Ableitungen eines MLP über [x,y,z,t]. Im damaligen Dataloader werden alle Koordinaten auf [0,1] skaliert, während `heat_residual` dieselbe α vor die ungewichtete Summe der Koordinatenableitungen setzt. Mit ξ=x/Lx und τ=t/H wäre dagegen erforderlich:

```math
(1/H) u_τ − α[(1/Lx²)u_ξξ + (1/Ly²)u_ηη + (1/Lz²)u_ζζ] − s = 0
```

Zusätzlich bekommt das damalige MLP weder das jeweilige Anfangsfeld noch ein Experimentkennzeichen als Eingang. Mehrere zufällige Quellanordnungen können am selben [x,y,z,t] unterschiedliche Temperaturen verlangen; eine einzige Funktion dieser vier Koordinaten kann sie nicht allgemein unterscheiden. Der Quellterm wird zwar im Loss verwendet, macht die Modellvorhersage aber nicht selbst experimentabhängig. Ein separates starkes Anfangs-/Randbedingungs-Loss ist im damaligen Trainingsskript nicht implementiert. Das ist eine plausible strukturelle Erklärung, keine nachträglich isolierte Ablation.

| PINN-Epoche | Val.-Daten | Val.-PDE | Val.-Gesamt |
| --- | --- | --- | --- |
| 1 | 4.871e-04 | 2.501e-02 | 2.550e-02 |
| 50 | 4.853e-04 | 2.482e-02 | 2.530e-02 |

Der Gesamt-Loss sinkt nur um rund 0,76 %. Das belegt einen stagnierenden konkreten PINN-Versuch, nicht die generelle Untauglichkeit von PINNs. Quellen: `13cf5d8:dataset.py`, `models/pinn.py`, `training/train_pinn.py`; Run `pinn_20260220-154004`.

## 6. 20.–25. Februar: Quellen-Rücknahme, Laplace am Output, kleinere Schritte
`13cf5d8` setzt die Quellerkennung vorübergehend wieder auf die rohe Schwelle 1000 zurück. Auf normalisiertem u₀ wird die Quelle dadurch typischerweise null. Das erklärt, warum „OLDSOURCE“-Runs plötzlich sehr kleine Residuen zeigen können: Sie lösen eine leichtere, andere Gleichung. Nicht als erfolgreiche Behebung des ursprünglichen Quellenproblems interpretieren.

Danach wird die Diffusion von u₀ auf v(t) verschoben. Mit `2237a9c` sind im committeden Stand gleichzeitig normierte Quelle, Schwelle 500, fein gespeicherte Daten und h=0,001 enthalten:

```math
r_current = [vₙ−vₙ₋₁]/h − α L_unscaled(vₙ) − s(u₀)
```

Mit einem korrekt skalierten L wäre das eine rückwärts-/implizit-Euler-artige Nebenbedingung. Die Daten entstehen aber explizit aus dem vorherigen Zustand. Der Wechsel von L(u₀) zu L(vₙ) behebt das eingefrorene Anfangsfeld, trifft aber noch nicht exakt den Simulator. Zudem fehlen hier weiterhin die räumlichen Skalierungsfaktoren und die Trennung von Rand- und inneren Zellen.

| Run / Variante | Ep. | λP / λD | Val.-MSE | Val.-Physik | Val.-Gesamt |
| --- | --- | --- | --- | --- | --- |
| 7d625b · OLDSOURCE, f64 im Namen | 10 | 1.0/1.0 | — | — | 5.605e-06 |
| 96df19 · OLDSOURCE latent | 10 | 1.0/1.0 | 6.110e-06 | 2.490e-06 | 8.599e-06 |
| 3523f7 · latent nur Daten | 10 | 0.0/1.0 | 1.436e-06 | — | 1.436e-06 |
| 54d367 · früher timefirst-Lauf | 10 | 1.0/1.0 | 6.081e-06 | 3.167e-06 | 9.248e-06 |
| c8f929 · timefirst nur Daten | 10 | 0.0/1.0 | 2.463e-06 | — | 2.463e-06 |
| 99922b · L(output) | 10 | 1.0/1.0 | 1.181e-03 | 2.184e-02 | 2.302e-02 |
| 46d966 · feine Daten, final_straw | 50 | 1.0/1.0 | 3.755e-03 | 1.034e-01 | 1.072e-01 |
| 17af3a · timefirst, final_straw | 50 | 1.0/1.0 | 4.906e-03 | 1.026e-01 | 1.075e-01 |
| d6c999 · feine Daten, nur Daten | 50 | 0.0/1.0 | 4.256e-07 | — | 4.256e-07 |

**Ableitbar:** Niedrige Residuen der OLDSOURCE-Phase sind durch die abgeschaltete Quelle konfundiert. Die hohen Werte um 0,107 bei den fein aufgelösten Daten sind ebenfalls kein reiner Effekt „kleineres dt ist schlechter“: Quelle, Laplace-Zustand, Datenauflösung und teilweise Normalisierung wechselten gleichzeitig. Der Lauf `99922b` beginnt vor dem zugehörigen Commit und dokumentiert L(output) im Kommentar; sein vollständiger lokaler Zustand lässt sich nicht exakt rekonstruieren.

Bei h:0,1→0,001 wächst die Empfindlichkeit gegenüber einer festen Differenz der beiden Vorhersagefehler um Faktor 100; ihr quadratischer Beitrag kann um Faktor 10000 wachsen. Das gilt bei festgehaltenem Fehlerunterschied, nicht zwangsläufig für korrelierte glatte Modellfehler. Bei float32 wird zusätzlich die Subtraktion fast gleicher Ausgaben empfindlicher. Diese Skalierung erklärt, warum ein unverändertes λP keinen unveränderten Optimierungsdruck garantiert.

## 7. 26. Februar–4. März: V0.2/V0.3, konsistenteres Euler-Residuum
V0.2 korrigiert den räumlichen Operator von L_unscaled zu Lh und wertet das Residuum nur auf inneren Zellen aus. Der z=0-Rand wird von einem über Batch und Fläche broadcasteten Skalar auf die vollständige Eingangsfläche umgestellt. V0.3 verwendet schließlich den Laplace-Term der vorherigen Modellvorhersage:

```math
r_V03 = P{ [vₙ−vₙ₋₁]/h − α Lh(vₙ₋₁) − s(u₀) }
```

Bei korrekten h, Quelle und Datenrandbedingungen entspricht das einem expliziten Euler-Schritt. Der Code-Commit `1ee33d4` fasst die Änderungen zusammen; die V0.2/V0.3-Kommentare der Runs erlauben eine feinere zeitliche Zuordnung, aber keinen lückenlosen Snapshotnachweis.

| Run / Variante | Ep. | λP / λD | Val.-MSE | Val.-Physik | Val.-Gesamt |
| --- | --- | --- | --- | --- | --- |
| 1b9c31 · V0.2 latent | 10 | 1.0/1.0 | 7.034e-05 | 3.320e-05 | 1.035e-04 |
| f539ad · V0.2 timefirst | 9 | 1.0/1.0 | 6.195e-05 | 1.020e-04 | 1.639e-04 |
| 52aa35 · V0.2 latenttime1 | 9 | 1.0/1.0 | 6.569e-05 | 3.494e-05 | 1.006e-04 |
| 94b47b · V0.3 a=0,1 | 10 | 0.1/1.0 | 3.260e-05 | 1.404e-04 | 4.664e-05 |
| 1123ba · V0.3 a=1 | 10 | 1.0/1.0 | 6.148e-05 | 3.024e-05 | 9.172e-05 |
| 056946 · V0.3 timefirst | 10 | 1.0/1.0 | 5.317e-05 | 4.881e-05 | 1.020e-04 |
| b6153e · V0.3 latenttime1 | 10 | 1.0/1.0 | 6.363e-05 | 3.040e-05 | 9.403e-05 |
| a91705 · V0.3 lr=1e-4 | 10 | 1.0/1.0 | 6.287e-05 | 3.823e-05 | 1.011e-04 |

**Ergebnis:** Bei a=0,1 ist der Datenfehler kleiner als bei a=1 (3,26e-5 gegenüber 6,15e-5), der Physikfehler aber größer (1,40e-4 gegenüber 3,02e-5). Das zeigt den Zielkonflikt direkt innerhalb der Komponenten. Beide Runs nennen denselben Seed 897496; der Code setzt den Seed jedoch zentral und speichert die Splits nicht, sodass identische Seed-Zahlen allein keinen identischen Split oder Netzstart beweisen. Andere Architektur-/Lernratenvergleiche verwenden unterschiedliche Seeds.

Die Quellenverlagerung im Generator beseitigt das Problem nur für neu erzeugte Daten. Zahlreiche V0.3/V0.4-Runs verwenden laut Config weiterhin `new_detailed_heat_sim_f64`; die vorhandenen Dateien stammen vom 23.02. und besitzen die problematischen heißen Anfangsränder. Die Verbesserung des Loss-Codes ist daher nicht gleichbedeutend mit einem vollständig konsistenten Trainingsproblem.

| Run / Variante | Ep. | λP / λD | Val.-MSE | Val.-Physik | Val.-Gesamt |
| --- | --- | --- | --- | --- | --- |
| d1a417 · 6 Experimente, beide Losses | 50 | 1.0/1.0 | 1.328e-05 | 1.982e-05 | 3.310e-05 |
| 1b1203 · 6 Experimente, nur Daten | 50 | 0.0/1.0 | 7.401e-06 | — | 7.401e-06 |
| 18f41e · 30 Exp., lr=1e-4 | 50 | 1.0/1.0 | 5.903e-05 | 2.677e-05 | 8.581e-05 |
| 6b06f9 · 30 Exp., lr=1e-4 Wiederholung | 50 | 1.0/1.0 | 6.193e-05 | 2.876e-05 | 9.070e-05 |
| ac802b · 30 Exp., latenttime1 | 50 | 1.0/1.0 | 5.885e-05 | 2.424e-05 | 8.309e-05 |
| 9f831c · 30 Exp., timefirst nur Daten | 50 | 0.0/1.0 | 1.428e-06 | — | 1.428e-06 |
| 260835 · latenttime1 nur Physik | 50 | 1.0/0.0 | — | 1.573e-06 | 1.573e-06 |
| 4054e1 · 29 Exp., nur Daten | 100 | 0.0/1.0 | 7.273e-06 | — | 7.273e-06 |
| 824698 · 29 Exp., nur Physik | 89 | 1.0/0.0 | — | 2.825e-07 | 2.825e-07 |
| 45ed6b · 29 Exp., timefirst nur Physik | 84 | 1.0/0.0 | — | 1.794e-06 | 1.794e-06 |

**Ableitbar:** Im Sechs-Experimente-Vergleich mit je 50 Epochen ist der MSE des kombinierten Laufs um Faktor 1,79 größer als der Daten-Baseline. Kleinere Lernrate und 50 Epochen beseitigen den Konflikt nicht. Die kleinen Physics-only-Werte sind keine gemessene Temperaturgenauigkeit. `824698` und `45ed6b` wurden nach 89 bzw. 84 vollständigen Epochen abgebrochen; Logs belegen Abbruch, nicht mathematische Divergenz.

### 7.1 Ein leicht zu übersehender Phasenwechsel
Run `61a255` heißt DATADRIVENONLY, enthält aber 50 Epochen nur MSE und danach 50 Epochen nur Physik. Die globalen Startgewichte der Config sind kein verlässlicher Ersatz für den Schedule bzw. die CSV.

| Run / Variante | Ep. | λP / λD | Val.-MSE | Val.-Physik | Val.-Gesamt |
| --- | --- | --- | --- | --- | --- |
| 61a255 · Datenphase | 50 | 0.0/1.0 | 8.036e-06 | — | 8.036e-06 |
| 61a255 · erste Physik-Epoche | 51 | 1.0/0.0 | — | 3.480e-06 | 3.480e-06 |
| 61a255 · Ende Physikphase | 100 | 1.0/0.0 | — | 2.400e-07 | 2.400e-07 |

Das Residuum fällt von 3,48e-6 auf 2,40e-7 während der Physikphase. Ob die Vorhersage dabei besser oder schlechter wird, wurde nicht gemessen: Ab Epoche 51 ist der MSE deaktiviert. Der Übergang von 8,04e-6 „Loss“ zu 2,40e-7 darf deshalb nicht als Genauigkeitsgewinn um Faktor 33 gelesen werden.

## 8. 5.–10. März: dt aus Metadaten und der erste Zeitschritt
Mit `0a2e3e6` wird bei t=h der vergangene Zustand explizit durch u₀ ersetzt. Das verankert den ersten Euler-Schritt an der bekannten Anfangsbedingung, statt auch fθ(u₀,0) frei lernen zu lassen. Diese Behandlung wird beim großen Refactor `59472c7` wieder entfernt; seitdem nutzt `model_past` auch am Anfang eine zweite freie Vorhersage.

`5408f5d` lässt den Dataloader die Zeitbasis aus Metadaten lesen, aber im Trainer bleibt h noch fest 0,001 s. `fe8e545` vom 09.03. beseitigt den Fallback und übernimmt `dataset.dt` auch im Trainer. Damit entsteht eine wichtige Übergangsphase auf dem dt=0,01-Datensatz: Je nach lokalem Metadatenstand konnte die Eingangszeit um Faktor 10 falsch zugeordnet sein oder der Loss zwischen nicht gespeicherten Zwischenzeiten mit h=0,001 differenzieren. Letzteres ist für ein kontinuierliches Modell nicht an sich verboten, ist aber kein exakter Ein-Schritt-Test gegen den Euler-Datenoperator mit dt=0,01.

| Run / Variante | Ep. | λP / λD | Val.-MSE | Val.-Physik | Val.-Gesamt |
| --- | --- | --- | --- | --- | --- |
| 022c50 · 40 Exp., MSE-only | 100 | 0.0/1.0 | 1.146e-06 | — | 1.146e-06 |
| 152003 · 40 Exp., Physics-only | 100 | 1.0/0.0 | — | 2.634e-06 | 2.634e-06 |

Die Runs vom 05.03. liegen vor der committeden dt-Korrektur. Ihre kleinen Endwerte belegen daher noch keinen Erfolg der endgültigen Zeitbehandlung. Die sechs Runs vom 10.03. (`072061`, `88b108`, `b5847c`, `501759`, `a211c1`, `22cc8c`) haben keine protokollierte Epoche. Logs der zugehörigen Jobs enthalten `NameError: _require_zarr is not defined`. Frühe Jobs vom 05.03. scheitern außerdem an einem Index außerhalb der 1001 gespeicherten Frames. Das sind Pipelinefehler, keine Resultate zur Qualität des Physics-Loss.

## 9. 13. März: Explizite Quelle und drei verschiedene mathematische Ziele
`79056df`, `7a9e58c`, `59472c7` trennen T₀, Quellmaske und Q-Feld, führen konsistente Metadaten ein und übergeben den normierten Quellterm explizit. Der Laplace-Operator verwendet die echten anisotropen Abstände. Simulation und Loss nutzen Umgebungstemperatur zum Padding; Randzellen bleiben im Residuum ausgeschlossen. Dadurch wirkt das äußere Padding bei einem Radius-1-Stencil nicht direkt auf die ausgewerteten Innenzellen; entscheidend sind die tatsächlichen Randwerte selbst.

### 9.1 model_past: Dynamik der Vorhersagen, keine GT-Zeitableitung
```math
r_model = P{ (vₙ−vₙ₋₁)/dt − α Lh vₙ₋₁ − s }
Bei eₙ=vₙ−yₙ und konsistenten Daten:
r_model = P{ (eₙ−A eₙ₋₁)/dt }
```

Das Residuum misst die **Fortpflanzung des Fehlers**, nicht dessen absoluten Betrag. Eine falsche Anfangslösung mit Fehlerentwicklung eₙ=Aeₙ₋₁ kann LP=0 erfüllen. Darum braucht diese Variante eine gesicherte Anfangsbedingung oder genügend Datensupervision. Seit dem März-Refactor ist die besondere Verankerung am ersten Zeitschritt wieder entfallen. Zwei Vorhersagen erlauben zudem korrelierte Fehler, die sich im Residuum kompensieren.

### 9.2 gt_past: Ein-Schritt-Target aus der vergangenen Ground Truth
```math
r_gt_past = P{ (vₙ−yₙ₋₁)/dt − α Lh yₙ₋₁ − s }
yₙ = A yₙ₋₁ + dt s  auf inneren Zellen
⇒ r_gt_past = P(vₙ−yₙ)/dt = P eₙ/dt
⇒ LP = MSE_inner / dt²
```

Das ist unter diesen Datenbedingungen ein **supervidierter Fehler mit anderer Skalierung**, keine unabhängige physikalische Information zusätzlich zum Target. Bei harten korrekten Rändern ist der Randfehler null; der MSE wird aber über alle 32768 Zellen gemittelt, LP über 26040 innere Zellen. Daher gilt hier:

```math
LP / LD = (64·32·16)/(62·30·14) · 1/0,1² = 125,83717358
L_gesamt = (λD + 125,83717358 λP) LD
```

Bei λP=1 ist der kombinierte Loss also etwa 126,84-mal der MSE, bei λP=0,01 etwa 2,2584-mal. Das Optimum derselben Batch-Aufgabe ändert sich durch einen konstanten Faktor nicht. Adam ist gegenüber globaler Skalierung weitgehend, aber wegen ε und vorhandener Momentzustände nicht exakt invariant; Warmstart und Logging bleiben gesondert zu betrachten. Unterschiede zwischen unabhängig gestarteten gt_past- und MSE-only-Runs beweisen deshalb nicht, dass „zusätzliche Physik“ geholfen oder geschadet hat.

| Run | Ep. | LD | LP | Gemessen LP/LD |
| --- | --- | --- | --- | --- |
| ad89e8 | 20 | 4.063e-05 | 5.112e-03 | 125.83717032 |
| e2cca7 | 15 | 6.098e-06 | 7.674e-04 | 125.83717219 |
| d013c6 | 14 | 4.229e-06 | 5.321e-04 | 125.83716987 |
| c950fe | 30 | 3.648e-06 | 4.591e-04 | 125.83717133 |

Die Übereinstimmung bis auf float32-/Mittelungsfehler ist ein direkter numerischer Nachweis dieser Interpretation, nicht bloß eine theoretische Vermutung.

### 9.3 gt_future: Fehler nach einem expliziten Vorwärtsschritt
```math
r_gt_future = P{ (yₙ₊₁−vₙ)/dt − α Lh vₙ − s }
⇒ r_gt_future = −P A eₙ/dt
⇒ LP = mean_inner((A eₙ)²)/dt²
```

Hier muss die aktuelle Vorhersage nach einem Euler-Schritt zur zukünftigen Ground Truth passen. Das ist ein räumlich gefilterter supervidierter Fehler. Er koppelt Nachbarfehler und unterscheidet sich damit tatsächlich vom gewöhnlichen MSE, nutzt aber weiterhin Simulationslabels. Für einen Laplace-Eigenmodus Lh φ=−μ φ wird dessen Fehler mit (1−dt α μ)²/dt² gewichtet. Unter der hier gewählten stabilen Diskretisierung dämpft A insbesondere feinräumige Fehlermoden; „gt_future bestraft automatisch alle hohen Frequenzen stärker“ wäre falsch. Bei den Stahlparametern liegt die konservative Untergrenze des A-Faktors bei rund 0,855, sodass der Filter in einem Schritt relativ moderat wirkt.

### 9.4 Ground-Truth-Test der Diskretisierung
Je drei vorhandene Experimente aus Debug und Medium wurden an n=1,100,600,1200 geprüft (24 Zustandsdreiergruppen). Mit float64 und den gespeicherten Metadaten liegen die mittleren quadratischen Residuen von gt_past und gt_future zwischen etwa 1,08e-32 und 2,96e-32. Die aktuelle Datenrepräsentation ist damit in dieser Stichprobe exakt zum expliziten Euler-Residual konsistent. Das prüft die Dateien von heute, nicht rückwirkend einen unbekannten Trainingssnapshot. Details stehen in `ground_truth_residual_checks.json`.

### 9.5 Ergebnisse auf steel_debug_small

| Run / Variante | Ep. | λP / λD | Val.-MSE | Val.-Physik | Val.-Gesamt | Min. MSE (Ep.) |
| --- | --- | --- | --- | --- | --- | --- |
| 908735 · nur Daten | 20 | 0.0/1.0 | 1.969e-05 | — | 1.969e-05 | 1.969e-05 (20) |
| 3f65f8 · gt_future | 20 | 1.0/1.0 | 1.098e-05 | 1.283e-03 | 1.294e-03 | 1.098e-05 (20) |
| ad89e8 · gt_past | 20 | 1.0/1.0 | 4.063e-05 | 5.112e-03 | 5.153e-03 | 2.137e-05 (19) |
| 477782 · model_past | 20 | 1.0/1.0 | 3.021e-04 | 1.600e-05 | 3.181e-04 | 2.945e-04 (18) |

Alle vier Runs: sechs Experimente, 7200 Samples, 20 Epochen, PITCNN_dynamic, unterschiedliche Seeds. gt_future erreicht am Ende rund 44 % weniger MSE als die Daten-Baseline, gt_past etwa das 2,06-Fache und model_past das 15,35-Fache. **Ableitbar:** gt_future war in diesem kleinen Versuch vielversprechend. Wegen unterschiedlicher Splits/Initialisierungen und der algebraischen gt_past-Identität ist das kein kausaler Nachweis eines generellen Physikvorteils. Ein kleines LP bei model_past kann mit großem Datenfehler zusammengehen, genau wie die Fehlerfortpflanzungsformel erwarten lässt.

## 10. 23.–27. März: Größerer Stahl-Datensatz und Gewichte
Alle vier ursprünglichen Medium-Runs nutzen PITCNN_dynamic, 32 Experimente und 38400 Samples. Dateinamen mit „timefirst“ ändern diese aufgezeichnete Modellklasse nicht. Die für 100 Epochen angesetzten Jobs wurden laut Logs nach 15 vollständigen Epochen während der nächsten Epoche durch das Zeitlimit beendet. Der Datenrun wurde später fortgesetzt.

### 10.1 Vergleich bei derselben vollständigen Epoche 14

| Run / Variante | Ep. | λP / λD | Val.-MSE | Val.-Physik | Val.-Gesamt |
| --- | --- | --- | --- | --- | --- |
| 7e2ea6 · nur Daten | 14 | 0.0/1.0 | 2.187e-06 | — | 2.187e-06 |
| e6895c · gt_future λP=1 | 14 | 1.0/1.0 | 3.716e-06 | 4.339e-04 | 4.377e-04 |
| e2cca7 · gt_past λP=1 | 14 | 1.0/1.0 | 5.409e-06 | 6.807e-04 | 6.861e-04 |
| 7a6070 · model_past λP=1 | 14 | 1.0/1.0 | 2.882e-04 | 1.919e-05 | 3.074e-04 |
| 24c1fb · gt_future λP=0,01 | 14 | 0.01/1.0 | 2.230e-05 | 2.772e-03 | 5.002e-05 |
| d013c6 · gt_past λP=0,01 | 14 | 0.01/1.0 | 4.229e-06 | 5.321e-04 | 9.550e-06 |
| 6275b7 · model_past λP=0,01 | 14 | 0.01/1.0 | 1.692e-05 | 3.887e-03 | 5.579e-05 |

Bei λP=1 ist der MSE von gt_future etwa 1,70-mal, von gt_past 2,47-mal und von model_past 131,77-mal so groß wie der Baseline-MSE. Der Vorteil von gt_future aus dem kleinen Debug-Versuch reproduziert sich somit in diesen unabhängigen Medium-Runs nicht.

Zur Größenordnung: Mit R=101,25644 entspricht der Baseline-MSE bei Epoche 14 einem globalen RMSE von etwa 0,150 K, gt_future mit λP=1 etwa 0,195 K und model_past etwa 1,719 K. Das ist R·sqrt(MSE), über das ganze Feld einschließlich der fehlerfreien Ränder; lokale Fehler nahe Quellen können deutlich größer sein. Einzelne Batch-Mittel werden im Trainer ungewichtet gemittelt, daher ist das eine Umrechnung der protokollierten Metrik und keine neu berechnete voxelgewichtete Testauswertung.

Die Reduktion von λP auf 0,01 senkt bei model_past den MSE gegenüber dem λP=1-Lauf um etwa Faktor 17,0, lässt aber das Residuum deutlich größer werden. Das passt zu einer zu stark oder widersprüchlich wirkenden Kopplung. Bei gt_past verändert die Gewichtung unter den genannten Voraussetzungen im Wesentlichen nur die globale Skalierung; eine Interpretation als neuer physikalischer Kompromiss wäre irreführend.

Der gt_future-Run `24c1fb` endet ungünstig, hat aber zwischenzeitlich einen MSE von 2,386e-6 erreicht. Die Endpunkt-Tabelle allein verdeckt solche Schwankungen. Minima der einzelnen Kurven stehen im Inventar; sie werden nicht ohne Kennzeichnung gegen Endpunkte anderer Läufe verglichen. Alle hier gestarteten Medium-Vergleichsruns haben verschiedene Seeds.

### 10.2 MSE-Warmstart und Wechsel zu Physik
Der bestehende Datenrun wird auf 20 Epochen fortgesetzt. Zwei neue Ordner übernehmen daraus Modell, Optimizer-Zustand und Epochenzähler und trainieren Epochen 21–30 mit λD=λP=1.

| Run / Variante | Ep. | λP / λD | Val.-MSE | Val.-Physik | Val.-Gesamt |
| --- | --- | --- | --- | --- | --- |
| 7e2ea6 · Daten-Basis | 20 | 0.0/1.0 | 2.250e-06 | — | 2.250e-06 |
| c950fe · gt_past nach Wechsel | 21 | 1.0/1.0 | 4.972e-05 | 6.256e-03 | 6.306e-03 |
| c950fe · gt_past Ende | 30 | 1.0/1.0 | 3.648e-06 | 4.591e-04 | 4.627e-04 |
| 43631b · gt_future nach Wechsel | 21 | 1.0/1.0 | 8.731e-05 | 1.046e-02 | 1.055e-02 |
| 43631b · gt_future Ende | 30 | 1.0/1.0 | 5.725e-06 | 6.877e-04 | 6.934e-04 |

Unmittelbar nach dem Wechsel liegt der MSE wesentlich höher und erholt sich anschließend. Am Ende ist er noch 1,62-mal (gt_past) bzw. 2,54-mal (gt_future) so groß wie die protokollierte Baseline bei Epoche 20. Diese Zahlen zeigen keinen Genauigkeitsgewinn durch das Nachtraining.

**Aber:** Die neuen Runs setzen neue Seeds und erzeugen den zufälligen Train/Val-Split erneut. Damit sind es keine unveränderten Validierungsmengen; nach einem Splitwechsel können zuvor trainierte Samples in der neuen Validation liegen. Zusätzlich fällt der Start der Follow-ups zeitlich nach den Commit zur Zeitskalierung. Wurde der Basis-Checkpoint noch mit Sekunden als Zeiteingang trainiert und der Follow-up mit t/H, ändert sich beim Warmstart die Bedeutung desselben Eingangs. Der Commit-Verlauf macht das möglich, der fehlende Runtime-Snapshot erlaubt aber keine sichere Behauptung für den konkreten Basisrun. Der beobachtete Sprung kann nicht allein dem neuen Loss zugeschrieben werden.

## 11. 27.–31. März: Soft-Physics, Zeitnormierung und Physics-only
### 11.1 Soft-Physics verändert die Gleichung
```math
r_soft = P{ [vₙ−vₙ₋₁]/dt − α Lh(u₀) }
```

Die Quelle wird bewusst entfernt und die räumliche Diffusion am Anfang eingefroren. Unter korrekter Zeitbehandlung würde r_soft≈0 eine nahezu lineare zeitliche Fortschreibung mit konstantem Anfangs-Diffusionsterm fördern. Das ist keine bloß schwächere Gewichtung derselben Wärmeleitungsgleichung. Es ähnelt strukturell dem frühen eingefrorenen Laplace-Ansatz, nun mit zwei Modellvorhersagen und ohne Quelle.

Auf der tatsächlich gespeicherten Ground Truth ist dieses Residuum **nicht** null: In der oben genannten Stichprobe beträgt sein MSE 6,54e-5 bis 1,51e-4, während das korrekte Euler-Residuum bei etwa 1e-32 liegt. Das beweist eine gezielte Modellabweichung. Ein kleinerer Soft-Residual als auf der Ground Truth kann gerade durch eine Abweichung von den richtigen Temperaturen erreicht werden. Dieser Check isoliert die Definition von Soft-Physics; er verwendet den richtigen zeitlichen Nachbarn und ist kein Nachrechnen des tatsächlichen Netz-Loss.

Die ersten beiden Soft-Runs `f87953` und `4325ed` scheitern ohne Epoche an zwei Eingangskanälen vor einer ein-kanaligen Laplace-Faltung. Der lokale Fix verwendet nur Kanal 0 (T₀). Der erfolgreiche Run vom 30.03. ist `db1749`.

### 11.2 Nachgewiesener Einheitenfehler im letzten Code-Stand
Seit `4867499` liefert `data.py` die Modellzeit τ=t/H. Im Trainer wird aber weiterhin `t_past=t−dt` gerechnet, wobei t nun τ bedeutet und dt Sekunden bezeichnet:

```math
korrekt:   τ_past = τ − dt/H
im Code:   τ_past = τ − dt
bei H=120 s und dt=0,1 s:
korrekter Modellabstand = 0,000833333
verwendeter Modellabstand = 0,1  ⇒ physikalischer Abstand 12 s
```

Im Nenner des Loss steht trotzdem 0,1 s. Die zweite Vorhersage liegt 120 Simulationsschritte zurück; für die ersten 119 Zielzeitpunkte wird sogar eine negative Modellzeit abgefragt. Betroffen sind im aktuellen Code beide Zweigvarianten `model_past` und `soft_physics`. `gt_past` und `gt_future` nutzen echte Nachbarframes und sind von genau diesem zweiten-Vorhersage-Fehler nicht betroffen.

Ein zusätzlicher Diagnosecheck ersetzt die Modellzustände durch korrekte Simulationsfelder im Abstand von 120 Schritten, dividiert aber durch 0,1. Bei n=600 und 1200 ergeben sich Residual-MSEs von 4,06e-4 bis 4,24e-3 statt etwa 1e-32. Damit ist die Inkonsistenz auch numerisch sichtbar. Das ist eine Prüfung des Operators, kein gemessener Modellfehler.

**Zeitliche Einordnung:** Der Fehler ist im letzten Git-Stand sicher belegt. Der erfolgreiche Soft-Run vom 30.03. liegt danach; ein zeitlich passender unveränderlicher Code-Snapshot fehlt trotzdem. Die model_past-Läufe vom 13./23.03. liegen vor diesem Commit und sind daher nicht nachweislich dadurch erklärt. Auch die λP=0,01-Runs vom 27.03. starteten bereits vormittags vor dem Commit. Vordatierte uncommittete Änderungen sind möglich, aber nicht belegt. Im Bericht werden diese Runs deshalb nicht pauschal als von diesem Fehler betroffen behandelt.

| Run / Variante | Ep. | λP / λD | Val.-MSE | Val.-Physik | Val.-Gesamt |
| --- | --- | --- | --- | --- | --- |
| db1749 · Soft + Daten | 14 | 1.0/1.0 | 2.993e-04 | 2.046e-05 | 3.197e-04 |
| eb6e6e · gt_future nur Physik | 13 | 1.0/0.0 | — | 5.765e-04 | 5.765e-04 |
| 6f7386 · gt_past nur Physik | 13 | 1.0/0.0 | — | 4.288e-04 | 4.288e-04 |

Soft-Physics erreicht nach 14 Epochen 2,993e-4 MSE und liefert hier keinen Datenvorteil. Das Ergebnis testet allerdings eine geänderte Gleichung und vermutlich die problematische zweite Zeitabfrage, nicht bloß „weichere Physik“. Die beiden Physics-only-Jobs enden laut Slurm-Zeitlimit nach 13 vollständigen Epochen; eine 14. vollständige Epoche ist nicht protokolliert.

### 11.3 Was aus Physics-only trotzdem quantitativ rekonstruierbar ist
Für gt_past lässt die nachgewiesene Identität trotz deaktivierter MSE-Protokollierung eine Rekonstruktion zu:

```math
LD_rekonstruiert ≈ LP / 125,83717358
```

Für `6f7386` ergibt das bei Epoche 13 **3,407e-6**, beim kleinsten protokollierten LP **2,666e-6**. Das sind mathematisch rekonstruierte Werte unter den überprüften Diskretisierungs-/Randannahmen, keine separat gemessenen MSEs. Für gt_future geht diese einfache skalare Rückrechnung wegen A nicht; dafür braucht man die tatsächlichen Fehlerfelder oder eine neue Evaluation.

Damit ist „Physics-only“ in den GT-Modi nicht gleichbedeutend mit datenfreiem Training: Die Nachbarzustände aus der Simulation tragen die Supervision. Beim alten model_past-Physics-only fehlt diese direkte GT-Verankerung; die Varianten beantworten verschiedene Forschungsfragen.

## 12. Was die gesamte Versuchsfolge tatsächlich zeigt
**Erstens: Viele vermeintliche Optimierungsprobleme waren Probleme der Aufgabenformulierung.** Unskalierter Laplace-Kernel, zeitlich eingefrorene Diffusion, eine aus normalisierten Werten falsch rekonstruierte Quelle und widersprüchliche Randwerte erzeugten andere Ziele als der Simulator. Eine bessere Metrik nach einem Umbau kann sowohl bessere Physik als auch einen weggefallenen schwierigen Term bedeuten.

**Zweitens: Schrittweitenänderungen wirken auf drei Ebenen.** Sie ändern die Simulationsdiskretisierung, die Dichte der Daten und die Skalierung/Empfindlichkeit des Loss. Der Wechsel 0,1→0,001 war kein isolierter dt-Sweep. Der spätere Wechsel zu 0,01 und Stahl-dt=0,1 verändert wiederum Daten, Material und Geometrie. Ein übergreifendes Ranking nach dem kleinsten LP ist daher nicht sinnvoll.

**Drittens: Die drei Residuen sind keine gleichwertigen Arten, dieselbe Information einzuspeisen.** model_past prüft die Konsistenz einer gesamten gelernten Zeitentwicklung. gt_past ist hier fast exakt MSE-Supervision mit Faktor 125,84. gt_future ist ein räumlich gefilterter supervidierter Fehler. Soft-Physics passt nicht zur erzeugten inhomogenen Wärmeleitungsgleichung.

**Viertens: Die jüngsten Zahlen belegen keinen robusten Genauigkeitsvorteil des zusätzlichen Physics-Loss.** gt_future gewinnt im kleinen Debug-Endpunktvergleich, verliert aber gegen die Medium-Baseline. Eine generelle Aussage gegen physikgestütztes Lernen wäre ebenso unbegründet: Die Vergleiche enthalten unterschiedliche Splits, frühere Inkonsistenzen und teilweise geänderte Eingangskoordinaten.

**Fünftens: Die vorhandene Validation misst überwiegend zeitliche Interpolation innerhalb bekannter Experimente.** Der dynamische Dataset-Eintrag ist ein Paar (Experiment, Zeitpunkt). `random_split` verteilt einzelne Paare, nicht vollständige Experimente. Dadurch liegen typischerweise verschiedene Zeitpunkte derselben Quelle/T₀-Konfiguration in Train und Validation. Bei den GT-Modi können benachbarte Simulationszustände zusätzlich in Trainingsresiduen auftauchen. Die Metriken zeigen nicht, wie gut das Netz völlig neue Quellanordnungen oder längere Horizonte generalisiert. Kein separater reproduzierbarer Testdatensatzvergleich dieser vollständigen Run-Serie ist in den verwendeten Tabellen enthalten.

## 13. Ein sinnvoller nächster Vergleich – aus den Befunden abgeleitet
1. **Zuerst Einheiten und Anfangsbedingung festziehen:** τ_past=τ−dt/H; am ersten Schritt u₀ verwenden oder fθ(u₀,0)=u₀ explizit erzwingen. Diese Analyse ändert den Trainingscode nicht.
2. **Ein eingefrorenes Dataset und feste Splits pro Experiment:** identische Experimente und Zeitschritte, identische Initialisierung, gleiche Epochen und mehrere Seeds. Code-SHA, Metadatenhash, Normalisierung und Split-Indizes pro Run speichern.
3. **Alle diagnostischen Komponenten immer messen:** auch bei λD=0 den MSE auswerten; auch bei λP=0 ein einheitliches, korrektes model_past-Residuum nur zur Diagnose berechnen. Trainierte Zielfunktion und unabhängige Gütemaße getrennt führen.
4. **Den mathematischen Vergleich gezielt wählen:** MSE-only versus korrektes model_past beantwortet die Frage nach zusätzlicher Dynamik-Konsistenz. gt_past ist eine gute Implementierungsprüfung (Faktoridentität), aber hier keine eigenständige neue Informationsquelle. gt_future prüft den Nutzen des räumlich gefilterten Fehlers.
5. **dt isoliert variieren:** erst dieselben gespeicherten Felder und klar definierte Zeitabstände verwenden; bei mehreren Simulationsschritten den passenden Mehrschrittoperator berücksichtigen. Nicht gleichzeitig Quelle, Domäne, Material, Normierung und Netz verändern.
6. **Warmstarts nur bei unveränderter Bedeutung aller Eingänge:** derselbe Split und dieselbe Zeit-/Quellskalierung; ein zusätzlicher MSE-only-Follow-up dient als Kontrolle gegen bloß längeres Training.

## 14. Quellen, Nachvollziehbarkeit und vollständiges Run-Register
Die Hauptquellen sind lokale Artefakte; es wurden keine externen Informationen benötigt. Zum Nachlesen:
- Historische Formeln: `42aa6bd`, `7f9d778`, `310ebe5`, `fcaa462`, `4a7f6de`, `cd47e19`, `13cf5d8`, `2237a9c`, `1ee33d4`, `0a2e3e6`, `5408f5d`, `fe8e545`, `79056df`, `7a9e58c`, `59472c7`, `0790762`, `4867499`.
- Aktueller Einheitenfehler: `data.py`, Zeilen 500–507; `training/train_pitcnn_dynamic.py`, Zeilen 173–182 und 218–223. Erste-Schritt-Verankerung entfernt: Diff von `59472c7`.
- Residuen und deaktivierte Messung: `training/loss.py`, `CombinedLoss_dynamic.compute_components`; Quelle im Loader: `normalize_source_term`; Randwerte: `models/pitcnn_latenttime.py`.
- Daten: `data/steel_debug_small/info.json`, `data/steel_train_medium/info.json`, jeweilige Experiment-Metadaten; alte Daten unter `data/old/`.
- Fehlstarts und Zeitlimits: `logs/*_929617.err`, `*_929618.err`, `*_929619.err`, `*_930460.err` bis `*_930463.err`, `*_942733.err` bis `*_942736.err`, `*_949203.err`, `*_949207.err`, `*_954004.err`, `*_954005.err`.

`run_inventar.csv` enthält für alle 93 Runs Pfad, Namen, Seeds, Datenumfang, tatsächliche Gewichtsphasen, End-/Minimalwerte und Epochen. `run_details.json` enthält zusätzlich die zusammengeführten Epochenwerte und Originalkonfigurationen. Das folgende Register ordnet die in den Tabellen verwendeten Kürzel eindeutig zu. Gruppierungen sind eine analytische zeitliche Einordnung, kein behaupteter exakter Runtime-Code-Snapshot.

### A: Langzeit-Sekante

| Run-ID | Ep. erfasst | MSE zuletzt | Physik zuletzt | Gesamt zuletzt |
| --- | --- | --- | --- | --- |
| dynamic_20260209-152237_d056d1 | 10 | — | — | 5.646e-05 |
| dynamic_20260209-153143_27e545 | 25 | — | — | 6.052e-05 |
| dynamic_20260209-232430_ca1b44 | 10 | 7.803e-07 | — | 7.803e-07 |
| dynamic_20260209-233119_62167a | 24 | 7.362e-06 | — | 7.362e-06 |

### B: normierte Quelle

| Run-ID | Ep. erfasst | MSE zuletzt | Physik zuletzt | Gesamt zuletzt |
| --- | --- | --- | --- | --- |
| dynamic_20260213-172404_606d45 | 25 | — | — | 2.330e-02 |
| dynamic_20260216-100315_388723 | 25 | — | — | 2.329e-02 |
| dynamic_20260216-111244_d6509f | 25 | 4.175e-06 | — | 4.175e-06 |
| dynamic_20260216-111249_6d127d | 0 | — | — | — |
| dynamic_20260216-121715_ef2de9 | 10 | — | — | 2.326e-02 |
| dynamic_20260216-162201_8b5b13 | 10 | 4.444e-06 | — | 4.444e-06 |
| dynamic_20260216-165046_88ed97 | 10 | — | — | 2.328e-02 |
| dynamic_20260216-170403_75c372 | 10 | 4.193e-06 | — | 4.193e-06 |
| dynamic_20260216-173737_7051f0 | 10 | — | — | 2.317e-02 |
| dynamic_20260219-114610_eb07d7 | 10 | — | — | 2.321e-02 |
| dynamic_20260219-120631_58287a | 10 | — | — | 2.320e-02 |

### C: zwei Vorhersagen, dt=0.1

| Run-ID | Ep. erfasst | MSE zuletzt | Physik zuletzt | Gesamt zuletzt |
| --- | --- | --- | --- | --- |
| dynamic_20260220-122051_845509 | 0 | — | — | — |
| dynamic_20260220-144814_26b526 | 10 | — | — | 2.312e-02 |
| dynamic_20260220-151404_8475da | 10 | — | — | 2.336e-02 |
| dynamic_20260220-170731_7d625b | 10 | — | — | 5.605e-06 |

### D: Quelle/Laplace/dt-Umbau

| Run-ID | Ep. erfasst | MSE zuletzt | Physik zuletzt | Gesamt zuletzt |
| --- | --- | --- | --- | --- |
| dynamic_20260223-105039_96df19 | 10 | 6.110e-06 | 2.490e-06 | 8.599e-06 |
| dynamic_20260223-105339_54d367 | 10 | 6.081e-06 | 3.167e-06 | 9.248e-06 |
| dynamic_20260223-114221_3523f7 | 10 | 1.436e-06 | — | 1.436e-06 |
| dynamic_20260223-114434_c8f929 | 10 | 2.463e-06 | — | 2.463e-06 |
| dynamic_20260223-124105_99922b | 10 | 1.181e-03 | 2.184e-02 | 2.302e-02 |
| dynamic_20260223-154928_46d966 | 50 | 3.755e-03 | 1.034e-01 | 1.072e-01 |
| dynamic_20260223-161602_d6c999 | 50 | 4.256e-07 | — | 4.256e-07 |
| dynamic_20260223-161722_1d3424 | 10 | 4.922e-03 | 1.024e-01 | 1.074e-01 |
| dynamic_20260223-184320_a1208b | 10 | 2.820e-06 | — | 2.820e-06 |
| dynamic_20260225-113444_17af3a | 50 | 4.906e-03 | 1.026e-01 | 1.075e-01 |

### E: V0.2/V0.3/V0.4

| Run-ID | Ep. erfasst | MSE zuletzt | Physik zuletzt | Gesamt zuletzt |
| --- | --- | --- | --- | --- |
| dynamic_20260226-132516_1b9c31 | 10 | 7.034e-05 | 3.320e-05 | 1.035e-04 |
| dynamic_20260226-140730_f539ad | 9 | 6.195e-05 | 1.020e-04 | 1.639e-04 |
| dynamic_20260226-141513_52aa35 | 9 | 6.569e-05 | 3.494e-05 | 1.006e-04 |
| dynamic_20260226-163425_94b47b | 10 | 3.260e-05 | 1.404e-04 | 4.664e-05 |
| dynamic_20260226-185740_1123ba | 10 | 6.148e-05 | 3.024e-05 | 9.172e-05 |
| dynamic_20260227-083713_056946 | 10 | 5.317e-05 | 4.881e-05 | 1.020e-04 |
| dynamic_20260227-083720_b6153e | 10 | 6.363e-05 | 3.040e-05 | 9.403e-05 |
| dynamic_20260227-110947_a91705 | 10 | 6.287e-05 | 3.823e-05 | 1.011e-04 |
| dynamic_20260227-150523_260835 | 50 | — | 1.573e-06 | 1.573e-06 |
| dynamic_20260227-161748_d1a417 | 50 | 1.328e-05 | 1.982e-05 | 3.310e-05 |
| dynamic_20260227-184422_1b1203 | 50 | 7.401e-06 | — | 7.401e-06 |
| dynamic_20260302-113809_ac802b | 50 | 5.885e-05 | 2.424e-05 | 8.309e-05 |
| dynamic_20260302-114842_18f41e | 50 | 5.903e-05 | 2.677e-05 | 8.581e-05 |
| dynamic_20260302-114842_9f831c | 50 | 1.428e-06 | — | 1.428e-06 |
| dynamic_20260302-162917_6b06f9 | 50 | 6.193e-05 | 2.876e-05 | 9.070e-05 |
| dynamic_20260304-164907_824698 | 89 | — | 2.825e-07 | 2.825e-07 |
| dynamic_20260304-165121_4054e1 | 100 | 7.273e-06 | — | 7.273e-06 |
| dynamic_20260304-165630_61a255 | 100 | — | 2.400e-07 | 2.400e-07 |
| dynamic_20260304-174552_45ed6b | 84 | — | 1.794e-06 | 1.794e-06 |

### F: dt=0.01/Metadaten

| Run-ID | Ep. erfasst | MSE zuletzt | Physik zuletzt | Gesamt zuletzt |
| --- | --- | --- | --- | --- |
| dynamic_20260305-160044_022c50 | 100 | 1.146e-06 | — | 1.146e-06 |
| dynamic_20260305-160044_152003 | 100 | — | 2.634e-06 | 2.634e-06 |
| dynamic_20260310-091546_072061 | 0 | — | — | — |
| dynamic_20260310-091546_88b108 | 0 | — | — | — |
| dynamic_20260310-091546_b5847c | 0 | — | — | — |
| dynamic_20260310-092215_501759 | 0 | — | — | — |
| dynamic_20260310-092450_a211c1 | 0 | — | — | — |
| dynamic_20260310-092645_22cc8c | 0 | — | — | — |

### G: Stahl-Debug

| Run-ID | Ep. erfasst | MSE zuletzt | Physik zuletzt | Gesamt zuletzt |
| --- | --- | --- | --- | --- |
| dynamic_20260313-150029_3f65f8 | 20 | 1.098e-05 | 1.283e-03 | 1.294e-03 |
| dynamic_20260313-150032_477782 | 20 | 3.021e-04 | 1.600e-05 | 3.181e-04 |
| dynamic_20260313-150032_908735 | 20 | 1.969e-05 | — | 1.969e-05 |
| dynamic_20260313-150032_ad89e8 | 20 | 4.063e-05 | 5.112e-03 | 5.153e-03 |

### H: Stahl-Medium

| Run-ID | Ep. erfasst | MSE zuletzt | Physik zuletzt | Gesamt zuletzt |
| --- | --- | --- | --- | --- |
| dynamic_20260323-103938_7a6070 | 15 | 2.748e-04 | 2.208e-05 | 2.969e-04 |
| dynamic_20260323-103938_e2cca7 | 15 | 6.098e-06 | 7.674e-04 | 7.735e-04 |
| dynamic_20260323-104107_7e2ea6 | 20 | 2.250e-06 | — | 2.250e-06 |
| dynamic_20260323-104107_e6895c | 15 | 4.162e-06 | 4.898e-04 | 4.940e-04 |

### I: Gewicht/Warmstart/Soft-Fehlstart

| Run-ID | Ep. erfasst | MSE zuletzt | Physik zuletzt | Gesamt zuletzt |
| --- | --- | --- | --- | --- |
| dynamic_20260327-114506_d013c6 | 14 | 4.229e-06 | 5.321e-04 | 9.550e-06 |
| dynamic_20260327-115233_24c1fb | 14 | 2.230e-05 | 2.772e-03 | 5.002e-05 |
| dynamic_20260327-115233_6275b7 | 14 | 1.692e-05 | 3.887e-03 | 5.579e-05 |
| dynamic_20260327-140720_f87953 | 0 | — | — | — |
| dynamic_20260327-141428_4325ed | 0 | — | — | — |
| dynamic_20260327-191555_c950fe | 10 | 3.648e-06 | 4.591e-04 | 4.627e-04 |
| dynamic_20260327-191727_43631b | 10 | 5.725e-06 | 6.877e-04 | 6.934e-04 |

### J: Soft / Physics-only

| Run-ID | Ep. erfasst | MSE zuletzt | Physik zuletzt | Gesamt zuletzt |
| --- | --- | --- | --- | --- |
| dynamic_20260330-113324_db1749 | 14 | 2.993e-04 | 2.046e-05 | 3.197e-04 |
| dynamic_20260330-141321_6f7386 | 13 | — | 4.288e-04 | 4.288e-04 |
| dynamic_20260330-141321_eb6e6e | 13 | — | 5.765e-04 | 5.765e-04 |

### PINN

| Run-ID | Ep. erfasst | MSE zuletzt | Physik zuletzt | Gesamt zuletzt |
| --- | --- | --- | --- | --- |
| pinn_20260220-154004 | 50 | 4.853e-04 | 2.482e-02 | 2.530e-02 |

### Statisch

| Run-ID | Ep. erfasst | MSE zuletzt | Physik zuletzt | Gesamt zuletzt |
| --- | --- | --- | --- | --- |
| static_20260209-170128_952339 | 10 | — | — | 2.203e-04 |
| static_20260209-175430_5f5ad2 | 10 | — | — | 1.038e-04 |
| static_20260209-184728_eb9d45 | 10 | — | — | 7.568e-05 |
| static_20260209-193925_d56387 | 10 | 2.507e-04 | — | 2.507e-04 |
| static_20260209-203129_4624df | 10 | 4.672e-04 | — | 4.672e-04 |
| static_20260209-212330_287556 | 10 | 9.032e-05 | — | 9.032e-05 |
| static_20260216-172452_704464 | 10 | 5.645e-05 | — | 5.645e-05 |
| static_20260216-173219_e4eb74 | 10 | 6.735e-05 | — | 6.735e-05 |
| static_20260216-173912_087ac2 | 10 | 1.231e-04 | — | 1.231e-04 |
| static_20260216-174609_6f3462 | 10 | — | — | 1.338e-02 |
| static_20260216-175311_778736 | 10 | — | — | 2.204e-02 |
| static_20260216-180002_34bf39 | 10 | — | — | 5.198e+00 |
| static_20260219-142543_59f80b | 10 | 8.071e-05 | — | 8.071e-05 |
| static_20260219-151858_795aad | 10 | 3.462e-04 | — | 3.462e-04 |
| static_20260219-161135_fbae34 | 10 | 5.471e-04 | — | 5.471e-04 |
| static_20260219-170414_6064a1 | 10 | — | — | 1.346e-02 |
| static_20260219-175652_6ada61 | 10 | — | — | 2.290e-02 |
| static_20260219-184928_feabe2 | 10 | — | — | 2.642e+00 |

**Hinweise zum Register:** Für fortgesetzte Runs meint „Ep. erfasst“ die Anzahl der protokollierten Epochen im jeweiligen Ordner, nicht den maximalen Epochenindex; die Warmstart-Ordner enthalten zehn Zeilen für Epochen 21–30. Ohne CSV sind weder Konvergenz noch Divergenz nachgewiesen. Die zehn leeren Runs sind zwei frühe Initialisierungen, sechs Starts vom 10.03. und zwei Soft-Physics-Fehlstarts.

**Korrektur gegenüber einem rein oberflächlichen Vergleich:** Der für Physics-only protokollierte Wert MSE=0 ist kein Ergebnis. Unterschiedliche „physics loss“-Spalten sind ohne identischen Operator nicht vergleichbar. Auch die Aussage „mehr Physik verschlechtert den MSE“ muss bei gt_past als Unterschied unabhängig gestarteter Optimierungen gelesen werden, nicht als Nachweis eines veränderten mathematischen Optimums.
