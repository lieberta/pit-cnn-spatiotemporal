"""Build the German analysis from audited run CSVs; execute after inventar.py."""
import json,csv,math,re,sys,subprocess
from pathlib import Path
OUT=Path('reports/versuchsanalyse');R=json.loads((OUT/'run_details.json').read_text());doc=[]
def add(x):doc.append(x.strip())
def table(headers,rows):
 add('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in rows))
def f(x):return '—' if x is None else ('%.3e'%x)
def row(suffix,epoch=None):
 rr=R[suffix]['rows'];return next(x for x in rr if x['epoch']==epoch) if epoch else rr[-1]
def comparison(items,include_best=False):
 hs=['Run / Variante','Ep.','λP / λD','Val.-MSE','Val.-Physik','Val.-Gesamt'];rows=[]
 if include_best:hs+=['Min. MSE (Ep.)']
 for item in items:
  suf,label,*ee=item;y=row(suf,ee[0] if ee else None)
  rr=[suf+' · '+label,str(y['epoch']),str(y['lp'])+'/'+str(y['mw']),f(y['vm']),f(y['vp']),f(y['vt'])]
  if include_best:
   a=[x for x in R[suf]['rows'] if x['vm'] is not None];b=min(a,key=lambda x:x['vm']) if a else None;rr.append(f(b['vm'])+' ('+str(b['epoch'])+')' if b else '—')
  rows.append(rr)
 table(hs,rows)
add('''# Mathematische Analyse deines Versuchsverlaufs
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
''')
add('### 2.1 Mathematischer Fahrplan: Zuordnung der Loss-Familien')
table(['Familie / Zeitpunkt','Zeitdifferenz','Laplace / Quelle','Beleg und Grenze'],[
['Früh dynamisch bis 09.02.','(v(t)−u₀)/t','L_unscaled(u₀); rohe Schwelle >1000','Git: 7f9d778, 6a1f0ac; kein Run-Snapshot'],
['Normierung ab 13.02.','Weiterhin (v(t)−u₀)/t','L_unscaled(u₀); s=S/R','fcaa462; normsource-Kommentare'],
['Zwei Vorhersagen ab 20.02.','[v(t)−v(t−0,1)]/0,1','L_unscaled(u₀); Quelle teils normiert, teils zurückgenommen','cd47e19, 13cf5d8; lokale Zwischenstände unsicher'],
['L(output), 23.–25.02.','Übergang 0,1 → 0,001 s','L_unscaled(vₙ); normierte Quelle im Commit','2237a9c; mehrere Änderungen gleichzeitig'],
['V0.2, 26.02.','Zwei Ausgaben, h≈0,001 s','Lh(vₙ); nur Innenraum','Run-Kommentare; kein isolierter V0.2-Commit'],
['V0.3/V0.4 ab 26.02.','Zwei Ausgaben, h=0,001 s','Lh(vₙ₋₁); Quelle aus u₀','1ee33d4; alte Daten oft weiterbenutzt'],
['05.–09.03.','h noch 0,001; Daten teils dt=0,01','Weiter Lh(vₙ₋₁)','5408f5d; Übergang von Loader und Trainer'],
['Ab 09.03.','h aus dataset.dt','Weiter Lh(vₙ₋₁)','fe8e545; Starts am 10.03. ohne Messwerte'],
['Ab 13.03., model_past','Zwei Ausgaben mit dt=0,1','Lh(vₙ₋₁), explizites s','59472c7; Anfangsverankerung entfernt'],
['Ab 13.03., GT-Modi','vₙ−yₙ₋₁ oder yₙ₊₁−vₙ, geteilt durch dt','Lh(yₙ₋₁) bzw. Lh(vₙ), explizites s','59472c7; Modi in Run-CSV belegt'],
['Ab 27.03., Soft-Physics','Zwei Ausgaben','Lh(u₀), kein s','0790762; erfolgreicher Run erst 30.03.'],
['Letzter Git-Stand','τ=t/H, aber vergangene Abfrage τ−dt','Zweiter Netzaufruf zeitlich inkonsistent','4867499 + unveränderter Trainer; historische Betroffenheit nicht vollständig beweisbar']])
add('Diese Zuordnung benennt die rekonstruierbare Formel-Familie. Wo nur Datum oder Name vorliegen, wird daraus kein exakt belegter Runtime-Snapshot gemacht. Der spätere Text trennt innerhalb jeder Familie die tatsächlich gemessenen Vergleichsläufe.')
add('## 3. Datengenerierung: Was sich physikalisch geändert hat')
table(['Stand / Daten','Raumgitter / Geometrie','Zeit','Quelle / Material'],[
['2024, alte Raumdaten','64×32×16; 6,3×3,1×1,5; dx=dy=dz=0,1','dt_sim=0,001; Speicherung jedes 100. Frames → 0,1; H=10 s','α=0,0257; S=100000; T₀-Quellen 1100, Umgebung 20'],
['23.02., new_detailed_heat_sim_f64','Gleiche Geometrie und Gitterzahl','Alle 10001 Frames; dt_store=0,001; H=10 s','Gleiche alte Quelle; float64-Simulation/-Speicherung'],
['26./27.02., überarbeiteter Generator','Quellen weg von x/y-Rändern; z=1:3 statt :2','Parametrisierbares T und DT','Änderung betrifft neu erzeugte Daten; alte Dateien bleiben unverändert'],
['05.–09.03., TRAIN_DATASET_10s_dt0.01','Alte Raumgeometrie; laut Generator','dt=0,01; 1001 Frames; 10 s','Info: R=57343,328; 400 Experimente in späteren Run-Configs'],
['13.03., steel_debug_small','64×32×16; 0,5×0,5×0,15','dt=0,1; H=120 s; heute 1202 Frames','k=45, ρ=7850, cp=470; Q=3e7'],
['23.03., steel_train_medium','Gleiche Stahl-Physik; dx≈0,00794; dy≈0,01613; dz=0,01','dt=0,1; 1200 Targets/Experiment','32 Experimente; 18/20/22/24 Quellen; R=101,25644']])
add('''Die „high-res“-Umstellung im Februar ist in den untersuchten Generatoren vor allem eine **zeitliche** Verfeinerung der gespeicherten Daten um Faktor 100, nicht eine Erhöhung von Nx, Ny, Nz. Im März bleibt die Gitterzahl ebenfalls gleich, aber die physikalische Domäne schrumpft. Damit ändern sich die räumlichen Ableitungskoeffizienten drastisch und anisotrop.

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
''')
add('## 4. 2024 bis Anfang Februar 2026: Sekante über den gesamten Horizont')
add('''Der frühe dynamische Loss verwendet nur eine Vorhersage. Die zeitliche „Ableitung“ ist die mittlere Änderung vom Anfang bis zum Zielzeitpunkt, während der Laplace-Term am Anfangszustand festgehalten wird. Der räumliche Kernel enthält 1 an den sechs Nachbarn und −6 in der Mitte, aber keinen Faktor 1/dx².

```math
r_old(t) = (v(t) − u₀)/t − α L_unscaled(u₀) − s(u₀)
L_unscaled = sum_six_neighbors(u_neighbor) − 6u_center
```

Damit verlangt LP≈0 näherungsweise v(t)=u₀+t[αL_unscaled(u₀)+s(u₀)]: eine lineare Fortschreibung mit eingefrorener Anfangsdiffusion. Das ist keine lokale Ableitung an t. Je länger der Horizont, desto weniger entspricht diese Nebenbedingung der iterierten Simulation. Bei dx=0,1 müsste der Laplace-Term 100-mal so groß sein.

Hinzu kommt die Quellerkennung `input > 1000` auf normalisierten Daten. Für Werte im üblichen Bereich [0,1] bleibt die Maske leer: Die frühe „Physik“ kann effektiv eine Gleichung ohne Wärmequelle erzwingen. Die Quelle ist dann nicht nur falsch skaliert, sondern verschwindet vollständig.

Der Autograd-Entwurf von 2024 wurde auskommentiert und im Februar 2026 entfernt. Der damalige Versuch differenziert nach einem nachträglich expandierten Zeit-Tensor und verwendet `grad_outputs=ones`. Das ist kein belegter erfolgreicher voxelweiser Zeitableitungsweg. Der Codekommentar beschreibt Probleme, aber ein grundsätzliches mathematisches Verbot, automatische Zeitableitung mit räumlichen Differenzen zu kombinieren, folgt daraus nicht.

Die archivierten Auswertungen von August 2024 zeigen bei zehn Fällen mittlere absolute Temperaturabweichungen: frühes dynamisches Modell 47,33→133,53 von 1→10 s, „latenttimeforrealthistime“ 26,62→13,82, „smalltest“ 29,37→28,33. Unterschiedliche Trainingsbedingungen und keine vollständigen Run-Metadaten erlauben daraus keinen isolierten Architekturvergleich. Quelle: `310ebe5:plots/*/deviations.txt` und `evaluate.py`.
''')
comparison([('d056d1','latent + Physik'),('ca1b44','latent, nur Daten'),('27e545','timefirst + Physik'),('62167a','timefirst, nur Daten')])
table(['Run-Ordner','Seed der CSV-Zeile','Epoche','Val.-Gesamt','Bedeutung'],[
 ['27e545','60553',25,'1,691e-4','MSE + Physik'],['27e545','16912',25,'5,092e-4','MSE + Physik'],['27e545','94993',25,'6,052e-5','MSE + Physik'],
 ['62167a','60553',24,'1,353e-6','nur MSE'],['62167a','16912',24,'5,113e-6','nur MSE'],['62167a','94993',24,'7,362e-6','nur MSE']])
add('''Die frühen Gesamt-Loss-Werte sind nicht mit einem MSE gleichzusetzen; Komponenten fehlen. Die timefirst-Ordner enthalten 55 bzw. 52 CSV-Zeilen, aber nur 25 bzw. 24 unterschiedliche Epochenindizes. Ab Epoche 11 sind jeweils drei Seeds (60553, 16912, 94993) unter derselben Run-ID vermischt. Das sind keine eindeutigen 55-/52-Epochen-Lernkurven. Die Übersicht zeigt lediglich den letzten CSV-Eintrag des letzten Epochenindex, hier Seed 94993; die anderen Fortsetzungen stehen unten. Welcher Zweig dem zuletzt überschriebenen Checkpoint entspricht, ist damit nicht gesichert. Der MSE-only-Run `ca1b44` zeigt aber, dass das Netz grundsätzlich sehr kleine Datenfehler erreichen konnte.

### 4.1 Statische Versuche: zusätzlich vertauschte Argumente
In den historischen `CombinedLoss`-Versionen, auch noch im Snapshot `4a7f6de`, wird `physics_loss(current_input, output)` aufgerufen, obwohl `HeatEquationLoss.forward(model_output, current_input)` die umgekehrte Bedeutung erwartet. Daraus folgt für diese Version:

```math
r_static_actual = (u₀ − v(t))/t − α L_unscaled(v(t)) − s(v(t))
```

Neben der falschen Richtung der Zeitableitung wird die Quelle damit aus der Vorhersage statt aus dem Anfangsfeld abgeleitet. Nach Aktivierung der normierten Quellschwelle kann die Vorhersage die Quellmaske selbst verändern. Das ist ein gewichtiger Störfaktor für die statischen Ergebnisse. Am 13.03. wird die statische Schnittstelle im Rahmen der expliziten Quelle umgebaut; die hier vorhandenen statischen Runs stammen davor. Die exakte lokale Version einzelner früher Läufe bleibt ungesichert.
''')
staticrows=[]
for label,sufs in [('09.02.',['952339','d56387','5f5ad2','4624df','eb9d45','287556']),('16.02. f32',['6f3462','704464','778736','e4eb74','34bf39','087ac2']),('19.02. f64',['6064a1','59f80b','6ada61','795aad','feabe2','fbae34'])]:
 for i in range(0,6,2):
  a,b=sufs[i:i+2];staticrows.append([label,str(R[a]['config']['predicted_time']),a+' / '+b,f(row(a)['vt']),f(row(b)['vm'])])
table(['Serie','Horizont s','Run +Physik / Daten','Gesamt +Physik, Ep.10','MSE nur Daten, Ep.10'],staticrows)
add('''**Ableitbar:** Der starke Anstieg des statischen Physik-Loss nach Aktivierung der Quelle und die problematischen langen Horizonte passen zu einer inkonsistenten Nebenbedingung. Die Tabelle beweist keine Überlegenheit oder Unterlegenheit „physikalischer CNNs“ als Methode. float64 beseitigt die strukturellen Fehler nicht.

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
''')
comparison([('606d45','timefirst, normierte Quelle'),('388723','Wiederholung normierte Quelle'),('8b5b13','latent nur Daten'),('88ed97','latent + normierte Quelle'),('75c372','timefirst nur Daten'),('7051f0','timefirst + Quelle'),('eb07d7','f64 laut Name'),('58287a','timefirst f64 laut Name'),('26b526','zwei Vorhersagen, Quelle'),('8475da','timefirst, zwei Vorhersagen')])
add('''**Ergebnis:** Mehrere Physik-Runs bleiben bei etwa 0,023 Gesamt-Loss, während Daten-Baselines etwa 4e-6 MSE erreichen. Die neue Zeitableitung und höhere Präzision reichen allein nicht. Mangels Komponenten in diesen frühen Runs lässt sich das Plateau nicht quantitativ in Daten- und PDE-Fehler zerlegen. Die float-Angaben sind teilweise nur Run-Namen/-Kommentare, nicht aufgezeichnete Runtime-Dtypes.

### 5.1 Sonderzweig: separates PINN
Das PINN nutzt automatische erste und zweite Ableitungen eines MLP über [x,y,z,t]. Im damaligen Dataloader werden alle Koordinaten auf [0,1] skaliert, während `heat_residual` dieselbe α vor die ungewichtete Summe der Koordinatenableitungen setzt. Mit ξ=x/Lx und τ=t/H wäre dagegen erforderlich:

```math
(1/H) u_τ − α[(1/Lx²)u_ξξ + (1/Ly²)u_ηη + (1/Lz²)u_ζζ] − s = 0
```

Zusätzlich bekommt das damalige MLP weder das jeweilige Anfangsfeld noch ein Experimentkennzeichen als Eingang. Mehrere zufällige Quellanordnungen können am selben [x,y,z,t] unterschiedliche Temperaturen verlangen; eine einzige Funktion dieser vier Koordinaten kann sie nicht allgemein unterscheiden. Der Quellterm wird zwar im Loss verwendet, macht die Modellvorhersage aber nicht selbst experimentabhängig. Ein separates starkes Anfangs-/Randbedingungs-Loss ist im damaligen Trainingsskript nicht implementiert. Das ist eine plausible strukturelle Erklärung, keine nachträglich isolierte Ablation.
''')
pinn=next(v for v in R.values() if v['id'].startswith('pinn'))
table(['PINN-Epoche','Val.-Daten','Val.-PDE','Val.-Gesamt'],[[x['epoch'],f(x['vm']),f(x['vp']),f(x['vt'])] for x in [pinn['rows'][0],pinn['rows'][-1]]])
add('''Der Gesamt-Loss sinkt nur um rund 0,76 %. Das belegt einen stagnierenden konkreten PINN-Versuch, nicht die generelle Untauglichkeit von PINNs. Quellen: `13cf5d8:dataset.py`, `models/pinn.py`, `training/train_pinn.py`; Run `pinn_20260220-154004`.

## 6. 20.–25. Februar: Quellen-Rücknahme, Laplace am Output, kleinere Schritte
`13cf5d8` setzt die Quellerkennung vorübergehend wieder auf die rohe Schwelle 1000 zurück. Auf normalisiertem u₀ wird die Quelle dadurch typischerweise null. Das erklärt, warum „OLDSOURCE“-Runs plötzlich sehr kleine Residuen zeigen können: Sie lösen eine leichtere, andere Gleichung. Nicht als erfolgreiche Behebung des ursprünglichen Quellenproblems interpretieren.

Danach wird die Diffusion von u₀ auf v(t) verschoben. Mit `2237a9c` sind im committeden Stand gleichzeitig normierte Quelle, Schwelle 500, fein gespeicherte Daten und h=0,001 enthalten:

```math
r_current = [vₙ−vₙ₋₁]/h − α L_unscaled(vₙ) − s(u₀)
```

Mit einem korrekt skalierten L wäre das eine rückwärts-/implizit-Euler-artige Nebenbedingung. Die Daten entstehen aber explizit aus dem vorherigen Zustand. Der Wechsel von L(u₀) zu L(vₙ) behebt das eingefrorene Anfangsfeld, trifft aber noch nicht exakt den Simulator. Zudem fehlen hier weiterhin die räumlichen Skalierungsfaktoren und die Trennung von Rand- und inneren Zellen.
''')
comparison([('7d625b','OLDSOURCE, f64 im Namen'),('96df19','OLDSOURCE latent'),('3523f7','latent nur Daten'),('54d367','früher timefirst-Lauf'),('c8f929','timefirst nur Daten'),('99922b','L(output)'),('46d966','feine Daten, final_straw'),('17af3a','timefirst, final_straw'),('d6c999','feine Daten, nur Daten')])
add('''**Ableitbar:** Niedrige Residuen der OLDSOURCE-Phase sind durch die abgeschaltete Quelle konfundiert. Die hohen Werte um 0,107 bei den fein aufgelösten Daten sind ebenfalls kein reiner Effekt „kleineres dt ist schlechter“: Quelle, Laplace-Zustand, Datenauflösung und teilweise Normalisierung wechselten gleichzeitig. Der Lauf `99922b` beginnt vor dem zugehörigen Commit und dokumentiert L(output) im Kommentar; sein vollständiger lokaler Zustand lässt sich nicht exakt rekonstruieren.

Bei h:0,1→0,001 wächst die Empfindlichkeit gegenüber einer festen Differenz der beiden Vorhersagefehler um Faktor 100; ihr quadratischer Beitrag kann um Faktor 10000 wachsen. Das gilt bei festgehaltenem Fehlerunterschied, nicht zwangsläufig für korrelierte glatte Modellfehler. Bei float32 wird zusätzlich die Subtraktion fast gleicher Ausgaben empfindlicher. Diese Skalierung erklärt, warum ein unverändertes λP keinen unveränderten Optimierungsdruck garantiert.

## 7. 26. Februar–4. März: V0.2/V0.3, konsistenteres Euler-Residuum
V0.2 korrigiert den räumlichen Operator von L_unscaled zu Lh und wertet das Residuum nur auf inneren Zellen aus. Der z=0-Rand wird von einem über Batch und Fläche broadcasteten Skalar auf die vollständige Eingangsfläche umgestellt. V0.3 verwendet schließlich den Laplace-Term der vorherigen Modellvorhersage:

```math
r_V03 = P{ [vₙ−vₙ₋₁]/h − α Lh(vₙ₋₁) − s(u₀) }
```

Bei korrekten h, Quelle und Datenrandbedingungen entspricht das einem expliziten Euler-Schritt. Der Code-Commit `1ee33d4` fasst die Änderungen zusammen; die V0.2/V0.3-Kommentare der Runs erlauben eine feinere zeitliche Zuordnung, aber keinen lückenlosen Snapshotnachweis.
''')
comparison([('1b9c31','V0.2 latent'),('f539ad','V0.2 timefirst'),('52aa35','V0.2 latenttime1'),('94b47b','V0.3 a=0,1'),('1123ba','V0.3 a=1'),('056946','V0.3 timefirst'),('b6153e','V0.3 latenttime1'),('a91705','V0.3 lr=1e-4')])
add('''**Ergebnis:** Bei a=0,1 ist der Datenfehler kleiner als bei a=1 (3,26e-5 gegenüber 6,15e-5), der Physikfehler aber größer (1,40e-4 gegenüber 3,02e-5). Das zeigt den Zielkonflikt direkt innerhalb der Komponenten. Beide Runs nennen denselben Seed 897496; der Code setzt den Seed jedoch zentral und speichert die Splits nicht, sodass identische Seed-Zahlen allein keinen identischen Split oder Netzstart beweisen. Andere Architektur-/Lernratenvergleiche verwenden unterschiedliche Seeds.

Die Quellenverlagerung im Generator beseitigt das Problem nur für neu erzeugte Daten. Zahlreiche V0.3/V0.4-Runs verwenden laut Config weiterhin `new_detailed_heat_sim_f64`; die vorhandenen Dateien stammen vom 23.02. und besitzen die problematischen heißen Anfangsränder. Die Verbesserung des Loss-Codes ist daher nicht gleichbedeutend mit einem vollständig konsistenten Trainingsproblem.
''')
comparison([('d1a417','6 Experimente, beide Losses'),('1b1203','6 Experimente, nur Daten'),('18f41e','30 Exp., lr=1e-4'),('6b06f9','30 Exp., lr=1e-4 Wiederholung'),('ac802b','30 Exp., latenttime1'),('9f831c','30 Exp., timefirst nur Daten'),('260835','latenttime1 nur Physik'),('4054e1','29 Exp., nur Daten'),('824698','29 Exp., nur Physik'),('45ed6b','29 Exp., timefirst nur Physik')])
add('''**Ableitbar:** Im Sechs-Experimente-Vergleich mit je 50 Epochen ist der MSE des kombinierten Laufs um Faktor 1,79 größer als der Daten-Baseline. Kleinere Lernrate und 50 Epochen beseitigen den Konflikt nicht. Die kleinen Physics-only-Werte sind keine gemessene Temperaturgenauigkeit. `824698` und `45ed6b` wurden nach 89 bzw. 84 vollständigen Epochen abgebrochen; Logs belegen Abbruch, nicht mathematische Divergenz.

### 7.1 Ein leicht zu übersehender Phasenwechsel
Run `61a255` heißt DATADRIVENONLY, enthält aber 50 Epochen nur MSE und danach 50 Epochen nur Physik. Die globalen Startgewichte der Config sind kein verlässlicher Ersatz für den Schedule bzw. die CSV.
''')
comparison([('61a255','Datenphase',50),('61a255','erste Physik-Epoche',51),('61a255','Ende Physikphase',100)])
add('''Das Residuum fällt von 3,48e-6 auf 2,40e-7 während der Physikphase. Ob die Vorhersage dabei besser oder schlechter wird, wurde nicht gemessen: Ab Epoche 51 ist der MSE deaktiviert. Der Übergang von 8,04e-6 „Loss“ zu 2,40e-7 darf deshalb nicht als Genauigkeitsgewinn um Faktor 33 gelesen werden.

## 8. 5.–10. März: dt aus Metadaten und der erste Zeitschritt
Mit `0a2e3e6` wird bei t=h der vergangene Zustand explizit durch u₀ ersetzt. Das verankert den ersten Euler-Schritt an der bekannten Anfangsbedingung, statt auch fθ(u₀,0) frei lernen zu lassen. Diese Behandlung wird beim großen Refactor `59472c7` wieder entfernt; seitdem nutzt `model_past` auch am Anfang eine zweite freie Vorhersage.

`5408f5d` lässt den Dataloader die Zeitbasis aus Metadaten lesen, aber im Trainer bleibt h noch fest 0,001 s. `fe8e545` vom 09.03. beseitigt den Fallback und übernimmt `dataset.dt` auch im Trainer. Damit entsteht eine wichtige Übergangsphase auf dem dt=0,01-Datensatz: Je nach lokalem Metadatenstand konnte die Eingangszeit um Faktor 10 falsch zugeordnet sein oder der Loss zwischen nicht gespeicherten Zwischenzeiten mit h=0,001 differenzieren. Letzteres ist für ein kontinuierliches Modell nicht an sich verboten, ist aber kein exakter Ein-Schritt-Test gegen den Euler-Datenoperator mit dt=0,01.
''')
comparison([('022c50','40 Exp., MSE-only'),('152003','40 Exp., Physics-only')])
add('''Die Runs vom 05.03. liegen vor der committeden dt-Korrektur. Ihre kleinen Endwerte belegen daher noch keinen Erfolg der endgültigen Zeitbehandlung. Die sechs Runs vom 10.03. (`072061`, `88b108`, `b5847c`, `501759`, `a211c1`, `22cc8c`) haben keine protokollierte Epoche. Logs der zugehörigen Jobs enthalten `NameError: _require_zarr is not defined`. Frühe Jobs vom 05.03. scheitern außerdem an einem Index außerhalb der 1001 gespeicherten Frames. Das sind Pipelinefehler, keine Resultate zur Qualität des Physics-Loss.

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
''')
rat=[]
for suf in ['ad89e8','e2cca7','d013c6','c950fe']:
 y=row(suf);rat.append([suf,y['epoch'],f(y['vm']),f(y['vp']),'%.8f'%(y['vp']/y['vm'])])
table(['Run','Ep.','LD','LP','Gemessen LP/LD'],rat)
add('''Die Übereinstimmung bis auf float32-/Mittelungsfehler ist ein direkter numerischer Nachweis dieser Interpretation, nicht bloß eine theoretische Vermutung.

### 9.3 gt_future: Fehler nach einem expliziten Vorwärtsschritt
```math
r_gt_future = P{ (yₙ₊₁−vₙ)/dt − α Lh vₙ − s }
⇒ r_gt_future = −P A eₙ/dt
⇒ LP = mean_inner((A eₙ)²)/dt²
```

Hier muss die aktuelle Vorhersage nach einem Euler-Schritt zur zukünftigen Ground Truth passen. Das ist ein räumlich gefilterter supervidierter Fehler. Er koppelt Nachbarfehler und unterscheidet sich damit tatsächlich vom gewöhnlichen MSE, nutzt aber weiterhin Simulationslabels. Für einen Laplace-Eigenmodus Lh φ=−μ φ wird dessen Fehler mit (1−dt α μ)²/dt² gewichtet. Unter der hier gewählten stabilen Diskretisierung dämpft A insbesondere feinräumige Fehlermoden; „gt_future bestraft automatisch alle hohen Frequenzen stärker“ wäre falsch. Bei den Stahlparametern liegt die konservative Untergrenze des A-Faktors bei rund 0,855, sodass der Filter in einem Schritt relativ moderat wirkt.

### 9.4 Ground-Truth-Test der Diskretisierung
Je drei vorhandene Experimente aus Debug und Medium wurden an n=1,100,600,1200 geprüft (24 Zustandsdreiergruppen). Mit float64 und den gespeicherten Metadaten liegen die mittleren quadratischen Residuen von gt_past und gt_future zwischen etwa 1,08e-32 und 2,96e-32. Die aktuelle Datenrepräsentation ist damit in dieser Stichprobe exakt zum expliziten Euler-Residual konsistent. Das prüft die Dateien von heute, nicht rückwirkend einen unbekannten Trainingssnapshot. Details stehen in `ground_truth_residual_checks.json`.
''')
add('### 9.5 Ergebnisse auf steel_debug_small')
comparison([('908735','nur Daten'),('3f65f8','gt_future'),('ad89e8','gt_past'),('477782','model_past')],True)
add('''Alle vier Runs: sechs Experimente, 7200 Samples, 20 Epochen, PITCNN_dynamic, unterschiedliche Seeds. gt_future erreicht am Ende rund 44 % weniger MSE als die Daten-Baseline, gt_past etwa das 2,06-Fache und model_past das 15,35-Fache. **Ableitbar:** gt_future war in diesem kleinen Versuch vielversprechend. Wegen unterschiedlicher Splits/Initialisierungen und der algebraischen gt_past-Identität ist das kein kausaler Nachweis eines generellen Physikvorteils. Ein kleines LP bei model_past kann mit großem Datenfehler zusammengehen, genau wie die Fehlerfortpflanzungsformel erwarten lässt.

## 10. 23.–27. März: Größerer Stahl-Datensatz und Gewichte
Alle vier ursprünglichen Medium-Runs nutzen PITCNN_dynamic, 32 Experimente und 38400 Samples. Dateinamen mit „timefirst“ ändern diese aufgezeichnete Modellklasse nicht. Die für 100 Epochen angesetzten Jobs wurden laut Logs nach 15 vollständigen Epochen während der nächsten Epoche durch das Zeitlimit beendet. Der Datenrun wurde später fortgesetzt.

### 10.1 Vergleich bei derselben vollständigen Epoche 14
''')
comparison([('7e2ea6','nur Daten',14),('e6895c','gt_future λP=1',14),('e2cca7','gt_past λP=1',14),('7a6070','model_past λP=1',14),('24c1fb','gt_future λP=0,01',14),('d013c6','gt_past λP=0,01',14),('6275b7','model_past λP=0,01',14)])
add('''Bei λP=1 ist der MSE von gt_future etwa 1,70-mal, von gt_past 2,47-mal und von model_past 131,77-mal so groß wie der Baseline-MSE. Der Vorteil von gt_future aus dem kleinen Debug-Versuch reproduziert sich somit in diesen unabhängigen Medium-Runs nicht.

Zur Größenordnung: Mit R=101,25644 entspricht der Baseline-MSE bei Epoche 14 einem globalen RMSE von etwa 0,150 K, gt_future mit λP=1 etwa 0,195 K und model_past etwa 1,719 K. Das ist R·sqrt(MSE), über das ganze Feld einschließlich der fehlerfreien Ränder; lokale Fehler nahe Quellen können deutlich größer sein. Einzelne Batch-Mittel werden im Trainer ungewichtet gemittelt, daher ist das eine Umrechnung der protokollierten Metrik und keine neu berechnete voxelgewichtete Testauswertung.

Die Reduktion von λP auf 0,01 senkt bei model_past den MSE gegenüber dem λP=1-Lauf um etwa Faktor 17,0, lässt aber das Residuum deutlich größer werden. Das passt zu einer zu stark oder widersprüchlich wirkenden Kopplung. Bei gt_past verändert die Gewichtung unter den genannten Voraussetzungen im Wesentlichen nur die globale Skalierung; eine Interpretation als neuer physikalischer Kompromiss wäre irreführend.

Der gt_future-Run `24c1fb` endet ungünstig, hat aber zwischenzeitlich einen MSE von 2,386e-6 erreicht. Die Endpunkt-Tabelle allein verdeckt solche Schwankungen. Minima der einzelnen Kurven stehen im Inventar; sie werden nicht ohne Kennzeichnung gegen Endpunkte anderer Läufe verglichen. Alle hier gestarteten Medium-Vergleichsruns haben verschiedene Seeds.

### 10.2 MSE-Warmstart und Wechsel zu Physik
Der bestehende Datenrun wird auf 20 Epochen fortgesetzt. Zwei neue Ordner übernehmen daraus Modell, Optimizer-Zustand und Epochenzähler und trainieren Epochen 21–30 mit λD=λP=1.
''')
comparison([('7e2ea6','Daten-Basis',20),('c950fe','gt_past nach Wechsel',21),('c950fe','gt_past Ende',30),('43631b','gt_future nach Wechsel',21),('43631b','gt_future Ende',30)])
add('''Unmittelbar nach dem Wechsel liegt der MSE wesentlich höher und erholt sich anschließend. Am Ende ist er noch 1,62-mal (gt_past) bzw. 2,54-mal (gt_future) so groß wie die protokollierte Baseline bei Epoche 20. Diese Zahlen zeigen keinen Genauigkeitsgewinn durch das Nachtraining.

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
''')
comparison([('db1749','Soft + Daten'),('eb6e6e','gt_future nur Physik'),('6f7386','gt_past nur Physik')])
add('''Soft-Physics erreicht nach 14 Epochen 2,993e-4 MSE und liefert hier keinen Datenvorteil. Das Ergebnis testet allerdings eine geänderte Gleichung und vermutlich die problematische zweite Zeitabfrage, nicht bloß „weichere Physik“. Die beiden Physics-only-Jobs enden laut Slurm-Zeitlimit nach 13 vollständigen Epochen; eine 14. vollständige Epoche ist nicht protokolliert.

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
''')
for group in sorted(set(x['phase'] for x in R.values())):
 add('### '+group)
 rows=[]
 for x in R.values():
  if x['phase']!=group:continue
  ys=x['rows'];y=ys[-1] if ys else {};rows.append([x['id'],len(ys),f(y.get('vm')),f(y.get('vp')),f(y.get('vt'))])
 table(['Run-ID','Ep. erfasst','MSE zuletzt','Physik zuletzt','Gesamt zuletzt'],rows)
add('''**Hinweise zum Register:** Für fortgesetzte Runs meint „Ep. erfasst“ die Anzahl der protokollierten Epochen im jeweiligen Ordner, nicht den maximalen Epochenindex; die Warmstart-Ordner enthalten zehn Zeilen für Epochen 21–30. Ohne CSV sind weder Konvergenz noch Divergenz nachgewiesen. Die zehn leeren Runs sind zwei frühe Initialisierungen, sechs Starts vom 10.03. und zwei Soft-Physics-Fehlstarts.

**Korrektur gegenüber einem rein oberflächlichen Vergleich:** Der für Physics-only protokollierte Wert MSE=0 ist kein Ergebnis. Unterschiedliche „physics loss“-Spalten sind ohne identischen Operator nicht vergleichbar. Auch die Aussage „mehr Physik verschlechtert den MSE“ muss bei gt_past als Unterschied unabhängig gestarteter Optimierungen gelesen werden, nicht als Nachweis eines veränderten mathematischen Optimums.
''')
text='\n\n'.join(doc)+'\n';(OUT/'Detaillierte_Versuchsanalyse.md').write_text(text,encoding='utf-8');print('Markdown characters',len(text))
