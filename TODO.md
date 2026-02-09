# TODO / Bugfixes

## Bugfix #1 – delta_t Inkonsistenz
- [ ] Prüfen, ob `delta_t` korrekt ist (Dataset vs. Loss)
- [ ] Dataloader nutzt `predicted_time * 0.1`
- [ ] Statische HeatEquationLoss verwendet festen Wert
- [ ] Entscheidung: delta_t = 1.0 oder dynamisch aus `t`