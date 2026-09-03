# exploration_optimierung

Suche nach der besten Betriebseinstellung der **Laengeneinheit-Mission**:
ergodische Bahn planen, genau *eine* Laengeneinheit fahren, Glauben
fortschreiben, neu planen — ohne jedes Vorwissen ueber die Zielverteilung.

Eine Laengeneinheit ist die Diagonale der Zieldomaene, auf `[0,1]^2` also
`sqrt(2) = 1,4142`.

## Was gesucht wird

| Zieldichte-Modell | freier Parameter | Bedeutung |
|---|---|---|
| `ucb`    | `kappa` | Gewicht der Unsicherheit gegen das bereits Gefundene |
| `eid`    | `kappa` | Mischung aus aufgedeckter Dichte und Informationsdichte |
| `mass`   | `w`     | Anteil der Zielmasse, der auf Erkundung entfaellt |
| `niveau` | `tau`   | Schwelle, ab der ein Ort zum Traeger der Form zaehlt |

Dazu die **Zahl der Ausfuehrungen** `n` und — im zweiten Durchlauf — die
**Zahl der SVGD-Iterationen**.

`n` kostet nichts: ein Rollout ueber `--n_max` Runden enthaelt alle kuerzeren
Missionen als Praefix und liefert die Guete fuer jedes `n` auf einmal. **Jede
Zahl von Ausfuehrungen von 1 bis `n_max` ist damit vollstaendig vermessen**, nicht
gestichprobt.

## Wie weit die Suche reicht

Zwei Strategien, `--search`:

* **`abstieg`** (Voreinstellung) — Grobraster ueber den Parameter plus lokale
  Verfeinerung; bei SVGD ein Koordinatenabstieg. Findet die Einstellung schnell,
  misst die Iterationszahl aber nur beim *einen* besten Parameterwert. Die
  Wirkungsmatrix bleibt luecken haft.
* **`voll`** — vollstaendiges Kreuzprodukt Parameter x SVGD-Iterationen. Rund
  doppelt so teuer, liefert dafuer jede Zelle. Nur damit ist die Waermekarte
  vollstaendig und die Frage beantwortbar, ob sich das Parameteroptimum unter
  mehr Verfeinerung verschiebt. In `--mode beide` ist die Studie ohne SVGD
  kostenlos enthalten: die Spalte `svgd_iters = 0` *ist* der ungefuehrte Zweig.

Die Dichte des Parameterrasters steuert `--param_points` (ohne Angabe das
handgesetzte Raster mit 7 Punkten), die Zahl der Wiederholungen `--seeds`.

Aufwand auf einer RTX 2070 SUPER, 25 Formen, `n_max = 12`, alle vier Modelle:

| Suche | Parameterpunkte | Seeds | Auswertungen | Dauer |
|---|---|---|---|---|
| `abstieg` |  7 | 1 | 100 |  3 h 55 |
| `voll`    |  7 | 1 | 168 |  8 h 46 |
| `voll`    | 12 | 1 | 288 | 15 h 19 |
| `voll`    | 16 | 1 | 384 | 20 h 13 |
| `voll`    | 16 | 2 | 768 | 42 h 33 |

## Zielfunktion

    J = q  +  lambda_len * n  +  lambda_time * t

* `q` — mittlerer *bezogener* Abdeckungsfehler `cov / cov_blind` ueber alle
  25 Holdout-Formen. 1,0 = so gut wie ohne Bahn, 0,0 = perfekt.
* `n` — Zahl der Ausfuehrungen; weil jede genau eine Laengeneinheit lang ist,
  **ist** das zugleich die gefahrene Strecke.
* `t` — Rechenzeit in Sekunden. `lambda_time` ist ohne SVGD 0.

Die Gewichte sind nachtraeglich aenderbar, ohne etwas neu zu rechnen:
`--reweight` liest die abgelegten Spuren und stellt die Rangfolge neu auf.
Zusaetzlich wird immer die vollstaendige `(n, q)`-Kurve geschrieben.

## Benutzung

```bash
cd thesis_architecture

# Aufwand schaetzen, rechnet nichts
python -m exploration_optimierung.optimize --mode beide --estimate_only

# Der volle Lauf: komplettes Kreuzprodukt, dichtes Raster
python -m exploration_optimierung.optimize --mode beide --search voll --param_points 12

# Schnellvariante (Koordinatenabstieg, findet die Einstellung, nicht die Matrix)
python -m exploration_optimierung.optimize --mode beide

# Abbildungen: Panel ueber alle Formen, Parameterkurven, Abdeckungskurven
python -m exploration_optimierung.plots

# Rangfolge mit anderer Wegstrafe, ohne Neurechnung
python -m exploration_optimierung.optimize --reweight --lambda_len 0.04

# Selbsttest (67 Pruefungen, rund 4 Minuten)
python -m exploration_optimierung.test_smoke
```

## Was abgelegt wird

Alles, jede einzelne Messung. Je Studie (`ohne_svgd`, `mit_svgd`):

| Datei | Koernung | wofuer |
|---|---|---|
| `alle_laeufe_<tag>.csv` | **eine Zeile je (Einstellung, Form, Rundenzahl)** | die Rohtabelle, aus der sich jedes spaetere Diagramm bauen laesst |
| `kurven_<tag>.csv` | je (Einstellung, Rundenzahl), ueber die Formen gemittelt | die `(n, q)`-Kurven |
| `suche_<tag>.csv` | je Einstellung, beste Rundenzahl | die Rangfolge |
| `lauf_<tag>.json` | einmal | die vollstaendige Konfiguration des Laufs |
| `results/cache/*.json` | je Auswertung | Rohspur samt allen Randbedingungen |

`alle_laeufe_*.csv` ist die wichtige: sobald ueber die Formen gemittelt ist,
laesst sich nicht mehr sehen, ob ein Mittelwert aus 25 gleich guten Bahnen
entsteht oder aus 20 guten und 5 gescheiterten. Spalten sind `phi_model`,
`param`, `svgd_iters`, `seed`, `shape`, `n_exec`, `cov`, `cov_norm`,
`erg_truth`, `belief_rmse`, `info_gain`, `path_len`, `n_obs`, `plan_s`,
`svgd_s`. Bei der vollen Suche rund 86 000 Zeilen.

## Abbildungen

| Datei | zeigt |
|---|---|
| `panel_<tag>.png` | die beste Einstellung auf allen 25 Holdout-Formen |
| `waermekarte_<tag>.png` | `J` ueber (Parameter x SVGD-Iterationen), je Modell — wirken die beiden Regler aufeinander? |
| `formen_<tag>.png` | `q` je einzelner Form plus Verlauf ueber `n` — Streuung statt Mittelwert |
| `parameter_<tag>.png` | `J` ueber dem freien Parameter, je Modell |
| `abdeckung_<tag>.png` | `q` und `J` ueber der Zahl der Ausfuehrungen |

## Fortsetzbarkeit

Jede fertige Auswertung liegt als JSON unter `results/cache/`. Ein Neustart
ueberspringt, was schon da ist — ein abgebrochener Lauf kostet hoechstens die
angefangene Auswertung. Der Schluessel enthaelt *alle* Randbedingungen
(`n_max`, `n_shapes`, `debt_weight`, `sensor_radius`, `flow_steps`, ...), ein
Ergebnis wird also nie unter anderen Bedingungen wiederverwendet.

## Dateien

| Datei | Inhalt |
|---|---|
| `mission.py`    | die Mission selbst: Runde planen, eine Laengeneinheit fahren, messen |
| `objective.py`  | was "gut" heisst — `J`, Pareto-Front, beste Rundenzahl je Spur |
| `optimize.py`   | die Suche, Fortschrittsanzeige mit Restzeit, Zwischenspeicher, CSV |
| `plots.py`      | Panel ueber alle Formen, Parameter- und Abdeckungskurven |
| `test_smoke.py` | Selbsttest ohne Suchlauf |

`results/probe_4formen/` enthaelt einen frueheren Probelauf mit 4 Formen und
3 Runden — nur zur Ansicht, nicht die Studie.
