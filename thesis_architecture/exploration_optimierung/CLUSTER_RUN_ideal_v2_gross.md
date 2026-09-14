# Cluster-Lauf `ideal_v2_gross_cluster` — ergaenzende Optuna-Suche

**Zweck dieser Datei:** vollstaendiger Kontext, um diesen Cluster-Lauf von
einem anderen Rechner aus zu starten, ohne die vorangegangene Unterhaltung
zu kennen. Falls eine Claude-Code-Sitzung das hier liest: die Schritte unten
ausfuehren, aber die Cluster-Sicherheitsregeln aus der Projekt-`CLAUDE.md`
gelten unveraendert — insbesondere vor jedem ssh/srun/sbatch-Kommando noch
einmal explizit nachfragen und den genauen Befehl zeigen, auch wenn er hier
schon steht. Nie `srun run_job_optuna_gross.bash` direkt starten, nur
`sbatch run_job_optuna_gross.bash`.

## Warum dieser Lauf existiert

Lokal laeuft (Stand dieser Datei) die Optuna-Studie `ideal_v2`: 12-dimensionaler
Suchraum (`--space ideal`) ueber die Akquise-Einstellungen der
Laengeneinheit-Mission (`exploration_optimierung/mission.py::LaengenMission`),
mit dem Ziel, die im Projekt schon veroeffentlichte, aber nur grob von Hand
gesuchte Einstellung (`exploration_optimierung/results/bestwerte_beide.json`,
J=0.2581) zu verbessern und auf mehr Seeds abzusichern.

**Stand beim Schreiben dieser Datei:** 44 abgeschlossene, 96 geprunte Versuche,
bester Wert J=0.2585 (Versuch #115, `phi_model=eid`, `phi_mode=quantile`).
TPE ist dabei frueh auf `eid`+`quantile` eingerastet (80 von 126 Versuchen
insgesamt) — moeglich, dass andere der sieben Phi-Modelle
(`ucb,eid,mass,niveau,stretch,ei,mi`, aus `common/acquisition.py::PHI_MODELLE`)
mit mehr Erkundung besser abschneiden wuerden.

**Dieser Cluster-Lauf ist deshalb keine Kopie, sondern eine Ergaenzung:**
Raum `gross` (16 statt 12 Dimensionen — zusaetzlich `gp_variance`,
`flow_steps`, `gp_res`, `max_obs`, `visit_bandwidth`) und `--random_startup 150`
statt 50, damit alle sieben Modelle vor der TPE-Einengung ausreichend
gesehen werden. Die zusaetzlichen 16 CPU-Kerne des Cluster-Jobs machen
SVGD-Verfeinerung guenstiger, das Budget fuer die breitere Suche ist also da.

Live zusammenfuehren geht nicht — SQLite (`study.db`) laesst sich nicht
zwischen zwei Rechnern teilen. Was sich teilen laesst, ist der
`cache/`-Ordner: jede Datei darin ist nach dem Inhalts-Hash aus
Konfiguration+Seed benannt, ein Treffer ist unabhaengig davon gueltig, wer
ihn berechnet hat. Beide Studien bleiben als eigene `study.db` bestehen und
werden erst bei der Auswertung gemeinsam gelesen (letzter Abschnitt unten).

## Was gebaut wurde (bereits im Repo, siehe `git log`)

- `exploration_optimierung/optuna_search.py` — die Optuna-Studie selbst.
  TPE (`multivariate=True, group=True`) + `HyperbandPruner` (Sprossen
  4 -> 12 -> 36 Runden, Faktor 3; Ressourcenachse = durchlaufende Rundenzahl
  ueber alle Seeds, nicht nur den ersten — siehe Docstring im Modul fuer die
  Begruendung). Cache-Key umfasst Konfiguration + Kontext (n_max, n_shapes,
  Checkpoint, Raum) + Seed, absichtlich getrennt vom Cache-Schema in
  `optimize.py` (das kennt `max_obs`/`phi_mode` nicht im Key).
- `exploration_optimierung/mission.py` — `LaengenMission.__init__` um
  `gp_lengthscale`/`gp_variance` erweitert (Voreinstellung 0.08/1.0 = altes
  Verhalten, rueckwaertskompatibel). Vorher war die GP-Korrelationslaenge
  fest verdrahtet.
- `run_job_optuna_gross.bash` (Repo-Wurzel von `thesis_architecture/`) —
  das SLURM-Skript fuer diesen Lauf, nach dem Muster von
  `run_job_erkundung_opt.bash`.

## Schritt 1 — Repo-Stand auf den Cluster bringen

Datei-Synchronisation per rsync ist laut Projektregeln automatisch erlaubt,
keine Rueckfrage noetig:

```bash
rsync -av --progress \
  thesis_architecture/exploration_optimierung/ \
  stud_schonfeld@mn.ias.informatik.tu-darmstadt.de:Master_thesis/thesis_architecture/exploration_optimierung/

rsync -av --progress \
  thesis_architecture/run_job_optuna_gross.bash \
  stud_schonfeld@mn.ias.informatik.tu-darmstadt.de:Master_thesis/thesis_architecture/
```

Falls der Checkpoint noch nicht auf dem Cluster liegt (er sollte, wird aber
von keinem der beiden Studien-Laeufe veraendert, ein erneuter Sync schadet
nicht):

```bash
rsync -av --progress \
  transfer/netz2d_startpunkt.pt \
  stud_schonfeld@mn.ias.informatik.tu-darmstadt.de:Master_thesis/transfer/
```

## Schritt 2 — Optuna in der `thesis`-conda-Umgebung sicherstellen

Das ist ein Cluster-Kommando (ssh) und braucht laut Projektregeln eine
explizite Aufforderung im jeweiligen Moment, auch wenn es hier schon
vorformuliert ist:

```bash
ssh stud_schonfeld@mn.ias.informatik.tu-darmstadt.de \
  "source ~/miniconda3/etc/profile.d/conda.sh && conda activate thesis && python -c 'import optuna' || pip install optuna"
```

## Schritt 3 — Job einreichen

Ebenfalls ein Cluster-Kommando, vorher bestaetigen lassen:

```bash
ssh stud_schonfeld@mn.ias.informatik.tu-darmstadt.de \
  "cd Master_thesis/thesis_architecture && mkdir -p logs && sbatch run_job_optuna_gross.bash"
```

Der Suchraum, Seeds, Pruner-Sprossen etc. stehen fest im Skript
`run_job_optuna_gross.bash` — dort aendern, falls etwas anders laufen soll,
nicht per zusaetzlichen CLI-Flags beim Submit.

## Ueberwachen

```bash
ssh stud_schonfeld@mn.ias.informatik.tu-darmstadt.de "squeue -u stud_schonfeld"
ssh stud_schonfeld@mn.ias.informatik.tu-darmstadt.de \
  "tail -f Master_thesis/thesis_architecture/logs/optuna_gross-<JOBID>.out"
```

## Nach dem 24h-Limit fortsetzen

```bash
ssh stud_schonfeld@mn.ias.informatik.tu-darmstadt.de \
  "cd Master_thesis/thesis_architecture && sbatch --dependency=afterany:<JOBID> run_job_optuna_gross.bash"
```

Derselbe `--study ideal_v2_gross_cluster`-Name im Skript laedt die
bestehende `study.db` automatisch weiter (siehe `_clear_stale` in
`optuna_search.py` — ein verwaister Versuch aus dem harten Zeitlimit wird
beim naechsten Start als FAIL markiert, alles Fertige bleibt erhalten).

## Ergebnisse zusammenfuehren (nach Abschluss, wieder lokal)

```bash
rsync -av \
  stud_schonfeld@mn.ias.informatik.tu-darmstadt.de:Master_thesis/thesis_architecture/exploration_optimierung/results/optuna/cache/ \
  thesis_architecture/exploration_optimierung/results/optuna/cache/
```

Danach beide Studien getrennt lassen (`ideal_v2` lokal, `ideal_v2_gross_cluster`
in der zurueckgeholten `study.db` — die liegt unter
`results/optuna/study.db` auf dem Cluster und muesste separat zurueckgeholt
werden, z. B. nach `results/optuna_cluster/study.db` lokal, um sie nicht mit
der lokalen zu ueberschreiben), aber fuer eine gemeinsame Rangliste
zusammen einlesen:

```python
import optuna
from exploration_optimierung import objective as OBJ

local = optuna.load_study(study_name='ideal_v2', storage='sqlite:///.../study.db')
cluster = optuna.load_study(study_name='ideal_v2_gross_cluster', storage='sqlite:///.../study_cluster.db')

alle = [t for t in local.trials + cluster.trials if t.value is not None]
alle.sort(key=lambda t: t.value)
for t in alle[:20]:
    print(t.value, t.params)
```

`--report` (siehe `optuna_search.py`) schreibt zusaetzlich `trials.csv` und
`best.json` je Studie unter `results/optuna/`.
