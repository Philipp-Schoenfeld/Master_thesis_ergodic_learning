# ERGLOSS-Gewichtsablation — Herkunft der Bilder

Geholt am 17.08.2026 vom Cluster aus
`~/Master_thesis/thesis_architecture/Trajectory_data_generator/`.

Gemeinsame Konfiguration aller Läufe: `nxi25 D384 N256 C15 flip0.0 K8 tp2`,
`--epochs 500`, `--mini_batch 256`, SLURM-Limit 18 h.

| Lokale Datei | Gewicht | Epoche | Status | Original auf dem Cluster |
|---|---|---|---|---|
| `ergloss_w02_ep0492_emergency.png` | 2 | 492 | SIGTERM bei 18 h | `flow_matching_particle_ergodic_date_08_15_13h17min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w2-K8-tp2_emergency_ep0492.png` |
| `ergloss_w05_ep0500_lauf-08-11.png` | 5 | 500 | vollständig, **anderer Lauf** | `flow_matching_particle_ergodic_date_08_11_17h16min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w5-K8-tp2_ep0500.png` |
| `ergloss_w15_ep0486_emergency.png` | 15 | 486 | SIGTERM bei 18 h | `flow_matching_particle_ergodic_date_08_15_13h23min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w15-K8-tp2_emergency_ep0486.png` |
| `ergloss_w25_ep0485_emergency.png` | 25 | 485 | SIGTERM bei 18 h | `flow_matching_particle_ergodic_date_08_15_13h44min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w25-K8-tp2_emergency_ep0485.png` |
| `ergloss_w50_ep0486_emergency.png` | 50 | 486 | SIGTERM bei 18 h | `flow_matching_particle_ergodic_date_08_16_07h17min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w50-K8-tp2_emergency_ep0486.png` |

## Zwei Einschränkungen

**1. Vier von fünf Läufen haben das 18-h-Limit gerissen.** Sie wurden bei Epoche
485–492 vom SIGTERM-Handler gestoppt statt die 500 Epochen zu erreichen. Für den
Vergleich untereinander ist das unkritisch (alle vier liegen innerhalb von sieben
Epochen), gegenüber dem w=5-Lauf mit vollen 500 Epochen besteht aber eine kleine
Asymmetrie. Für einen Wiederholungslauf reichen 20 h Limit oder ca. 480 Epochen.

**2. Der Job für w=5 aus dieser Serie hat nie trainiert.** Der Resume-Glob in
`run_job_particles_ergloss_w5.bash` passte auf den bereits abgeschlossenen Lauf
vom 11.08. (identische Konfiguration: C15, w5, K8, tp2, 500 Epochen). Der Job
hat diesen Checkpoint geladen, das Training als beendet erkannt, die
Abschlussvisualisierungen geschrieben und nach rund fünf Minuten beendet — auf
dem Cluster liegen unter `date_08_15_13h17min_..._ERGLOSS-w5-...` nur
`_train.png` und `_holdout.png`, keine Epochenbilder.

Das hier abgelegte w=5-Bild stammt deshalb aus dem Lauf vom 11.08. Die
Hyperparameter sind identisch, es ist also ein gültiger Datenpunkt für die
Ablation — nur nicht aus derselben Charge.

## Nicht geholt

Auf dem Cluster liegen insgesamt rund 558 MB an Epochenbildern dieser Läufe
(je Lauf 25 Zwischenstände alle 20 Epochen). Geholt wurde nur der Endzustand
je Gewicht, um unter der 25-MB-Absprachegrenze zu bleiben.
