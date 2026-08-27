# -*- coding: utf-8 -*-
r"""
test_dichte_dedup.py
====================
Die Dichtekarten werden je Zielverteilung nur einmal gerechnet und nur einmal
gespeichert, obwohl bis zu neunzehn Datensatzzeilen sie teilen.

Das ist eine Speicher- und Rechenzeitfrage: 20.593 Gitter zu 128x128 float32
sind 1,35 GB, 1.187 Gitter sind 78 MB, und `pdf_on_grid` laeuft ueber JAX
17-mal seltener. Der Test prueft, dass dabei **nichts** an der Zuordnung
verrutscht: jedes Trainingsmuster muss nach der Entdopplung exakt dieselbe
Dichtekarte sehen wie ohne sie.

Verglichen wird gegen einen Aufbau, in dem jeder Schluessel eine EIGENE Kopie
seines Gitters bekommt — damit greift die Entdopplung (die ueber `id()` geht)
nicht, und es entsteht das alte Verhalten.

    python -u test_dichte_dedup.py [--db mini_length_test.db]
"""
import argparse
import os
import sys

import numpy as np
import torch

import flow_matching_runner_length as R

p = argparse.ArgumentParser()
p.add_argument('--db', default='mini_length_test.db')
p.add_argument('--nxi', type=int, default=32)
a = p.parse_args()

R._DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'ergodic_dataset_generator', a.db)
if not os.path.isfile(R._DB_PATH):
    print(f"[uebersprungen] Datenbank fehlt: {R._DB_PATH}")
    sys.exit(0)

fehler = []
def pruefe(b, t):
    print(f"[{'ok' if b else '!!'}] {t}")
    if not b:
        fehler.append(t)

(train_shapes, hold_shapes, train_dens, hold_dens,
 train_len, hold_len) = R._load_shapes(a.nxi, grid_res=64)

n_var = len(train_shapes)
n_eind = len({id(v) for v in train_dens.values()})
print(f"\n  {n_var} Varianten teilen sich {n_eind} eindeutige Dichtekarten")
pruefe(n_eind < n_var, "es werden tatsaechlich Karten geteilt")

# Geteilte Karten muessen wertgleich sein — sonst waere die Zusammenlegung falsch.
nach_id = {}
for k, v in train_dens.items():
    nach_id.setdefault(id(v), []).append(k)
gleich = True
for kid, keys in nach_id.items():
    grund = np.asarray(train_dens[keys[0]])
    for k in keys[1:]:
        if not np.array_equal(grund, np.asarray(train_dens[k])):
            gleich = False
pruefe(gleich, "alle zusammengelegten Karten sind wertgleich")

def bauen(dens):
    np.random.seed(4711)
    return R._build_dataset(train_shapes, dens, hold_shapes, hold_dens,
                            copies_per_char=2, n_particles=16,
                            device='cpu', sample_mode='uniform',
                            train_lengths=train_len)

# a) mit Entdopplung (geteilte Objekte)
x1_a, idx_a, stapel_a, _, len_a = bauen(train_dens)
# b) ohne: jeder Schluessel bekommt eine eigene Kopie
ohne = {k: np.array(v, copy=True) for k, v in train_dens.items()}
x1_b, idx_b, stapel_b, _, len_b = bauen(ohne)

print(f"\n  Stapel mit Entdopplung: {tuple(stapel_a.shape)}"
      f"   ohne: {tuple(stapel_b.shape)}")
pruefe(stapel_a.shape[0] == n_eind, "der Stapel enthaelt genau die eindeutigen Karten")
pruefe(stapel_b.shape[0] == n_var, "die Vergleichsfassung enthaelt alle Varianten")

pruefe(torch.equal(x1_a, x1_b), "dieselben Trainingsbahnen in derselben Reihenfolge")
pruefe(torch.equal(len_a, len_b), "dieselben Laengen in derselben Reihenfolge")

# Der Kern: je Muster dieselbe Dichtekarte.
karten_a = stapel_a[idx_a]
karten_b = stapel_b[idx_b]
gleich_alle = torch.equal(karten_a, karten_b)
if not gleich_alle:
    d = (karten_a - karten_b).abs()
    print(f"     max. Abweichung {float(d.max()):.3e} bei "
          f"{int((d.amax(dim=(1,2)) > 0).sum())} von {len(d)} Mustern")
pruefe(gleich_alle, "jedes Muster sieht exakt dieselbe Dichtekarte wie zuvor")

sp_a = stapel_a.numel() * 4 / 2**20
sp_b = stapel_b.numel() * 4 / 2**20
print(f"\n  Speicher fuer den Stapel: {sp_a:.1f} MB statt {sp_b:.1f} MB "
      f"({sp_b/max(sp_a,1e-9):.1f}x weniger)")

print()
if fehler:
    print(f"{len(fehler)} Pruefung(en) fehlgeschlagen:")
    for f in fehler:
        print("   -", f)
    sys.exit(1)
print("Alle Pruefungen bestanden.")
