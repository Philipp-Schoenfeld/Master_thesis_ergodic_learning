# -*- coding: utf-8 -*-
r"""
test_length_freqs.py
====================
Pruefungen zur Laengenkodierung des laengenkonditionierten Netzes.

Anlass: die Einbettung uebernahm den Oktavenstapel `2^k * pi` der
Ortskodierung. Fuer eine Koordinate in [0,1] ist das richtig — dort
durchlaeuft die niedrigste Frequenz eine halbe Periode. Die normierte Laenge
ist aber durch die Standardabweichung geteilt und liegt in [-3,33 ; +3,35].
Da jede Frequenz ein Vielfaches von pi ist, ist der *ganze* Merkmalsvektor
periodisch in u mit der Periode 2: verschiedene Laengen bekommen bitgleiche
Kodierungen, und das Netz kann sie nicht unterscheiden — nicht naeherungsweise,
sondern exakt.

    python -u test_length_freqs.py
"""
import math
import sys

import torch

from flow_matching_cond_particles_length import (
    LengthEmbedding, ParticleCrossAttnFlowNetwork,
    lade_modellzustand, ruecksetzen_laengenkopf, uebernehmen_aus,
)

# Normierung des tatsaechlichen Datensatzes (ergodic_dataset_length.db):
# Median der Trainingslaengen und Standardabweichung von u.
REF, SKALA = 11.045, 0.4041
L_MIN, L_MAX = 2.14, 45.67

fehler = []


def pruefe(bedingung, text):
    print(f"[{'ok' if bedingung else '!!'}] {text}")
    if not bedingung:
        fehler.append(text)


def merkmale(emb, L):
    u = emb.normiere(torch.tensor([float(L)]).reshape(-1, 1))
    a = u * emb.freqs
    return torch.cat([a.sin(), a.cos()], dim=-1)[0]


# ---------------------------------------------------------------- 1 Oktaven
print("\n--- Oktavenstapel: der dokumentierte Mangel ---")
alt = LengthEmbedding(D=64, log_ref=REF, log_scale=SKALA, freq_mode='oktaven')
spanne = alt.periodenspanne(L_MIN, L_MAX)
print(f"    Periodenspanne ueber [{L_MIN}; {L_MAX}] = {spanne:.2f}")
pruefe(spanne > 3.0, "Oktaven decken mehr als drei Perioden ab (mehrdeutig)")

# Genau eine Periode weiter in u -> identische Kodierung.
zwilling = lambda L: math.expm1(math.log1p(L) + 2.0 * SKALA)
# Vergleichsmassstab: zwei ehrlich verschiedene Laengen. Alles, was um
# Groessenordnungen darunter liegt, ist float32-Rundung und kein Signal.
massstab = float((merkmale(alt, 4.0) - merkmale(alt, 5.0)).abs().max())
print(f"    Massstab: L=4,00 gegen L=5,00 -> max|Differenz| = {massstab:.2e}")
for L in (4.0, 7.0, 15.0):
    L2 = zwilling(L)
    d = float((merkmale(alt, L) - merkmale(alt, L2)).abs().max())
    print(f"    L={L:6.2f} <-> L'={L2:6.2f}   max|Differenz| = {d:.2e}"
          f"   ({massstab / max(d, 1e-12):.0f}x kleiner als der Massstab)")
    pruefe(d < massstab / 1000.0 and L2 <= L_MAX,
           f"L={L:.2f} und L'={L2:.2f} sind unter Oktaven ununterscheidbar")

# ---------------------------------------------------------------- 2 linear
print("\n--- Lineare Frequenzen: die Korrektur ---")
neu = LengthEmbedding(D=64, log_ref=REF, log_scale=SKALA, freq_mode='linear')
spanne_neu = neu.periodenspanne(L_MIN, L_MAX)
print(f"    Periodenspanne ueber [{L_MIN}; {L_MAX}] = {spanne_neu:.2f}")
pruefe(spanne_neu < 1.0, "lineare Frequenzen bleiben unter einer Periode")

for L in (4.0, 7.0, 15.0):
    L2 = zwilling(L)
    if L2 > L_MAX:
        continue
    d = float((merkmale(neu, L) - merkmale(neu, L2)).abs().max())
    pruefe(d > 0.2, f"L={L:.2f} und L'={L2:.2f} sind unter linear unterscheidbar")

# Lokalitaet: die Aehnlichkeit muss mit wachsendem Abstand fallen.
def aehnlich(emb, La, Lb):
    a, b = merkmale(emb, La), merkmale(emb, Lb)
    return float(a @ b / (a.norm() * b.norm()))

reihe = [aehnlich(neu, REF, L) for L in (11.0, 15.0, 20.0, 28.0)]
print("    Aehnlichkeit zur Referenz L=11 bei L = 11 / 15 / 20 / 28:")
print("      " + "  ".join(f"{v:+.3f}" for v in reihe))
pruefe(all(reihe[i] > reihe[i + 1] for i in range(len(reihe) - 1)),
       "lineare Kodierung faellt monoton mit dem Abstand (Lokalitaet)")

reihe_alt = [aehnlich(alt, REF, L) for L in (11.0, 15.0, 20.0, 28.0)]
print(f"    unter Oktaven ist schon L=11,00 gegen die Referenz L=11,045 nur "
      f"zu {reihe_alt[0]:+.3f} aehnlich")
pruefe(reihe_alt[0] < 0.9,
       "Oktaven trennen bereits 0,4 % Laengenunterschied fast vollstaendig")
print("    dieselbe Reihe unter Oktaven:")
print("      " + "  ".join(f"{v:+.3f}" for v in reihe_alt))
pruefe(not all(reihe_alt[i] > reihe_alt[i + 1] for i in range(len(reihe_alt) - 1)),
       "Oktaven fallen nicht monoton (kein Lokalitaetsbegriff)")

# ------------------------------------------------------- 3 Frequenzen im Zustand
print("\n--- Frequenzen duerfen nicht im state_dict stehen ---")
netz_alt = ParticleCrossAttnFlowNetwork(nxi=16, nd=2, D=64, log_ref=REF,
                                        log_scale=SKALA,
                                        length_freq_mode='oktaven')
netz_neu = ParticleCrossAttnFlowNetwork(nxi=16, nd=2, D=64, log_ref=REF,
                                        log_scale=SKALA,
                                        length_freq_mode='linear')
pruefe(not [k for k in netz_neu.state_dict() if k.endswith('.freqs')],
       "kein *.freqs-Schluessel im state_dict")

# Ein Zustand im alten Format (mit persistenten Frequenzen) darf die Wahl
# des neuen Modells nicht ueberschreiben.
zustand = dict(netz_alt.state_dict())
zustand['length_emb.freqs'] = 2.0 ** torch.arange(8).float() * math.pi
zustand['start_emb.freqs'] = 2.0 ** torch.arange(8).float() * math.pi
vorher = netz_neu.length_emb.freqs.clone()
lade_modellzustand(netz_neu, zustand)
pruefe(torch.equal(netz_neu.length_emb.freqs, vorher),
       "alter Checkpoint ueberschreibt die Frequenzen des Modells nicht")

# ---------------------------------------------------------- 4 Ruecksetzen
print("\n--- Ruecksetzen des Laengenkopfs ---")
with torch.no_grad():
    netz_neu.null_length_token.fill_(0.7)
w_vorher = netz_neu.length_emb.net[0].weight.clone()
ruecksetzen_laengenkopf(netz_neu)
pruefe(not torch.allclose(w_vorher, netz_neu.length_emb.net[0].weight),
       "die Gewichte der Laengen-Einbettung wurden neu initialisiert")
pruefe(float(netz_neu.null_length_token.detach().abs().max()) == 0.0,
       "der Null-Token steht wieder auf null")

# ------------------------------------------------------- 5 Vorwaertsdurchlauf
print("\n--- Vorwaertsdurchlauf und Wirkung der Laenge ---")
torch.manual_seed(0)
x = torch.randn(1, 16, 2) * 0.1 + 0.5
t = torch.tensor([0.5])
P = torch.rand(1, 256, 3)
p0 = torch.tensor([[0.3, 0.3]])

for name, netz in (('oktaven', netz_alt), ('linear', netz_neu)):
    netz.eval()
    with torch.no_grad():
        v = lambda L: netz(x, t, P, start=p0, length=torch.tensor([float(L)]))[0]
        basis = v(REF)
        d = {L: float((v(L) - basis).abs().mean()) for L in (4, 8, 15, 25, 40)}
    pruefe(all(math.isfinite(val) for val in d.values()),
           f"Vorwaertsdurchlauf in Modus {name} liefert endliche Werte")
    # Bei frischer Initialisierung ist die Wirkung auf das Feld exakt null,
    # weil die FiLM-Projektionen null-initialisiert sind. Das ist Absicht und
    # wird hier festgehalten: ein neu angehaengter Konditionierungseingang
    # veraendert das Netz zu Trainingsbeginn nicht.
    pruefe(max(d.values()) == 0.0,
           f"Modus {name}: null-initialisiertes FiLM laesst die Laenge zu "
           f"Beginn wirkungslos")

print("    Wirkung auf die Einbettung selbst (dort greift der Unterschied):")
for name, emb in (('oktaven', alt), ('linear', neu)):
    with torch.no_grad():
        basis = emb(torch.tensor([REF]))
        d = {L: float((emb(torch.tensor([float(L)])) - basis).abs().mean())
             for L in (12, 15, 20, 28, 40)}
    print(f"    {name:8s} " + "  ".join(f"L={k}:{val:.3f}" for k, val in d.items()))
    # Die absolute Groesse haengt an der zufaelligen Initialisierung des MLP
    # und ist bedeutungslos; strukturell ist das *Verhaeltnis* zwischen einer
    # nahen und einer fernen Laenge. Es sagt, ob die Kodierung Naehe kennt.
    verhaeltnis = d[28] / max(d[12], 1e-9)
    print(f"             nah (L=12) gegen fern (L=28): Faktor {verhaeltnis:.1f}")
    if name == 'linear':
        pruefe(verhaeltnis > 3.0,
               "linear: eine nahe Laenge wirkt deutlich schwaecher als eine ferne")
    else:
        pruefe(verhaeltnis < 1.5,
               "oktaven: nah und fern wirken gleich stark (keine Naehe)")

# ---------------------------------------------- 6 Warmstart ohne Laengeneingang
print("\n--- Warmstart aus einem Modell ohne Laengeneingang ---")

# Quelle nachbilden: dasselbe Netz, aber ohne die fuenf Laengen-Tensoren und
# mit einem kleineren nxi, wie es das startpunktkonditionierte Modell hat.
quelle = ParticleCrossAttnFlowNetwork(nxi=25, nd=2, D=64, log_ref=REF,
                                      log_scale=SKALA, length_freq_mode='linear')
with torch.no_grad():                      # FiLM aus der Null holen, sonst ist
    for mod in quelle.modules():           # der Test blind fuer Stoerungen
        if hasattr(mod, 'film_proj'):
            for q in mod.film_proj.parameters():
                q.normal_(0.0, 0.02)
q_zustand = {k: v for k, v in quelle.state_dict().items()
             if 'length' not in k}
q_zustand['time_emb.freqs'] = torch.zeros(32)      # altes Format nachstellen
q_zustand['start_emb.freqs'] = torch.zeros(8)

ziel = ParticleCrossAttnFlowNetwork(nxi=32, nd=2, D=64, log_ref=REF,
                                    log_scale=SKALA, length_freq_mode='linear')
frisch = uebernehmen_aus(ziel, q_zustand)
pruefe(sorted(frisch) == sorted(['null_length_token',
                                 'length_emb.net.0.weight', 'length_emb.net.0.bias',
                                 'length_emb.net.2.weight', 'length_emb.net.2.bias']),
       "genau der Laengenkopf bleibt frisch, alles andere wird uebernommen")
pruefe(ziel.pos_emb.shape[-1] == 32,
       "pos_emb wurde auf das neue nxi interpoliert")

ruecksetzen_laengenkopf(ziel)
ziel.eval()
torch.manual_seed(1)
xz = torch.randn(1, 32, 2) * 0.1 + 0.5
Pz = torch.rand(1, 64, 3)
with torch.no_grad():
    ohne = ziel(xz, t, Pz, start=p0)[0]
    mit = [ziel(xz, t, Pz, start=p0, length=torch.tensor([float(L)]))[0]
           for L in (4.0, 11.0, 40.0)]
abw = max(float((m - ohne).abs().max()) for m in mit)
print(f"    max|Ausgabe mit Laenge - Ausgabe ohne| = {abw:.2e}")
pruefe(abw == 0.0,
       "nach dem Warmstart ist der Laengeneingang exakt wirkungslos "
       "(das erweiterte Netz gleicht dem Ausgangsmodell)")

# Der Pfad darf trotzdem nicht tot sein. Wichtig ist dabei die Reihenfolge:
# im ersten Schritt bekommt nur die LETZTE Schicht einen Gradienten (ihr
# Gradient ist das Produkt aus Rueckwaertssignal und den Aktivierungen davor,
# beide von null verschieden). Die frueheren Schichten bekommen ihn erst,
# sobald die letzte nicht mehr null ist — also ab dem zweiten Schritt. Genau
# so verhaelt sich auch eine Zero-Convolution.
ziel.train()
opt = torch.optim.SGD(ziel.parameters(), lr=1e-2)
aus = ziel(xz, t, Pz, start=p0, length=torch.tensor([7.0]))[0]
aus.square().mean().backward()
g_letzte = ziel.length_emb.net[2].weight.grad
g_erste = ziel.length_emb.net[0].weight.grad
pruefe(g_letzte is not None and float(g_letzte.abs().max()) > 0.0,
       "Schritt 1: die letzte Schicht des Laengenkopfs bekommt einen Gradienten")
pruefe(float(g_erste.abs().max()) == 0.0,
       "Schritt 1: die erste Schicht bekommt noch keinen (Ausgang ist null)")
opt.step(); opt.zero_grad()
aus = ziel(xz, t, Pz, start=p0, length=torch.tensor([7.0]))[0]
aus.square().mean().backward()
pruefe(float(ziel.length_emb.net[0].weight.grad.abs().max()) > 0.0,
       "Schritt 2: der ganze Laengenkopf lernt")

print()
if fehler:
    print(f"{len(fehler)} Pruefung(en) fehlgeschlagen:")
    for f in fehler:
        print("   -", f)
    sys.exit(1)
print("Alle Pruefungen bestanden.")
