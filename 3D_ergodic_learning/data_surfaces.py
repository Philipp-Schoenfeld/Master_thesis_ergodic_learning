r"""
data_surfaces.py
================
Ladepfad fuer `ergodic_dataset_3d.db` — die projizierte 3D-Datenbank.

Der Unterschied zu `data_3d.py` in einem Satz: dort wird die Zieldichte als
Volumen gehalten und die Partikelwolke bei jedem Schritt neu daraus gezogen,
hier liegen Bahn, Rahmen und Partikel fertig in der Datenbank.

Das hat zwei Folgen, die beim Trainieren zu bedenken sind:

* **Die Partikel sind fest.** Beim Volumenpfad wirkte das Neuziehen wie eine
  leichte Augmentierung der Konditionierung; die faellt hier weg. Dafuer sind
  die Partikel exakt die, auf denen der Pfad erzeugt wurde.
* **Die Rahmen kommen aus der Datenbank**, nicht aus einer Stufe-0-Heuristik.
  `orientation_targets` wird also nicht mehr gebraucht — was die frueher
  gemessene Schwaeche behebt, dass die Zielrichtung bei planaren Daten fast
  konstant war.

Die Guetemasse jedes Eintrags stehen als Spalten in der Tabelle, damit sich
schlechte Beispiele beim Laden aussortieren lassen, statt sie erst im Training
zu bemerken.
"""
import os
import sqlite3
import numpy as np

DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'ergodic_dataset_3d.db')


ALLE_SPLITS = ('train', 'val_form', 'val_flaeche', 'val_beides')


def _spalten(con):
    return {r[1] for r in con.execute('PRAGMA table_info(ergodic_pairs_3d)')}


def load_surface_db(db_path=DEFAULT_DB, nxi=25, surfaces=None,
                    splits=ALLE_SPLITS, max_jump=None, max_miss=None,
                    n_train_shapes=None):
    """Eintraege der projizierten Datenbank.

    -> Liste von dicts mit `name`, `split`, `surface`, `gruppe`, `x1` (nxi, 9),
       `parts` (N, 4) und `start` (3,).

    `max_jump` und `max_miss` filtern ueber die mitgeschriebenen Guetemasse.
    Ein Fehlschuss entsteht, wo die 2D-Bahn ueber die Silhouette hinauslief;
    solche Rohpunkte sind in `traj_pos`/`traj_rot6` bereits entfernt (siehe
    `project_db_3d.py`), `n_points` also ggf. kleiner als die volle Rohbahn.
    Ein Sprung ist trotzdem moeglich, wo zwei der verbliebenen Punkte auf
    verschiedenen Seiten einer laengeren Fehlschuss-Luecke oder einer Kante
    liegen.

    Abwaertskompatibel gegenueber der alten Datenbank: fehlen die Spalten
    `gruppe` und `start_pos`, wird die Gruppe auf `'unbekannt'` gesetzt und der
    Startpunkt aus dem ersten Bahnpunkt genommen. Ohne diese Ruecksicht liesse
    sich `ergodic_dataset_3d_alt.db` nicht mehr laden — und die ist der
    Vergleichsstand, gegen den der Warmstart gemessen wird.
    """
    con = sqlite3.connect(db_path)
    hat = _spalten(con)
    felder = ["id", "shape_name", "split", "surface", "traj_pos", "traj_rot6",
              "particles", "n_points", "n_particles", "miss_frac", "jump_max"]
    felder.append("gruppe" if 'gruppe' in hat else "'unbekannt'")
    felder.append("start_pos" if 'start_pos' in hat else "NULL")
    q = (f"SELECT {', '.join(felder)} FROM ergodic_pairs_3d "
         f"WHERE split IN ({','.join('?' * len(splits))})")
    par = list(splits)
    if surfaces:
        q += f" AND surface IN ({','.join('?' * len(surfaces))})"
        par += list(surfaces)
    q += " ORDER BY id ASC"

    gesehen, out = {}, []
    for row in con.execute(q, par):
        (rid, nm, sp, sf, bp, br, bpa, npts, npar, miss, jump, grp, bst) = row
        if max_jump is not None and jump > max_jump:
            continue
        if max_miss is not None and miss > max_miss:
            continue
        if n_train_shapes and sp == 'train':
            gesehen.setdefault(sf, set())
            if nm not in gesehen[sf] and len(gesehen[sf]) >= n_train_shapes:
                continue
            gesehen[sf].add(nm)

        pos = np.frombuffer(bp, dtype=np.float32).reshape(npts, 3)
        rot = np.frombuffer(br, dtype=np.float32).reshape(npts, 6)
        idx = np.linspace(0, npts - 1, nxi).astype(int)
        # Der Startpunkt ist der erste Punkt der *vollen* Bahn, nicht der erste
        # der auf nxi ausgeduennten — bei nxi=25 aus 201 Punkten faellt der
        # erste zwar zusammen, aber darauf soll sich nichts stillschweigend
        # verlassen.
        start = (np.frombuffer(bst, dtype=np.float32).copy() if bst is not None
                 else pos[0].copy())
        out.append(dict(
            id=rid, name=nm, split=sp, surface=sf, gruppe=grp,
            n_particles=npar,
            x1=np.concatenate([pos[idx], rot[idx]], axis=-1).astype(np.float32),
            parts=np.frombuffer(bpa, dtype=np.float32).reshape(npar, 4).copy(),
            start=start.astype(np.float32)))
    con.close()
    return out


def ziehgewichte(entries, mix=None, ebene_flach_anteil=0.5):
    """Ziehgewicht je Eintrag, sodass die Gruppen im vorgegebenen Verhaeltnis
    vorkommen.

    Warum ueberhaupt gewichtet und nicht einfach gleichverteilt: die Datenbank
    hat jetzt sechsundzwanzig Buchstaben und sechsunddreissig Koerper, aber nur
    zehn Ebenen. Gleichverteilt ueber die Eintraege kaeme die Trainingslage
    `ebene_flach` — die Lage, auf der der geladene Checkpoint 1750 Epochen
    verbracht hat — auf gut ein Prozent der Zuege. Das ist genau der Weg, auf
    dem ein Feinabgleich vergisst, was er schon konnte.

    Warum nicht ueber eine zweite Datenbank: die Wiederholung soll *dieselben*
    Beispiele treffen, mit denselben Partikeln, nur oefter gezogen. Eine zweite
    Quelle brachte zusaetzlich die Frage mit, ob ihre Eintraege dieselbe Bauart
    haben.

    Warum nicht 50 % Ebenen: bei 79 gekruemmten Flaechen waere die Haelfte
    aller Zuege auf einer unverzerrten Projektion ein Ruecktritt hinter das,
    was der Datensatz kann.

    `mix` ist ein dict Gruppe -> Anteil; fehlt eine Gruppe, teilt sie sich den
    Rest gleichmaessig mit den anderen fehlenden. Innerhalb der Gruppe `ebene`
    geht `ebene_flach_anteil` an genau diese eine Flaeche.
    """
    gruppen = {}
    for i, e in enumerate(entries):
        gruppen.setdefault(e['gruppe'], []).append(i)

    mix = dict(mix or {})
    fest = {g: a for g, a in mix.items() if g in gruppen}
    offen = [g for g in gruppen if g not in fest]
    rest = max(0.0, 1.0 - sum(fest.values()))
    for g in offen:
        fest[g] = rest / max(len(offen), 1)

    w = np.zeros(len(entries), dtype=np.float64)
    for g, idx in gruppen.items():
        anteil = fest.get(g, 0.0)
        if anteil <= 0 or not idx:
            continue
        if g == 'ebene':
            flach = [i for i in idx if entries[i]['surface'] == 'ebene_flach']
            rest_i = [i for i in idx if entries[i]['surface'] != 'ebene_flach']
            if flach and rest_i:
                w[flach] = anteil * ebene_flach_anteil / len(flach)
                w[rest_i] = anteil * (1 - ebene_flach_anteil) / len(rest_i)
                continue
        w[idx] = anteil / len(idx)
    s = w.sum()
    return w / s if s > 0 else np.full(len(entries), 1.0 / max(len(entries), 1))


def mix_parsen(argumente):
    """['ebene=0.25', 'extern=0.1'] -> {'ebene': 0.25, 'extern': 0.1}"""
    out = {}
    for a in (argumente or []):
        if '=' not in a:
            raise SystemExit(f"--mix erwartet Gruppe=Anteil, bekam {a!r}")
        k, v = a.split('=', 1)
        out[k.strip()] = float(v)
    if sum(out.values()) > 1.0 + 1e-9:
        raise SystemExit(f"--mix summiert sich auf {sum(out.values()):.3f} > 1")
    return out


class LazyParticles:
    """Rueckfallebene: Partikelwolken bei Bedarf aus der Datenbank lesen.

    Gedacht fuer den Fall, dass der Stapel selbst als angehefteter CPU-Tensor
    nicht mehr passt. Der Preis ist eine SQLite-Abfrage je Minibatch; bei
    Minibatch 128 und 512 Partikeln sind das 256 kB, die pro Schritt vom
    Datentraeger kommen. Spuerbar, aber es laeuft — anders als ein Lauf, der an
    einem vollen Speicher stirbt.
    """

    def __init__(self, db_path, ids, n_particles):
        self.db_path, self.n = db_path, int(n_particles)
        self.ids = list(ids)
        self._con = None
        self.shape = (len(self.ids), self.n, 4)

    def _c(self):
        if self._con is None:
            self._con = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._con

    def __len__(self):
        return len(self.ids)

    def hole(self, idx):
        """idx: 1D-Feld von Zeilenpositionen -> (len(idx), n, 4) float32."""
        import torch
        wahl = [self.ids[int(i)] for i in idx]
        platz = ','.join('?' * len(wahl))
        roh = dict(self._c().execute(
            f'SELECT id, particles FROM ergodic_pairs_3d WHERE id IN ({platz})',
            wahl))
        out = np.stack([np.frombuffer(roh[i], dtype=np.float32).reshape(self.n, 4)
                        for i in wahl])
        return torch.from_numpy(out)


def volume_from_particles(parts, res=48, sigma=1.2):
    """Grobe Dichte aus einer Partikelwolke — nur zum Zeichnen.

    Der Volumenpfad braucht ein Gitter fuer die Darstellung der Zieldichte.
    Die projizierte Datenbank kennt keins, also wird eins gesplattet: jedes
    Partikel traegt sein Gewicht in seine Zelle, danach eine kurze Glaettung.
    Fuer das Training wird das nie benutzt.
    """
    from scipy.ndimage import gaussian_filter
    vol = np.zeros((res, res, res), dtype=np.float32)
    p = np.clip(parts[:, :3], 0.0, 1.0)
    ix = np.clip((p * (res - 1)).round().astype(int), 0, res - 1)
    np.add.at(vol, (ix[:, 2], ix[:, 1], ix[:, 0]), parts[:, 3])
    vol = gaussian_filter(vol, sigma=sigma)
    m = vol.max()
    return vol / m if m > 0 else vol


def stapeln(entries, device='cpu'):
    """-> x1 (E, nxi, 9), parts (E, N, 4), index (E,) als Torch-Tensoren."""
    import torch
    x1 = torch.from_numpy(np.stack([e['x1'] for e in entries]))
    pa = torch.from_numpy(np.stack([e['parts'] for e in entries]))
    return x1.to(device), pa.to(device)
