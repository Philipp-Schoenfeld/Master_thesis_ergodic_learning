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


def load_surface_db(db_path=DEFAULT_DB, nxi=25, surfaces=None,
                    splits=('train', 'val'), max_jump=None, max_miss=None,
                    n_train_shapes=None):
    """Eintraege der projizierten Datenbank.

    -> Liste von dicts mit `name`, `split`, `surface`, `x1` (nxi, 9) und
       `parts` (N, 4).

    `max_jump` und `max_miss` filtern ueber die mitgeschriebenen Guetemasse.
    Ein Sprung entsteht, wo zwei benachbarte Bahnpunkte auf verschiedene Seiten
    einer Kante treffen; ein Fehlschuss dort, wo die 2D-Bahn ueber die
    Silhouette hinauslief und der naechstgelegene Punkt einspringen musste.
    """
    con = sqlite3.connect(db_path)
    q = ("SELECT shape_name, split, surface, traj_pos, traj_rot6, particles, "
         "n_points, n_particles, miss_frac, jump_max FROM ergodic_pairs_3d "
         f"WHERE split IN ({','.join('?' * len(splits))})")
    par = list(splits)
    if surfaces:
        q += f" AND surface IN ({','.join('?' * len(surfaces))})"
        par += list(surfaces)
    q += " ORDER BY id ASC"

    gesehen, out = {}, []
    for (nm, sp, sf, bp, br, bpa, npts, npar, miss, jump) in con.execute(q, par):
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
        out.append(dict(
            name=nm, split=sp, surface=sf,
            x1=np.concatenate([pos[idx], rot[idx]], axis=-1).astype(np.float32),
            parts=np.frombuffer(bpa, dtype=np.float32).reshape(npar, 4).copy()))
    con.close()
    return out


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
