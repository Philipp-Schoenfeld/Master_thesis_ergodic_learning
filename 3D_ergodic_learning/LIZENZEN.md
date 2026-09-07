# Eigene prozedurale Formen (Erweiterung auf 200 Flaechen)

Vier neue Kategorien in `organismen.py`, `baugruppen.py`, `superquadriken.py`
und `woerter.py` — 123 zusaetzliche Formen zu den bisherigen 77. Alle vier
sind **vollstaendig selbst erzeugt**: reine Formeln (Superquadriken), ein
selbst gewaehltes Skalarfeld plus Marching Cubes (Organismen), boolesche
Verknuepfung eigener Grundkoerper aus `koerper.py` (Baugruppen), oder dieselbe
bereits geklaerte Schrift wie bei den Einzelbuchstaben (Woerter, DejaVu Sans).
Keine davon hat eine externe Quelle — diese Datei fuehrt sie trotzdem auf,
damit die Belegstelle vollstaendig bleibt und nicht nur die Faelle nennt, bei
denen tatsaechlich etwas zu klaeren war.

| Kategorie | Modul | Anzahl | Verfahren |
|---|---|---|---|
| Organismen | `organismen.py` | 45 | Metaball-Skalarfeld (Gauss-Blobs) + `skimage.measure.marching_cubes` |
| Baugruppen | `baugruppen.py` | 25 | Boolesche Vereinigung/Differenz eigener `koerper.py`-Grundkoerper via `manifold3d` |
| Superquadriken | `superquadriken.py` | 28 | Geschlossene Formel (Exponenten `e1`, `e2`) auf Ikosphaeren-Topologie |
| Woerter | `woerter.py` | 25 | `text_volumen.baue()` auf kurze Zeichenketten statt Einzelbuchstaben |

`manifold3d` (Apache-2.0) ist die neue Abhaengigkeit fuer die Baugruppen —
vorher fehlte sie in dieser Umgebung, siehe die Notiz in `koerper.halbkugel()`.

---

# Herkunft und Lizenz der externen Netze

Keines der hier aufgeführten Modelle liegt im Repository. Sie werden beim
ersten Gebrauch in einen lokalen Zwischenspeicher geholt — die Open3D-Modelle
nach `~/open3d_data/`, die Stanford-Scans nach `3D_ergodic_learning/cache/netze/`.
Beide Verzeichnisse gehören in `.gitignore`.

Diese Datei ist die Belegstelle für die Arbeit: welches Bild welchem Urheber
gehört und unter welcher Bedingung es gezeigt werden darf.

## Über Open3D bezogen

Open3D verteilt diese Modelle über eigene, mit Prüfsumme versehene
Downloader (`open3d.data.*`). Die Herkunftsangaben stammen aus der
Open3D-Dokumentation und den Modell-Repositorien.

| Schlüssel | Modell | Quelle | Lizenz | Zitierhinweis |
|---|---|---|---|---|
| `armadillo` | Stanford-Armadillo | Stanford 3D Scanning Repository, via `open3d.data.ArmadilloMesh` | frei für Forschung und Lehre | „Armadillo, Stanford Computer Graphics Laboratory" |
| `knoten` | Kleeblattknoten | Open3D-Testdaten, via `open3d.data.KnotMesh` | MIT (Open3D) | „Knot mesh, Open3D" |
| `affenkopf` | Suzanne | Blender-Foundation-Modell, via `open3d.data.MonkeyModel` | CC0 | „Suzanne, Blender Foundation (CC0)" |
| `avocado` | Avocado | Khronos glTF-Sample-Models, via `open3d.data.AvocadoModel` | CC-BY 4.0 | „Avocado, © Microsoft, CC-BY 4.0, via Khronos glTF-Sample-Models" |
| `helm` | Damaged Helmet | Khronos glTF-Sample-Models, via `open3d.data.DamagedHelmetModel` | CC-BY 4.0 | „Battle Damaged Sci-fi Helmet, © ctxwing/theblueturtle_, CC-BY 4.0" |
| `fliegerhelm` | Flight Helmet | Khronos glTF-Sample-Models, via `open3d.data.FlightHelmetModel` | CC0 | „FlightHelmet, © Microsoft (CC0)" |

Ausgeschlossen und **nicht** in der Flächenliste:

| Schlüssel | Modell | Grund |
|---|---|---|
| `kiste` | Holzkiste (`CrateModel`) | Nach dem Laden bleiben zwölf Dreiecke. Die ganze Gestalt steckt in der Textur; als Zielfläche wäre sie eine zweite Ausgabe von `koerper.quader`. |
| `schwert` | Schwert (`SwordModel`) | Silhouettentreffer 88,7 % < 90 %. Die Klinge ist eine offene Schale, durch die Strahlen hindurchschlüpfen. |

Beide Entscheidungen fallen im Aufnahmetor von `externe_netze.py` und stehen
dort mit Begründung im Protokoll, statt später im Datenbankbau als
unerklärliche Fehlschüsse aufzutauchen.

## Stanford 3D Scanning Repository

**Stand der Entscheidung: nicht geholt.** Die Flächenliste kommt ohne
Stanford-Modelle aus — 79 Trainingsflächen und 4 zurückgehaltene. Der Abruf
bleibt vorbereitet, damit die Entscheidung später ohne Codeänderung umkehrbar
ist; ausgeführt wurde er nicht, und im Zwischenspeicher liegt nichts.

Diese Dateien werden **nur auf ausdrückliche Anweisung** geholt
(`python externe_netze.py --fetch <name>`). Vorher zeigt `--manifest` URL und
erwartete Größe; der SHA-256 wird beim ersten Abruf festgestellt und in
`cache/netze/netze.lock` festgeschrieben, sodass jeder weitere Abruf dagegen
prüft. Stanford veröffentlicht selbst keine Prüfsummen — ein vorab bekannter
Hash ist deshalb nicht zu haben, und diese Datei behauptet auch nicht, einen zu
kennen.

| Schlüssel | Modell | URL | Größe |
|---|---|---|---|
| `happy` | Happy Buddha | `graphics.stanford.edu/pub/3Dscanrep/happy/happy_recon.tar.gz` | ~32 MB |
| `drache` | Stanford-Drache (Heldout) | `graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz` | ~12 MB |

Lizenz: Die Modelle des Stanford 3D Scanning Repository dürfen für Forschung
frei verwendet werden, unter Nennung der Herkunft. Der vorgesehene
Zitierhinweis lautet:

> The models used here are provided by the Stanford Computer Graphics
> Laboratory (Stanford 3D Scanning Repository).

### Warum Thai Statue und Asian Dragon fehlen

Beide stammen von der XYZ-RGB-Seite desselben Repositoriums. Deren Lizenz ist
genau die, die im Arbeitsplan als Ausschlussgrund für „XYZRGB" genannt wird:
Nutzung ausschließlich für nichtkommerzielle Zwecke, mit ausdrücklicher
Zustimmung für Weitergabe. Sie einzeln aufzunehmen und die Quelle im selben
Atemzug auszuschließen wäre ein Widerspruch. Sie stehen deshalb nicht in
`externe_netze.STANFORD`, und die Zahl der externen Flächen ist entsprechend
kleiner als ursprünglich geplant.
