# Rooftop PV · Energy Data Hackdays 2026

PV-Visual-placer- · Implementierung für den Branch `Botond`.

YOLO11-Instanzsegmentierung und wetterbasierte Berechnung des zusätzlichen
PV-Potenzials auf Schweizer Dächern. Das Projekt verbindet Luftbilder mit
Sonnendach-Dachflächen, zieht bereits belegte Flächen ab und berechnet den
stündlichen sowie monatlichen Stromertrag mit pvlib.

[Challenge des Swiss Data Science Center](https://www.energydatahackdays.ch/challenges/ai-for-accurate-rooftop-pv-potential)
· [Projekt- und Qualitätsvertrag](docs/PROJECT_CONTRACT.md)

Das Schweizer YOLO11-Modell wurde lokal für 50 Epochen trainiert. Auf 70 getrennten
Testbildern erreicht es **0.779 Masken-mAP50 und 0.683 Flächen-IoU**.
Der strengere mAP50–95-Grenzwert wird noch verfehlt; das Modell ist deshalb
ausdrücklich **experimentell**. [Messwerte und Fehleranalyse](docs/MODEL_CARD_PV.md).

Ein zweites YOLO11-Modell für Dachaufbauten wurde für 40 Epochen auf RID2 trainiert.
Die Hinderniserkennung verfehlt die Qualitätsgrenzen deutlich und darf nur mit
manueller Sichtprüfung verwendet werden. [Hindernis-Modellkarte](docs/MODEL_CARD_OBSTACLES.md).

## Lokal starten

Python 3.12 und [uv](https://docs.astral.sh/uv/) werden benötigt. Apple Silicon
wird automatisch über PyTorch MPS verwendet; alternativ sind CPU und CUDA möglich.

```sh
uv sync --frozen --python 3.12
uv run rooftop-pv doctor
make app
```

Die Anwendung ist unter **http://127.0.0.1:8501** erreichbar. Eine Schweizer Adresse
eingeben, eine Dachfläche auswählen und das Luftbild laden. Nach dem Training lassen
sich PV-Masken erzeugen und zusammen mit manuell geprüften Hindernissen abziehen.
Anschliessend Standort, Ausrichtung, Neigung, Modulparameter und Verluste prüfen
und die Ertragsberechnung starten. Ein manueller Szenariomodus funktioniert auch
ohne registriertes Modell. Die erste Seitenansicht führt keine externen Datenabfragen aus.

## Daten und Training

```sh
# Offiziellen Schweizer Datensatz herunterladen, Lizenz/Provenienz sichern
uv run rooftop-pv download-data
uv run rooftop-pv prepare-data

# YOLO11n-seg mit geografisch getrennten Daten trainieren
uv run rooftop-pv train --config configs/train.yaml

# Bestes Modell auf dem zuvor unbenutzten Testsplit auswerten
uv run rooftop-pv evaluate --split test

# Ein eigenes Luftbild segmentieren; Maske und Polygon-JSON speichern
uv run rooftop-pv predict /absoluter/pfad/luftbild.jpg
```

Die Vorbereitung erstellt `data/processed/swiss-pv/dataset.yaml` und `splits.json`.
Vorhandene Daten werden nicht automatisch ersetzt. Für einen anderen Split einen
neuen Ordner mit `prepare-data --output ...` verwenden.
Das Training lässt sich mit `--epochs`, `--imgsz`, `--batch`, `--device` und `--data`
anpassen. Ein unterbrochener Lauf wird mit
`train --resume artifacts/training/<lauf>/weights/last.pt` fortgesetzt.
Dabei werden Rolle, Daten und Parameter aus dem ursprünglichen Laufmanifest
übernommen; nur `--device` darf abweichen. Geänderte Dataset-Manifeste werden
abgewiesen. Für weitere Epochen eines bereits abgeschlossenen Modells ist ein
neues Fine-Tuning-Experiment nötig, kein Resume des finalisierten Checkpoints.

Der erste Schweizer Datensatz enthält **758 Bilder (1000 × 1000 Pixel)**,
aufgeteilt in **608 Training / 80 Validierung / 70 Test** aus getrennten geografischen
Gruppen. Er enthält die Klasse `solar_panel`. Die binären Quellmasken bleiben für
die Flächenbewertung erhalten; zusammenhängende Bereiche werden für YOLO in
Polygone umgewandelt. Einzelne Module innerhalb einer Anlage sind deshalb nicht
automatisch separate Instanzen. Details: [Datenvertrag](docs/DATA.md).

RID2 ergänzt explizite Dachaufbauten über ein eigenes Trainingsdataset und
`configs/train-obstacles.yaml`. Die Schweizer Bilder mit ausschliesslichen
PV-Labels werden nicht als negative Beispiele für Hindernisse verwendet.
Genfer Katastergeometrien werden als zusätzliche Vektordatenquelle angebunden;
ihre Eignung als Bildlabels hängt von Erfassungsdatum und räumlicher Ausrichtung ab.

```sh
# Zusätzliches Archiv: ca. 6.96 GB; vorhandene Datei wird geprüft und wiederverwendet
uv run rooftop-pv download-obstacles
uv run rooftop-pv prepare-obstacles
uv run rooftop-pv train --config configs/train-obstacles.yaml
```

Verwendet werden die 1’819 dachzentrierten RID2-Bilder aus vier räumlichen Gruppen,
nicht der gesamte separate 50-Regionen-Teil des Archivs. Übertragung von den
Niederlanden auf Schweizer Dächer ist eine zusätzliche, noch zu prüfende
Modellgrenze. [RID2-Datenvertrag](docs/OBSTACLE_DATA.md) · [Genfer Daten](docs/GENEVA.md).

## Alle Dachflächen eines Gebiets analysieren

Der Batch-Ablauf lädt sämtliche Sonnendach-IDs im angegebenen LV95-Ausschnitt
(höchstens 500 m pro Seite), ergänzt Luftbilder für vollständig abgedeckte Dächer
und berechnet eine Restfläche pro Dach. Überlappende Bildkacheln bleiben bei
10 cm/Pixel; bereits belegte PV-Flächen, Hindernisse und optional SITG-Objekte
werden geometrisch vereinigt, nicht mehrfach abgezogen.

```sh
# Genf, Dufour: echter Testbereich, 39 Dachflächen
uv run rooftop-pv analyze-roofs \
  --bbox 2499914.331920191 1117259.4878581492 2500014.331920191 1117359.4878581492 \
  --geneva --output artifacts/geneva-new/dufour

# Gesicherte Vorhersagen erneut geometrisch auswerten, ohne Download/Modelllauf
uv run rooftop-pv recalculate-roofs artifacts/geneva/dufour \
  --output artifacts/geneva-replay/dufour
```

Ausgaben: `roofs.csv`, `usable-roofs-lv95.geojson`, Masken, Quellenmetadaten,
Bildübersicht und Fehlerstatus pro Dach. Existierende Ausgabeordner werden nicht
überschrieben. Die GeoJSON-Koordinaten sind **LV95 / EPSG:2056**, nicht WGS84.
Fehlende Bildabdeckung oder unbrauchbare Dachgeometrie ergibt `null`, keine
erfundene nutzbare Fläche. Die pauschale Modulflächenquote ersetzt keinen
geometrischen Modulbelegungsplan. [Genfer Prüfergebnisse](docs/GENEVA_VALIDATION.md).

## Berücksichtigte Einflüsse

| Einfluss | Grundlage / Behandlung |
| --- | --- |
| Bestehende PV-Anlagen | Trainierte YOLO11-Masken, georeferenziert und auf Dachflächen beschnitten |
| Dachaufbauten | Geprüfte Polygon-Ausschlüsse; zusätzliches RID2-Modell und Genfer Vektorquelle |
| Dachfläche und Neigung | Sonnendach oder manuelle Eingabe; horizontale Fläche wird in geneigte Fläche umgerechnet |
| Ausrichtung und Sonnenstand | Standort und stündliche Sonnengeometrie; intern Nord=0°, Ost=90°, Süd=180° |
| Wolken / jahreszeitliche Einstrahlung | PVGIS Typical Meteorological Year mit direkter und diffuser Einstrahlung |
| Lufttemperatur / Wind | Faiman-Temperaturmodell und temperaturabhängige PV-Leistung |
| Reflexion | Einstellbare Albedo und diffuse Einstrahlung auf der Modulebene |
| Verschattung | PVGIS-Geländehorizont; optionales eigenes Horizontprofil im Rechenmodul |
| Schnee / Schmutz / Alterung | Explizite Verlustannahmen, auch monatlich im Rechenmodul konfigurierbar |
| Elektrische Verluste | Leitungen, Mismatch, Verfügbarkeit, Wechselrichterwirkungsgrad und Leistungsbegrenzung |
| Montageabstände | Geometrische Rand-/Hindernisabstände und konservativer Belegungsfaktor |

Ein einzelnes RGB-Luftbild liefert keine zuverlässige Dachstatik, keine vollständige
Verschattung durch Bäume und Nachbarbauten und keine Prognose von Schnee- oder
Hagelschäden. Dafür sind zusätzliche Messungen, Höhenmodelle oder eine Prüfung
vor Ort nötig. Typische Wetterjahre sind Ertragsszenarien, keine aktuelle Wettervorhersage.
Modulabhängige Spektral- und Einfallswinkelverluste werden separat als Modellgrenze
dokumentiert. [Physikalische Annahmen und Einheiten](docs/PHYSICS.md).

## Weitere Befehle

```sh
# Echte Schweizer Daten mit Quellenmetadaten lokal sichern
uv run rooftop-pv fetch-site "Hauptstrasse 1 5200 Brugg" --weather

# Transparentes Szenario: 80 m² bereits nutzbare, geneigte Modulfläche
uv run rooftop-pv simulate --area 80 --tilt 30 --azimuth 180

# Reproduzierbarer Prüflauf mit echten Brugg-Daten und trainierten Gewichten
uv run rooftop-pv fetch-site "Hauptstrasse 1 5200 Brugg" --weather --output data/sites/brugg-pvgis53
uv run python scripts/verify_brugg.py

# Zusätzlich die echten Dashboard-Aktionen prüfen (kein Browser-Layouttest)
uv run python scripts/verify_dashboard.py

# Automatische Tests für Daten, Geometrie, Physik und Anwendung
make test
make coverage
```

## Quellen

- [Swiss solar panels segmentation, Jean Perbet / SDSC](https://www.kaggle.com/datasets/jeanprbt/swiss-solar-panels-segmentation): Kaggle v1, dort CC0 deklariert; Luftbilder mit binären PV-Masken.
- [Zugehörige U-Net-/DeepLabv3-Referenzimplementierung](https://github.com/jeanprbt/swiss-solar-panel-segmentation): methodische Referenz; unser trainiertes Modell verwendet YOLO11.
- [swisstopo SWISSIMAGE](https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10): Orthofotos und Quellenangaben.
- [BFE Sonnendach](https://opendata.swiss/de/dataset/eignung-von-hausdachern-fur-die-nutzung-von-sonnenenergie): Dachgeometrie, Neigung, Ausrichtung und offizielle Potenzialattribute.
- [SITG Genf: Dachaufbauten](https://sitg.ge.ch/donnees/cad-batiment-horsol-toit-sp): projizierte 2D-Katastergeometrien; kein vollständiger Satz aller sichtbaren Hindernisse.
- [RID2, Krapf et al., TUM](https://doi.org/10.5281/zenodo.14062580): Dachaufbauten aus den Niederlanden, CC BY 4.0.
- [EU JRC PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/en/): typische Wetterjahre und Geländehorizont.
- [STDL: Detection of occupied and free surfaces on rooftops](https://tech.stdl.ch/PROJ-ROOFTOPS/): Grenzen vorhandener Katasterlabels, echte Orthofotos und Bewertung belegter Flächen.
- [EPFL: Quantification of the suitable rooftop area](https://infoscience.epfl.ch/server/api/core/bitstreams/474cb251-bc11-4ac3-a26c-bc39f770cc15/content): Forschungsreferenz zur nutzbaren Dachfläche.

[Quellschnittstellen und Datenstand](docs/SOURCES.md). Zu jedem Download werden
Quell- und Versionsinformationen lokal gespeichert. Rohdaten gehören nicht ins GitLab-Repository.

Die GitHub-Veröffentlichung enthält Code, Konfiguration, Tests und Dokumentation.
Die lokalen Rohdaten, Luftbilder, Modellgewichte und generierten Prüfergebnisse
sind absichtlich nicht eingecheckt. Nach einem frischen Clone müssen die oben
beschriebenen Download-/Trainingsschritte ausgeführt oder die lokal vorhandenen
Gewichte samt Registrierungsmanifesten separat übertragen werden. Die gemessenen
Modellwerte gelten für die in den Modellkarten bezeichneten Checkpoints.

## Ausgaben und Reproduzierbarkeit

| Pfad | Inhalt |
| --- | --- |
| `data/raw/` | Originalarchive und öffentliche Quellen-/Lizenzmetadaten |
| `data/processed/` | Bilder, YOLO-Labels, Originalmasken und Splitmanifest |
| `data/cache/`, `data/sites/` | Wetter, Geometrien, Luftbilder und Quellenstand |
| `artifacts/training/<lauf>/` | `best.pt`, `last.pt`, Lernkurven, Parameter und `run_manifest.json` |
| `artifacts/models/current.json` | Aktuell verwendetes Schweizer PV-Modell mit Qualitätsstatus |
| `artifacts/models/obstacles.json` | Trainiertes Hindernismodell mit separatem Qualitätsstatus |
| `artifacts/evaluations/` | Masken-mAP, IoU gegen Originalmasken, Flächenfehler und Fehlerbeispiele |
| `artifacts/predictions/` | Overlay-Bilder und Polygone in Originalbild-Pixelkoordinaten |
| `artifacts/simulation/` | Stündliche Leistung/Energie als CSV und Zusammenfassung als JSON |

Die Anwendung exportiert die berechneten Szenarien zusätzlich als JSON und CSV.
Gewichte und generierte Daten sind per `.gitignore` ausgeschlossen.
`uv.lock` und `requirements.txt` halten die Paketversionen fest. Trainingsmanifeste
enthalten Seed, Datenhash, Quellcodehash, Geräteangaben und tatsächlichen Laufstatus.
MPS ist nicht für alle Operationen bitgenau deterministisch.

## Lizenz

Der vorhandene Projektcode-Lizenztext bleibt [Apache 2.0](LICENSE.md).
Externe Daten, Bibliotheken und Modellgewichte besitzen eigene Bedingungen.
Insbesondere wird Ultralytics unter AGPL-3.0 bzw. einer kommerziellen Lizenz angeboten;
die Projektlizenz ersetzt diese Bedingungen nicht. Quellen und Attributionen stehen
in den jeweiligen Datendokumentationen.
