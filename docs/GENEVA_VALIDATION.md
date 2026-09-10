# Genfer Prüfung der Dachflächen-Pipeline

Stand: 10. September 2026. Dies ist ein nachvollziehbarer Forschungsprototyp,
keine freigegebene automatische Anlagenplanung. Beide trainierten Modelle
verfehlen weiterhin ihre Qualitätsgrenzen. Es wurde nicht auf den Genfer
Prüfbildern trainiert und kein Konfidenzwert anhand dieser Bilder optimiert.

## Tatsächlich ausgeführt

| Teilaufgabe | Nachweis |
| --- | --- |
| Schweizer PV-Modell prüfen und ausführen | 50 Epochen; separater 70-Bilder-Test; reale Inferenz in Brugg und Genf; [Modellkarte](MODEL_CARD_PV.md) |
| Genfer Aufbauten prüfen | Live-Abfragen der SITG-Quelle, exakte LV95-Abfragegrenzen, Paginierung und Quellenstand; [Datenvertrag](GENEVA.md) |
| Hindernismodell erstellen | Eigenes YOLO11n-seg, 40 Epochen auf 1’819 dachzentrierten RID2-Bildern; [Modellkarte](MODEL_CARD_OBSTACLES.md) |
| Auf Genfer swisstopo-Bildern testen | Beide eingefrorenen Checkpoints auf 17 echten SWISSIMAGE-Kacheln in drei Gebieten ausgeführt; Übersichten visuell geprüft |
| Bilder mit Sonnendach verbinden | Alle vom Dienst gelieferten Dach-IDs im jeweiligen Abfragegebiet; LV95-WMS-Bounds, Nordorientierung und 0,1-m-Pixeltransformation |
| Restfläche pro Dach extrahieren | Jede ID bleibt als CSV-/GeoJSON-Zeile erhalten; getrennte Quellenvereinigungen, Randabstände, Neigung und Fehlerstatus |

Die Eingaben sind Luftbilder/Orthofotos, keine Satellitenaufnahmen. Der Ablauf
gilt für ausdrücklich gewählte Gebiete, nicht als Vollerhebung des Kantons.

## Gebiete und Ausgaben

Je Gebiet wurde eine 100 × 100 m grosse Suchbox um die folgenden LV95-Zentren
verwendet. Die Bildabdeckung wurde darüber hinaus erweitert, um alle betroffenen
Dächer vollständig zu erfassen. Keine Auswahl der ersten oder grössten Dächer.

| Gebiet | Zentrum Ost / Nord | Dach-IDs | 100-m-Bildkacheln | Berechnete Restflächen | Geometriefehler |
| --- | --- | ---: | ---: | ---: | ---: |
| Dufour | 2499964.331920191 / 1117309.4878581492 | 39 | 4 | 39 | 0 |
| HUG | 2500448.710691178 / 1116622.9013157906 | 47 | 9 | 47 | 0 |
| Meyrin | 2498194.469302618 / 1119077.0819637834 | 111 | 4 | 90 | 21 |

Alle **197 IDs** wurden exportiert. **176** liefern eine geometrische Restfläche;
darunter können korrekt leere Restflächen sein. Bei **21** Meyrin-Segmenten ist
die gelieferte 2D-Geometrie degeneriert. Für diese bleibt die Restfläche `null`
mit Fehlergrund. Weder die amtliche Fläche noch ein künstlicher Puffer wird als
Ersatzgeometrie eingesetzt. Vollständige Bildabdeckung behebt diesen Quellfehler
nicht. Ein unabhängiger Vergleich mit dem offiziellen Get-Features-Endpunkt
bestätigte diese degenerierten Geometrien; alternative Ausgabeformate und
Präzisionsparameter lieferten keine wiederherstellbare Fläche.
[Unabhängiges Mapping-Audit](GENEVA_MAPPING_AUDIT.md).
Ein gültiges Polygon nach `make_valid` darf verwendet werden; dabei
entfernte flächenlose Reste werden pro Dach als Geometriehinweis dokumentiert.

Lokale Originalausgaben: `artifacts/geneva/{dufour,hug,meyrin}/`.
Neu berechnete Flächen mit vollständiger Abstandsbilanz:
`artifacts/geneva-final/{dufour,hug,meyrin}/`.
Letztere verwenden dieselben gespeicherten Masken und Quellen, ohne erneute
Inferenz. `report.json` hält Herkunft und SHA-256 der Replay-Eingaben fest.
Die Bilddateien bleiben in den Originalordnern; `sources.json` verweist auf sie.
Diese generierten Daten sind nicht Bestandteil des Git-Repositories.

Pro Gebiet existieren `roofs.csv`, `usable-roofs-lv95.geojson`, `analysis.json`,
`model-only.json`, `predictions.json`, `sources.json`, `geneva-inventory.json`,
`overview.jpg` und `report.json`. Die GeoJSON-Datei verwendet EPSG:2056.
Die maskenbasierte und die um SITG ergänzte Rechnung bleiben getrennt prüfbar.

## Was die Bildprüfung zeigt

Die Übersichten zeigen Sonnendach-Umrisse blau, PV-Komponenten grün,
KI-Hindernisse rot und SITG-Geometrien orange. Komponenten sind nicht gleich
Objektanzahl: getrennte Maskenteile und überlappende Inferenzkacheln können
dasselbe Objekt mehrfach liefern; vor der Flächenberechnung werden sie vereinigt.

- Dufour: Dachaufbauten werden teilweise getroffen. Zahlreiche kreisförmige
  SITG-Aufbauten am südlichen Gebäudeteil sind nicht entsprechend durch die KI
  abgedeckt. Eine grüne Vorhersage am dunklen Innenhof-/Dachrand ist verdächtig.
- HUG: Das Modell markiert grosse technische Aufbauten und Bereiche am runden
  Dach, liefert aber auch breite bzw. doppelte Konturen. PV-Vorhersagen auf
  länglichen Dachstreifen benötigen eine Prüfung anhand der Originalbilder.
- Meyrin: Mehrere technische Anlagen werden umrandet; andere sichtbare
  Strukturen bleiben ohne Maske. Schatten und Bäume verdecken Dachabschnitte.
- In allen Übersichten treten rote Vorhersagen ausserhalb von Dächern auf,
  etwa im Strassenraum. Sie werden beim Dachverschnitt nicht abgezogen.
- Dachlinien und sichtbare Dachkanten sind nicht überall deckungsgleich.
  Bildgeometrie, Gebäudehöhe und unterschiedliche Erfassungsstände sind mögliche
  Ursachen, durch diese Prüfung jedoch nicht einzeln nachgewiesen.

Diese Sichtprüfung ist qualitativ, keine unabhängige annotierte Ground Truth.
Keine fehlende Vorhersage wird als Nachweis einer hindernisfreien Fläche gewertet.

## SITG ist keine vollständige Testannotation

Die exakten ursprünglichen 100-m-Abfragen lieferten 62, 12 und 4 Objekte;
separate Count-Abfragen bestätigten diese Zahlen. Im vergrösserten Bildrahmen
sind es 103, 72 und 16. Der Unterschied ist durch die Abfrageausdehnung bedingt.
Der historische Quellenstand wurde nicht als zeitlich passend zu den Luftbildern
bestätigt. Das WMS-Aktualisierungsdatum ist ausdrücklich **kein Flugdatum**.

Die Flächen-IoU zwischen KI-Hindernissen und SITG innerhalb der Dächer beträgt
0,01660 / 0,01652 / 0,02337. Diese Werte beschreiben lediglich die geringe räumliche
Übereinstimmung zweier unterschiedlich definierter Quellen. Sie sind weder
Genfer Modellgenauigkeit noch Recall. Fehlende, anders klassifizierte und
zeitlich veränderte Objekte können die Abweichung beeinflussen.
Die [SITG-Katalogbeschreibung](https://sitg.ge.ch/donnees/cad-batiment-horsol-toit-sp)
und die [STDL-Untersuchung](https://tech.stdl.ch/PROJ-ROOFTOPS/)
begründen die gesonderte Prüfung der Katasterdaten als Bildlabels.

## Rechenannahmen und verbleibende Grenzen

Inferenz: Konfidenz 0,25; PV-Bildgrösse 512 bei 100-m-Kontext;
Hindernis-Bildgrösse 640 bei 40,96-m-Kontext; überlappende Kacheln.
Fläche: Vereinigungen aller Ausschlüsse, 0,30 m geometrischer Randabstand
plus 0,05 m halbes Bodenpixel. Geneigte Fläche = horizontale Fläche / cos(Neigung),
nur einmal angewendet. 85 % Restflächenquote ist ein konfigurierbares
Modulflächenbudget, kein Nachweis, dass rechteckige Module dort tatsächlich passen.

Die Mindestabstände sind Szenarioparameter, keine bestätigten Bauvorschriften.
Maschinenlesbare Fehler und Warnungen bleiben Teil der Ergebnisse.
Statik, reale Montagebelegung, Glasdächer, lokale Schatten, Brandschutz,
Zugänglichkeit und zeitlich passende Dachinformationen erfordern weitere Prüfung.
Maskenlöcher werden derzeit durch Aussenkonturen gefüllt. Überlappende
Sonnendach-Dachflächen dürfen nicht blind summiert werden.

## Reproduzieren

Siehe `rooftop-pv analyze-roofs --help` und das Beispiel im [README](../README.md).
Vorhandene Ausgabeordner werden nicht überschrieben. Der vollständige lokale
Testlauf nach Implementierung der Batch- und Replay-Befehle umfasste
**243 bestandene Tests bei 82,56 % Coverage**, Ruff und den Lockfile-/Paketcheck.
Der laufende Streamlit-Server beantwortete seinen Health-Endpunkt mit `ok`.
Die UI-Aktionen wurden separat mit Streamlit AppTest geprüft; ein visueller
Browser-Layouttest ist damit nicht behauptet.
