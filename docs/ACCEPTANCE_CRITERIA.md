# Abnahmekriterien

Dieses Dokument definiert überprüfbare Kriterien für einen lauffähigen
Projektstand. Es ersetzt keine technische Installationsfreigabe. Die Kriterien
gelten pro getesteter Revision, Modell-Checkpoint und Testumgebung.

## Abnahmeumfang

| Bereich | Muss erfüllt sein | Nachweis |
| --- | --- | --- |
| Start | `uv sync --frozen --python 3.12`, `uv run rooftop-pv doctor` und `make app` laufen ohne manuelle Codeänderung. | Terminal-Ausgabe mit Versionen und Commit-Hash |
| Grundfunktionen | Adresse suchen, Dachfläche laden, manueller Modus, Szenario und JSON-/CSV-Export funktionieren. | Ausgefülltes Prüflaufprotokoll |
| Geometrie | Bestehende PV, Hindernisse und manuelle Ausschlussflächen werden vereinigt, auf die Dachfläche begrenzt und nicht doppelt abgezogen. | `tests/test_geometry.py`, `tests/test_roofs_bbox.py` und sichtbare Prüfung einer Beispieladresse |
| PV-Modell | Modellstatus, Checkpoint-Hash, Testsplit und Konfidenz sind sichtbar. Fehlende oder nicht bestandene Modelle werden nicht als freigegeben dargestellt. | Modellregister und `docs/MODEL_CARD_PV.md` |
| Hindernisse | Nicht erkannte Hindernisse werden nicht als freie Fläche ausgegeben. Bis ein Schweizer Modell unabhängig geprüft ist, bleibt die Sichtprüfung obligatorisch. | Modellstatus, manuelle Ausschlussfläche und `docs/MODEL_CARD_OBSTACLES.md` |
| Ertrag | Wetterquelle, Zeitraum, Neigung, Azimut, Modul- und Verlustannahmen werden im Export mitgeführt. | Export-JSON und `tests/test_physics.py` |
| Fehlverhalten | Netz-, Quellen-, Modell- und Geometriefehler erklären den nächsten sicheren Schritt und hinterlassen kein Ergebnis eines alten Kontexts. | `tests/test_app.py`, manueller Negativtest |
| Accessibility | Die Checkliste in `ACCESSIBILITY.md` wurde für den getesteten Stand ausgefüllt; kritische Blocker sind behoben oder sichtbar dokumentiert. | Abgehakte Checkliste mit Datum/Revision |
| Leistung | Ein Baseline-Lauf wurde mit `scripts/benchmark_runtime.py` auf der Zielhardware gespeichert. Verschlechterungen von mehr als 25 % werden begründet. | JSON-Bericht unter `artifacts/benchmarks/` (nicht committen) |

## Prüfszenarien

Für jeden Release-Kandidaten wird mindestens je ein Fall aus diesen Gruppen
protokolliert. Adressen und Luftbilder dürfen nur verwendet werden, wenn ihre
Nutzungsbedingungen das erlauben.

| ID | Szenario | Erwartetes Ergebnis |
| --- | --- | --- |
| A1 | Adresse mit mehreren Sonnendach-Dachflächen | Gewählte Dachfläche, Bild und Berechnung beziehen sich auf dieselbe Fläche. |
| A2 | Dach mit bestehender PV | Maskierte, auf das Dach geclipte PV-Fläche reduziert die freie Fläche; keine Aussage, dass eine nicht erkannte Anlage nicht existiert. |
| A3 | Dach mit Kamin, Dachfenster oder manuellem Polygon | Ausschluss wird nur einmal abgezogen und im Flächenledger sichtbar. |
| A4 | Leeres oder nicht verfügbares Modellregister | Manuelles Szenario ist möglich; eine Modellmessung wird nicht vorgetäuscht. |
| A5 | Ungültiges oder zu grosses GeoJSON | Die Eingabe wird abgewiesen; kein vorheriges Szenario bleibt exportierbar. |
| A6 | Fehlender Netzwerkzugriff | Die App zeigt einen verständlichen Fehler; bereits berechnete Ergebnisse werden nicht einer neuen Anfrage zugeschrieben. |
| A7 | Tastaturbedienung | Alle Eingaben, Formulare, Downloads und die Moduswahl sind per Tastatur erreichbar und der Fokus bleibt sichtbar. |

## Entscheidungsregel

Ein Stand ist **demofähig**, wenn Start, Grundfunktionen, Geometrie,
Fehlverhalten und Accessibility ohne kritischen Befund erfüllt sind. Er ist
**nicht installationsreif**, solange die Modellkarten eine experimentelle
Qualität ausweisen oder die lokalen Gegebenheiten nicht fachlich geprüft wurden.

## Protokollvorlage

```text
Datum / Revision:
Prüfende Person / Hardware:
PV-Checkpoint und SHA256:
Hindernis-Checkpoint und SHA256:
Ausgeführte Szenarien (A1–A7):
Bestandene / nicht bestandene Kriterien:
Bekannte Einschränkungen:
Verweis auf Benchmark-JSON:
```
