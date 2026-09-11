# Fehleranalyse und Modellgrenzen

Die Anwendung schätzt zusätzlich planbare PV-Fläche. Sie liefert weder einen
baulichen Nachweis noch eine Installationsfreigabe. Dieses Dokument trennt
Messfehler, unsichere Annahmen und offene Entwicklungsrisiken.

## Fehlerkette

```text
Orthofoto / Geodaten → Lage- und Dachzuordnung → Segmentierung →
Geometrische Ausschlüsse → Modulbelegung → Wetter- und Ertragsszenario
```

Ein Fehler in einer frühen Stufe kann alle späteren Werte beeinflussen. Daher
ist eine präzise kWh-Ausgabe keine Aussage über die Genauigkeit der verfügbaren
Dachfläche.

| Stufe | Bekannte Fehlerquelle | Wirkung | Umgang im Projekt |
| --- | --- | --- | --- |
| Quellen | Unterschiedliche Aufnahmezeitpunkte von Orthofoto und Kataster | PV oder Umbauten stimmen zeitlich nicht überein. | Quellenzeit sichtbar machen; Sichtprüfung verlangen. |
| Lage | Versatz zwischen Bild, Sonnendach und Modellmaske | Maske wird am falschen Dach oder ausserhalb eines Hindernisses abgezogen. | Aktive Offset-Korrektur separat gegen unabhängige Referenzen prüfen; vorher/nachher dokumentieren. |
| PV-Segmentierung | Dunkle Dächer, kleine Anlagen und verdeckte Module | Bestehende PV wird übersehen oder freie Dachfläche wird zu stark reduziert. | Schweizer PV-Modell bleibt experimentell; Masken visuell prüfen. |
| Hindernisse | RID2 stammt nicht aus der Schweiz; seltene Klassen und kleine Objekte sind schwach belegt | Kamin, Fenster, Baum oder Schatten können fehlen oder falsch markiert werden. | Neue Hinderniserkennung auf Schweizer Referenzen getrennt evaluieren; manuelle Polygone zulassen. |
| Dachgeometrie | Vereinfachte, überlappende oder falsch zugeordnete Dachflächen | Falsche Planfläche, Neigung oder Azimut. | Dachfläche explizit wählen, Polygone clippen und überlappende Abzüge vereinigen. |
| Modulbelegung | Annahmen zu Abstand, Orientierung, Zugangsweg und Statik | Modulzahl kann fachlich unzulässig sein. | Annahmen im Export offenlegen; keine Normkonformität behaupten. |
| Ertrag | TMY-Wetter statt aktuellem Jahr; fehlende lokale Verschattung | Jahresertrag ist ein Szenario, keine Prognose. | PVGIS-Quelle und Verluste ausweisen; Horizont-/Objektschatten als Grenze nennen. |

## Aktive Risiken und Prüfschritte

1. **Offset-Korrektur:** Vor der Übernahme muss sie für Dächer mit verschiedenen
   Ausrichtungen, Bildquellen und Dachgrössen überprüft werden. Metriken sind
   Masken-IoU, Distanz Maske–Dachrand, visuelle Overlays und die Änderung der
   resultierenden freien Fläche. Ein einzelnes gelungenes Overlay genügt nicht.
2. **Hindernistraining:** Die unabhängige Schweizer Prüfung muss Nicht-PV-
   Hindernisse getrennt von PV messen. Fehlen positive Testbeispiele einer
   Klasse, darf für sie keine Leistung behauptet werden.
3. **Regressionen:** Jeder Fix erhält mindestens einen Gegenfall: korrekt
   ausgerichtetes Dach, absichtlich versetzte Maske und eine überlappende
   Ausschlussfläche. Die Kriterien stehen in `ACCEPTANCE_CRITERIA.md`.

## Meldung eines Fehlers

Eine verwertbare Meldung enthält Revision, Modell-/Checkpoint-Hash, Quelle und
Zeitpunkt des Bildes, Dach-ID oder anonymisierte Geometrie, Konfidenzschwelle,
Overlay sowie die erwartete und beobachtete Wirkung. Eine niedrige Modell-
Konfidenz ist kein Beweis für einen Fehler; umgekehrt ist hohe Konfidenz kein
Beweis für korrekte Geometrie.
