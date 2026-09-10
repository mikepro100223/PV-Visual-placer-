# RID2-Hindernismodell

Status dieses Experiments: trainiert, Qualitätsgrenzwerte nicht erreicht.
Die vorhandene Schweizer PV-Segmentierung ist ein separates Modell.

## Experiment

YOLO11n-seg wird mit elf Klassen auf dem dachzentrierten RID2-Teil trainiert.
Lauf: `yolo11-obstacles-20260910T122307677724Z`. Abgeschlossen: 40 Epochen,
640 Pixel, Batch 8, AdamW, Seed 42 und Early Stopping nach zehn Epochen ohne
Verbesserung. Training auf Apple M4 Max / MPS, Dauer rund 65 Minuten.
Die native Validierungs-Fitness wählte Epoche 39. Checkpoint-SHA256:
`d5924ee6e911d9ed9f81eed6a087e1018e83184b6cbf2511e15a16bbb82022a0`.
Gespeicherter Verlauf:
`artifacts/training/yolo11-obstacles-20260910T122307677724Z/results.csv`.

| Klasse | Trainingspolygone | Validierung | Test |
| --- | ---: | ---: | ---: |
| solar_panel | 4528 | 2117 | 79 |
| chimney | 932 | 3486 | 175 |
| skylight | 2183 | 1382 | 57 |
| dormer | 4267 | 1723 | 162 |
| roof_window | 3182 | 2146 | 244 |
| hvac | 1827 | 352 | 22 |
| tv_dish | 109 | 55 | 3 |
| ladder | 124 | 157 | 0 |
| balcony | 1693 | 87 | 0 |
| wall | 41 | 19 | 0 |
| other | 17649 | 9022 | 519 |

Das sind vorbereitete Polygonzahlen vor den 106 vom Ultralytics-Scanner entfernten
Balcony-Duplikaten im Training. Bildzahl: 1035 / 680 / 104. Vier räumlich getrennte
Gruppen, davon zwei im Training und je eine in Validierung und Test. Viele
dachzentrierte Ausschnitte innerhalb einer Gruppe überlappen; die Bildzahl ist
deshalb keine Anzahl unabhängig beobachteter Standorte.

## Finale Messung

Fester Betriebspunkt: 640 Pixel und Konfidenz 0.25 für Flächen; native AP-Kurven
mit Sammelschwelle 0.001. Test auf CPU, Validierung auf MPS. Die 104 Testbilder
aus einer zurückgehaltenen Region wurden nicht für Training, Epochenwahl oder
Schwellenoptimierung verwendet. 98 Bilder enthalten annotierte Hindernisse.

| Messgrösse | Validierung (680 Bilder) | Test (104 Bilder) | Grenzwert |
| --- | ---: | ---: | ---: |
| Masken-mAP50 | 0.2491 | 0.2979 | ≥ 0.70 |
| Masken-mAP50–95 | 0.1171 | 0.1702 | ≥ 0.45 |
| Union-IoU aller Klassen | 0.4887 | 0.3699 | ≥ 0.65 |
| Union-IoU nur Hindernisse | 0.3398 | 0.1706 | ≥ 0.65 |
| Aggregierter Hindernis-Flächenfehler | +0.9 % | +68.8 % | Diagnose |

Keiner der Qualitätsgrenzwerte wird erreicht. Die nahezu ausgeglichene gesamte
Validierungsfläche bedeutet keine korrekten Masken: räumliche Fehlalarme und
übersehene Bereiche gleichen sich teilweise aus. Im Test ist die Flächenbilanz
deutlich anders. Ein globaler Korrekturfaktor ist daraus nicht ableitbar.

Native Test-AP50 wird nur über die acht Klassen mit positiven Testlabels gemittelt,
Validierungs-AP über elf Klassen. Die beiden Mittelwerte sind deshalb auch in ihrer
Klassenzusammensetzung verschieden. Test-AP50: PV 0.5864, Kamin 0.1466,
Skylight 0.0728, Gaube 0.2027, Dachfenster 0.5457, HVAC 0.7697, TV-Schüssel 0.0000,
`other` 0.0588. HVAC hat nur 22 Testpolygone, TV-Schüssel nur drei.

Berichte mit Klassenwerten, Einzelfällen und Fehlerbildern:

- `artifacts/evaluations/yolo11-obstacles-20260910T122307677724Z-test-20260910T133346930101Z/report.json`
- `artifacts/evaluations/yolo11-obstacles-20260910T122307677724Z-val-20260910T133348101476Z/report.json`

Die Flächenwerte enthalten bereits die Korrektur künstlich verbundener Maskeninseln.
Frühere Berichte bleiben als Diagnose erhalten. Checkpoint, Konfidenz und Bildgrösse
wurden nicht verändert; siehe [METHODS.md](METHODS.md).

## Aussagegrenzen

- Nicht mit dem separaten vollständigen RID2-Teil mit 50 Regionen verwechseln.
- Die Verteilung der Klassen unterscheidet sich stark zwischen den Regionen.
  Für `ladder`, `balcony` und `wall` enthält dieser Testsatz keine positiven Labels.
  Für diese Klassen kann daraus keine Testleistung bestätigt werden.
- Die Klasse `other` ist ein explizites Sammellabel der Quelle, keine nachträglich
  erfundene Objektart. Typenverwechslungen und fehlerhafte Grenzen bleiben möglich.
- Bäume und Schatten sind nicht vollständig annotiert. Nicht erkannte oder nicht
  gelistete Hindernisse beweisen keine unbebaute, unverschattete Fläche.
- Niederländische Luftbilder ersetzen keine Validierung auf Schweizer Dächern.
  Erste rein qualitative Brugg-Prüfungen eines Zwischencheckpoints zeigen auch
  Fehlalarme auf Bäumen, Schatten und Flächen ausserhalb des Dachs. Dach-Clipping
  entfernt externe Masken, aber keine Fehlalarme innerhalb der Dachfläche.
- Das Dashboard verwendet nur Nicht-PV-Klassen dieses Modells als Hindernisse.
  Die Klasse `solar_panel` ersetzt nicht die Schweizer PV-Spezialsegmentierung.
- Georeferenzierte Inferenz erfolgt in überlappenden 40.96-m-Ausschnitten.
  Polygon-Überlappungen werden für Flächen unioniert; die Maskenzahl zählt keine
  unabhängigen physischen Hindernisse.

## Verwendung

`artifacts/models/obstacles.json` verweist auf den tatsächlichen Checkpoint
mit Status `experimental_gates_failed`. Testevaluation ist ein eigener Schritt und
überschreibt nicht das Schweizer Modellregister. Masken sind Planungsentwürfe,
keine bestätigten Installationshindernisse, und benötigen eine Sichtprüfung.

## Fehleranalyse und nächste Datenarbeit

Der korrigierte Zwischenbericht auf 680 Validierungsbildern
(`obstacles-intermediate-val-20260910T125614610660Z`) zeigte eine Hindernis-Union-IoU
von 0.194 und rund 55 % zu wenig vorhergesagte Hindernisfläche. Das sind Werte
eines Zwischencheckpoints, keine finalen Testwerte. In den geprüften Fehlerbildern
wurden kleine Kamine übersehen und teilweise grosse Dachregionen falsch markiert.
Die Transformprüfung fand dagegen keinen räumlichen Label-/Bildversatz.

Sinnvolle Folgearbeit vor einem weiteren langen Training:

1. Geografisch vielfältigere Trainingsregionen und Schweizer/Genfer Bildlabels
   ergänzen; die aktuelle Validierung besteht aus einer anderen Region.
2. Kleine und seltene Objektarten gezielt annotieren; neue unabhängige Prüfsplits
   müssen auch positive Beispiele der zu bewertenden Klassen enthalten.
3. Detailausschnitte in passender Bodenauflösung vergleichen, statt bloss die
   Gesamtbildauflösung für das unveränderte Modell zu erhöhen.
4. Die breiten Klassen `other` und `wall` sowie Balcony-Duplikate prüfen; einen
   klassenübergreifenden Hindernisflächen-Ansatz getrennt von der Typisierung testen.

Die vorhandenen Schweizer PV-Labels dürfen dabei weiterhin nicht als vollständige
negative Hindernislabels verwendet werden. Ein ausgewerteter Testsatz wird nicht
nachträglich zum Trainings- oder Schwellenoptimierungsdatensatz.

Quellen, CC-BY-4.0-Attribution und Transformprüfung: [OBSTACLE_DATA.md](OBSTACLE_DATA.md).
Ultralytics-Lizenzbedingungen gelten zusätzlich zur bestehenden Projektlizenz.
