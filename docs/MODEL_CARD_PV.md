# Schweizer PV-Modell: gemessener Stand am 10. September 2026

**Status: trainiert, experimentell.** Zwei von drei vorab definierten
Hackathon-Grenzwerten werden auf dem Testsatz erreicht. Die Genauigkeit der
Maskengrenzen reicht noch nicht für eine ungeprüfte Flächenplanung.

## Modell und Daten

- Ultralytics YOLO11n-seg, eine Klasse `solar_panel`.
- Start: offizielle vortrainierte YOLO11-Segmentierungsgewichte, anschliessend
  echtes Fine-Tuning auf dem Schweizer Kaggle-Datensatz v1.
- 758 Luftbilder mit binären PV-Masken: 608 Training, 80 Validierung, 70 Test.
- 76 geografische Kilometertile-Gruppen; Bilder desselben Gebiets und andere
  Aufnahmejahre desselben Tiles bleiben im gleichen Split.
- 50 Epochen, 512 Pixel, Batch 8, AdamW, Seed 42, Apple M4 Max / MPS.
- Lauf `yolo11-pv-20260910T115659346108Z`, abgeschlossen nach rund 17 Minuten.
- Checkpoint-SHA256: `15684cd04eca6d1282b99c2190355608ed65ffccf7c28b25dafd2ec051a9d215`.
- Binäre zusammenhängende Flächen werden zu Instanzen konvertiert. Eine Instanz
  ist damit nicht zwingend ein einzelnes physisches Modul. Originalmasken mit
  ihren Löchern bleiben die Referenz für belegte Fläche und Union-IoU.

## Unberührter Testsatz

Auswertung auf 70 Bildern aus zurückgehaltenen geografischen Gruppen, davon
36 mit PV und 34 ohne annotierte PV. Testinferenz auf CPU, Bildgrösse 512;
Flächenberechnung bei Konfidenz 0.25, AP-Kurven mit Sammelschwelle 0.001.

| Messgrösse | Testwert | Vorab-Grenzwert |
| --- | ---: | ---: |
| Masken-mAP50 | 0.7791 | ≥ 0.70: erreicht |
| Masken-mAP50–95 | 0.4288 | ≥ 0.45: nicht erreicht |
| Union-IoU gegen Originalmasken | 0.6830 | ≥ 0.65: erreicht |
| Mittlere IoU positiver Bilder | 0.6209 | Diagnose |
| Bilder ohne PV mit mindestens einem Fehlalarm | 4 / 34 | Diagnose |
| Gesamte vorhergesagte PV-Fläche relativ zur Referenz | +24.6 % | Diagnose |

Die Flächenüberschätzung ist über diesen Testsatz aggregiert, kein pauschaler
Korrekturfaktor für neue Dächer. mAP ist eine Segmentierungsmetrik, keine
„77.9 % Genauigkeit“ für Ertrag oder Dachfläche. Der kleine Testsatz ersetzt keine
Validierung für weitere Regionen, Jahreszeiten und Bildquellen.

Maschineller Bericht einschliesslich Einzelfällen:
`artifacts/evaluations/yolo11-pv-20260910T115659346108Z-test-20260910T133345773191Z/report.json`.
Fehlerbilder liegen daneben unter `review/`, Referenz links / Vorhersage rechts.
Das Modellregister `artifacts/models/current.json` meldet
`experimental_gates_failed` und verweist auf diesen Bericht.

## Entscheidungen vor dem Test

Die Validierung bei 512 Pixeln ergab Masken-mAP50 0.7705, mAP50–95 0.4509 und
Union-IoU 0.6587. Eine reine Erhöhung der Inferenz auf 1024 Pixel verschlechterte
mAP50–95 auf 0.3638 und IoU auf 0.6218. Deshalb bleiben Bildgrösse 512 und
Konfidenz 0.25 der festgelegte Betriebspunkt. Der Testsatz wurde nicht zur
Schwellen- oder Auflösungswahl verwendet.

Die oben angegebenen Flächenwerte wurden nach einer im unbeschrifteten
Brugg-Prüflauf gefundenen Konturkorrektur neu berechnet: getrennte Maskeninseln
werden nicht mehr künstlich verbunden. Gewichte und Betriebspunkt blieben gleich;
native mAP-Werte sind unverändert. Die frühere Test-IoU war 0.6821, jetzt 0.6830.
Die erneut berechnete Validierungs-IoU beträgt 0.6590. Details: [METHODS.md](METHODS.md).

## Beobachtete Grenzen und nächster sinnvoller Schritt

- Dunkle Dachflächen werden teilweise mit PV verwechselt.
- Kleine Anlagen sind schwieriger: mittlere positive Test-IoU 0.5694 bei weniger
  als 1 % Bildbelegung, gegenüber 0.8011 bei grösserer Belegung.
- Das Modell wurde nicht auf Kamine, Bäume, Schnee, Schatten oder Dachneigung
  trainiert. Hindernisse kommen aus einem getrennten Modell bzw. Vektordaten.
- Orthofoto-Versatz und unbekannte Aufnahmejahre beeinflussen die Zuordnung zu
  Sonnendach-Flächen. Eine nicht erkannte Anlage beweist keine freie Fläche.
- Sinnvolle Verbesserung: zusätzliche geografisch unabhängige Schweizer Labels,
  kontrollierte Prüfung dunkler Dächer und kleiner PV-Flächen sowie ein eigenes
  Training mit höherer Detailauflösung. Nicht einfach denselben Lauf wiederholen.

Für neue Modelle bleiben neue externe Prüfgebiete notwendig; der hier bereits
ausgewertete Testsatz darf nicht unbemerkt zum Trainingsmaterial werden.

## Verwendung und Lizenz

Das lokale Dashboard lädt diesen Checkpoint und kennzeichnet ihn als experimentell.
Masken und resultierende Ausschlussflächen müssen visuell kontrolliert werden.
PV-Ertragswerte sind Szenarien mit den separat dokumentierten Wetter-, Geometrie-
und Verlustannahmen, keine Installationsfreigabe.

Kaggle deklariert die Daten als CC0; genaue Provenienz siehe [DATA.md](DATA.md).
Ultralytics und abgeleitete Gewichte unterliegen eigenen Lizenzbedingungen
(AGPL-3.0 bzw. Enterprise). Die Apache-Projektlizenz ersetzt diese nicht.
