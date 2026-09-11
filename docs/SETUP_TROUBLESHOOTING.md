# Lokaler Start und Fehlerbehebung

## Unterstützte Umgebung

Der Branch verwendet Python **3.12** und `uv`; der exakte Python-Bereich steht
in `pyproject.toml`. GPU-Beschleunigung ist optional. Ein vorhandener
Checkpoint bestimmt nicht automatisch, dass er für die aktuelle Hardware oder
den aktuellen Code validiert ist.

```sh
uv sync --frozen --python 3.12
uv run rooftop-pv doctor
make app
```

Danach ist die lokale Anwendung unter `http://127.0.0.1:8501` erreichbar.
`doctor` dokumentiert Python-/Paketstatus, verfügbaren Beschleuniger und das
lokale Modellregister. Keine Rohdaten, Zugangsdaten oder Modellgewichte in
einen Commit aufnehmen.

## Häufige Probleme

| Symptom | Wahrscheinliche Ursache | Sicherer nächster Schritt |
| --- | --- | --- |
| `uv sync` scheitert | Falsche Python-Version oder unvollständige Tool-Installation | `uv python list` prüfen, Python 3.12 installieren und den Befehl erneut ausführen. |
| `doctor` meldet keinen Checkpoint | Gewichte oder Modellregister fehlen lokal | Manuelle Berechnung verwenden oder den validierten Checkpoint **mit zugehörigem Manifest und SHA256** separat bereitstellen. Keine beliebige `.pt`-Datei als freigegeben ausgeben. |
| App startet nicht auf Port 8501 | Port wird bereits verwendet | Anderen lokalen Prozess beenden oder Streamlit mit einem bewusst gewählten freien Port starten. |
| Adresse/Bild lädt nicht | Temporärer Ausfall, fehlendes Netz oder Quellenlimit | Verbindung und Quelle prüfen; später erneut versuchen. Keine alten Bilder einer neuen Adresse zuordnen. |
| PVGIS-Aufruf scheitert | Netzwerkproblem oder ungültige Koordinaten | WGS84-Koordinaten prüfen; für einen reproduzierbaren Lauf nur bewusst vorhandene Cache-Daten verwenden. |
| CUDA/MPS nicht verfügbar | Treiber, PyTorch-Build oder Hardware unterstützen den Beschleuniger nicht | CPU verwenden oder eine zur Hardware passende PyTorch-Installation nach offizieller PyTorch-Anleitung einrichten. Metriken immer mit Gerät dokumentieren. |
| Installation von Ultralytics/PyTorch scheitert | Plattformabhängige Binärpakete | Zuerst eine neue virtuelle Umgebung über `uv sync` erstellen; keine Paketversionen im Lockfile spontan ersetzen. |
| Tests laden grosse Daten oder Gewichte | Lokale Artefakte wurden nicht vorbereitet | Zuerst die reinen Unit-Tests ausführen; Daten- und Modelltests nur mit dokumentierter lokaler Testbasis starten. |

## Vor einem Fehlerbericht erfassen

```text
Commit-Hash, Betriebssystem, Python-/uv-Version
Ausgabe von `uv run rooftop-pv doctor`
Exakter Befehl und vollständige Fehlermeldung
Ob Netz, lokale Daten und ein Modellregister vorhanden waren
Bei Inferenz: Checkpoint-Hash, Gerät, Bildgrösse und Konfidenzschwelle
```

Personenbezogene Adressen, Bilder oder Zugangsdaten nicht ungeprüft in Issues
oder Logs kopieren.
