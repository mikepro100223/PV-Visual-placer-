# Accessibility-Prüfung

Diese Checkliste gilt für die Streamlit-Oberfläche des aktuellen Branches und
für eine spätere Übernahme in eine andere Oberfläche. Sie ist kein Ersatz für
einen Test mit Menschen, die assistive Technologien nutzen.

## Vor jeder Demo prüfen

- [ ] Die Seite ist ausschliesslich per Tastatur bedienbar: Moduswahl,
  Adresseingabe, Formular-Submit, Datei-Upload, Parameter, Berechnung und
  Exporte sind erreichbar.
- [ ] Der Tastaturfokus ist jederzeit sichtbar; ein Dialog, Dateidialog oder
  Fehlerhinweis lässt den Fokus nicht verschwinden.
- [ ] Alle Eingaben haben verständliche Labels. Platzhalter sind nur Beispiele,
  keine einzige Beschriftung.
- [ ] Fehlermeldungen nennen Ursache und nächsten Schritt; Farbe oder ein Icon
  ist nicht die einzige Information.
- [ ] Ergebniszahlen enthalten Einheiten, etwa `kWh`, `kWp`, `m²` und Grad.
- [ ] Wichtige Kartenfarben, Segmentierungen und Statusmeldungen haben eine
  Textalternative; auch bei Farbsehschwäche bleibt der Unterschied erkennbar.
- [ ] Bilder und Overlays besitzen einen aussagekräftigen Alternativtext oder
  eine gleichwertige tabellarische Zusammenfassung.
- [ ] Text und interaktive Elemente erfüllen mindestens WCAG-AA-Kontrast. Die
  UI-Orientierung für diesen Stand empfiehlt dunkles Grün auf sehr hellem Grün
  oder Weiss, sichtbare Fokuszustände und keine Information nur in Solar-Gelb.
- [ ] Bei 200 % Browser-Zoom und auf 375 px Breite gehen keine Bedienfelder,
  Tabellen oder Downloads verloren.
- [ ] Bewegte Ladeindikatoren sind nicht für das Verständnis allein nötig;
  der Zustand ist auch als Text verfügbar.

## Testablauf

1. Browser-Zoom auf 200 % setzen und die ganze Anwendung nur mit `Tab`,
   `Shift+Tab`, Leertaste und Enter bedienen.
2. Einen absichtlichen Fehler erzeugen (ungültiges GeoJSON oder fehlende
   Koordinaten) und prüfen, ob Nachricht und Fokus verständlich bleiben.
3. Ein fertiges Szenario mit einem Screenreader testen; mindestens Überschriften,
   Formlabels, Modellstatus, Resultate und Download-Schaltflächen vorlesen
   lassen.
4. In einem kurzen Protokoll Datum, Revision, Browser, Hilfstechnologie und
   offene Befunde festhalten.

## Prioritäten

**Blocker:** Nicht per Tastatur erreichbare Berechnung/Downloads, fehlende
Formlabels, unverständliche Fehler oder eine nur farbcodierte Sicherheitswarnung.

**Vor Veröffentlichung beheben:** unzureichender Kontrast, fehlende Bild-
Alternativen, nicht responsives Layout und unklare Einheiten.

**Später verbessern:** verkürzte Texte, zusätzliche Sprachen und optimierte
Screenreader-Reihenfolge für komplexe Karten-Overlays.
