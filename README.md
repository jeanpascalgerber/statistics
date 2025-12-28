# Aurora Forecast Pro - Analyse und Vorhersage geomagnetischer Stürme

Dieses Repository beinhaltet die Projektarbeit im Modul Statistik 2. Das Projekt untersucht die Vorhersagbarkeit von geomagnetischen Stuermen basierend auf Sonnenwind-Parametern mithilfe von Machine Learning Methoden.

## Projektstruktur und Abgabe

Der Code ist in zwei Hauptkomponenten unterteilt:

1.  **Analyse & Modellierung (Jupyter Notebook):**
    Pfad: **`data/gerber_jp_nordlichter.ipynb`**
    * Beinhaltet das Laden und Bereinigen der Daten.
    * Zeigt die explorative Datenanalyse.
    * Dokumentiert das Training und die Evaluation des Random Forest Modells.

2.  **Dashboard Applikation (Python Skript):**
    Pfad: **`data/app.py`**
    * Beinhaltet den Code für das interaktive Streamlit-Dashboard.
    * Verbindet das Modell mit Live-Daten der NOAA.

## Installation und Ausführung

**Voraussetzung:** Python 3.8 oder höher.

**1. Repository klonen**
Laden Sie das Repository auf Ihren lokalen Rechner.

**2. Abhängigkeiten installieren**
Navigieren Sie in den Ordner `data` und installieren Sie die benoetigten Bibliotheken (oder nutzen Sie die Installation im Notebook):
```bash
pip install streamlit pandas numpy scikit-learn plotly requests urllib3

3. Notebook ansehen (Statistik-Teil) Öffnen Sie data/gerber_jp_nordlichter.ipynb, um die statistische Herleitung und das Modelltraining nachzuvollziehen.

4. Applikation starten (Validierung) Um das Dashboard zu starten, öffnen Sie ein Terminal im Ordner data und fuehren Sie folgenden Befehl aus:

Bash
streamlit run app.py

Methodik
Datenbasis: NASA OMNIWeb Datensatz (Datei: omni_data.txt, im Repository enthalten).

Modell: Random Forest Classifier (Scikit-learn).

Zielvariable: Kp-Index >= 5 (Geomagnetischer Sturm) mit einem Vorhersagehorizont von 3 Stunden.

Validierung: Vergleich der Modell-Vorhersagen mit Live-Messdaten (Ground Truth) des NOAA Space Weather Prediction Center.

Autor
J.P. Gerber Statistik 2 Projekt | Herbstsemester 2025