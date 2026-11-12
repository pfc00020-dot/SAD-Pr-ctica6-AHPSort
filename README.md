# Práctica 6 · AHPSort II — Nivel de seguridad de las provincias españolas

Sistemas de Ayuda a la Decisión · Universidad de Jaén · Pablo Fregenal Castro

Clasificación de provincias en tres niveles de seguridad (Alta, Media, Baja) con **AHPSort II** a partir de
5 criterios de criminalidad (crímenes, robos, vehículos, orden público y drogas).

| Ejercicio | Qué resuelve |
|---|---|
| 1 | 8 provincias andaluzas con hechos absolutos 2017 (pesos AHP, Tabla 7, clasificación y mapa) |
| 2 | Andalucía normalizada por 10.000 habitantes (RP/CP recalculados, clasificación y mapa) |
| 3 | 50 provincias españolas 2017 (verificación de la normalización, RP/CP nacionales, mapa) |
| 4 | 2023 con la calibración de 2017 y truncamiento; mapas 2017/2023, tabla de transición y matriz de cambios |

## Cómo ejecutarlo

**Google Colab (sin instalar nada):** abrir
<https://colab.research.google.com/github/pfc00020-dot/SAD-Pr-ctica6-AHPSort/blob/main/Practica6_AHPSort.ipynb>
y pulsar *Entorno de ejecución → Ejecutar todas*. La primera celda clona este repositorio.

**En local (Python 3.10+):**
```bash
git clone https://github.com/pfc00020-dot/SAD-Pr-ctica6-AHPSort.git
cd SAD-Pr-ctica6-AHPSort
pip install -r requirements.txt
python ahpsort.py                 # imprime todas las tablas y guarda CSV y PNG en resultados/
jupyter notebook Practica6_AHPSort.ipynb   # versión paso a paso
```

## Estructura
- `ahpsort.py` — implementación (pesos AHP, funciones de prioridad lineales a trozos, asignación por perfiles centrales, mapas).
- `Practica6_AHPSort.ipynb` — cuaderno con los 4 ejercicios ya ejecutados.
- `datos/` — ficheros del enunciado (Andalucía.xls, habitantes.xlsx, tasas nacionales 2017 y 2023).
- `mapas/` — provincias en GeoJSON simplificado (IGN y Junta de Andalucía, de `Mapas.rar`; Canarias añadida desde un GeoJSON público de provincias porque el shapefile del IGN solo trae península y Baleares).
- `resultados/` — tablas CSV y figuras generadas.
