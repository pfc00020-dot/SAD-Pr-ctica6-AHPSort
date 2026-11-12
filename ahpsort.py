"""
Práctica 6 - Sistemas de Ayuda a la Decisión (UJA, curso 2026/27)
AHPSort II aplicado a la clasificación de provincias por nivel de seguridad.

Autor: Pablo Fregenal Castro

Contenido:
  * pesos_ahp(A)            -> pesos de los criterios por el método del autovalor + consistencia
  * FuncionPrioridad        -> función de prioridad local lineal a trozos (AHPSort II)
  * ModeloAHPSort           -> pesos + funciones + perfiles centrales -> clasificación
  * ejercicio1..ejercicio4  -> resolución de los 4 ejercicios del enunciado
  * mapa_andalucia / mapa_espana -> mapas coropléticos de la clasificación

Ejecución:  python ahpsort.py        (resuelve todo y guarda tablas y figuras en resultados/)
            o abrir Practica6_AHPSort.ipynb (Jupyter / Google Colab)
"""
import os
import unicodedata

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(BASE, 'datos')
MAPAS = os.path.join(BASE, 'mapas')
RESULTADOS = os.path.join(BASE, 'resultados')

CRITERIOS = ['Crímenes', 'Robos', 'Vehículos', 'Orden público', 'Drogas']
CLASES = ['Alto', 'Medio', 'Bajo']             # nivel de SEGURIDAD: CP1, CP2, CP3
COLORES = {'Alto': '#2E7D32', 'Medio': '#F9A825', 'Bajo': '#C62828'}

# ----------------------------------------------------------------------------------------
# 1. Pesos de los criterios (Tabla 4) por el método del autovalor
# ----------------------------------------------------------------------------------------
A_CRITERIOS = np.array([
    [1,     5,   3,   7,   3],
    [1/5,   1,   3,   1/2, 1/4],
    [1/3,   1/3, 1,   1/5, 1/5],
    [1/7,   2,   5,   1,   1/5],
    [1/3,   4,   5,   5,   1],
])


def indice_aleatorio(n):
    """IA usado en clase: 1,98·(n−2)/n."""
    return 1.98 * (n - 2) / n if n > 2 else 0.0


def pesos_ahp(A):
    """Vector propio principal normalizado, λmáx, IC, IA y RC de una matriz de comparaciones."""
    A = np.asarray(A, float)
    n = len(A)
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    lam = float(vals.real[k])
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    ic = (lam - n) / (n - 1)
    ia = indice_aleatorio(n)
    return {'pesos': w, 'lambda_max': lam, 'IC': ic, 'IA': ia, 'RC': ic / ia,
            'reciproca': bool(np.allclose(A * A.T, 1))}


# ----------------------------------------------------------------------------------------
# 2. Función de prioridad local (AHPSort II): lineal a trozos entre los puntos conocidos
# ----------------------------------------------------------------------------------------
class FuncionPrioridad:
    """Une con segmentos los puntos (x, prioridad) de los RP (y CP si se conocen).
    Fuera de [x_min, x_max] se trunca: se usa la prioridad del extremo (RP1 o RP6)."""

    def __init__(self, x, y):
        orden = np.argsort(x)
        self.x = np.asarray(x, float)[orden]
        self.y = np.asarray(y, float)[orden]

    def __call__(self, valor):
        return np.interp(valor, self.x, self.y)   # np.interp ya trunca en los extremos

    def tramo(self, valor):
        """Devuelve los dos puntos entre los que se interpola (para explicar el cálculo)."""
        v = min(max(valor, self.x[0]), self.x[-1])
        i = int(np.searchsorted(self.x, v, side='right')) - 1
        i = min(max(i, 0), len(self.x) - 2)
        return (self.x[i], self.y[i]), (self.x[i + 1], self.y[i + 1])


class ModeloAHPSort:
    def __init__(self, pesos, funciones, cp_x, cp_local=None):
        """pesos: vector (5); funciones: {criterio: FuncionPrioridad};
        cp_x: DataFrame 3x5 con los perfiles centrales (filas CP1..CP3);
        cp_local: prioridades locales de los CP si vienen del cuestionario (si no, se interpolan)."""
        self.pesos = pd.Series(pesos, index=CRITERIOS)
        self.funciones = funciones
        self.cp_x = cp_x
        if cp_local is None:
            cp_local = pd.DataFrame({c: [funciones[c](v) for v in cp_x[c]] for c in CRITERIOS}, index=cp_x.index)
        self.cp_local = cp_local
        self.cp_global = (self.cp_local[CRITERIOS] * self.pesos).sum(axis=1)

    def tabla_cp(self):
        t = self.cp_local.copy()
        t['Global'] = self.cp_global
        return t

    def prioridades(self, datos):
        loc = pd.DataFrame({c: self.funciones[c](datos[c].values) for c in CRITERIOS}, index=datos.index)
        loc['Global'] = (loc[CRITERIOS] * self.pesos).sum(axis=1)
        return loc

    def asignar(self, p, vision='pesimista'):
        """Asignación por perfiles centrales (diap. 164): clase del CP global más cercano.
        p >= cp1 -> Alto ; p < cp3 -> Bajo ; empate -> según visión optimista/pesimista."""
        cp = self.cp_global.values
        if p >= cp[0]:
            return CLASES[0]
        if p < cp[-1]:
            return CLASES[-1]
        for i in range(len(cp) - 1):
            if cp[i + 1] <= p < cp[i]:
                d_sup, d_inf = cp[i] - p, p - cp[i + 1]
                if np.isclose(d_sup, d_inf):
                    return CLASES[i] if vision == 'optimista' else CLASES[i + 1]
                return CLASES[i] if d_sup < d_inf else CLASES[i + 1]

    def clasificar(self, datos):
        r = self.prioridades(datos)
        r['Nivel de seguridad'] = [self.asignar(p) for p in r['Global']]
        return r


# ----------------------------------------------------------------------------------------
# 3. Datos y utilidades
# ----------------------------------------------------------------------------------------
# código INE de provincia -> nombre canónico y variantes que aparecen en los ficheros
PROVINCIAS = {
    '01': ('Araba/Álava', ['alava', 'araba']), '02': ('Albacete', []), '03': ('Alicante', ['alicante/alacant']),
    '04': ('Almería', []), '05': ('Ávila', []), '06': ('Badajoz', []), '07': ('Illes Balears', ['baleares', 'balears']),
    '08': ('Barcelona', []), '09': ('Burgos', []), '10': ('Cáceres', []), '11': ('Cádiz', []),
    '12': ('Castellón', ['castellon/castello']), '13': ('Ciudad Real', []), '14': ('Córdoba', []),
    '15': ('A Coruña', ['coruna']), '16': ('Cuenca', []), '17': ('Girona', []), '18': ('Granada', []),
    '19': ('Guadalajara', []), '20': ('Gipuzkoa', []), '21': ('Huelva', []), '22': ('Huesca', []),
    '23': ('Jaén', []), '24': ('León', []), '25': ('Lleida', []), '26': ('La Rioja', ['rioja']),
    '27': ('Lugo', []), '28': ('Madrid', []), '29': ('Málaga', []), '30': ('Murcia', []), '31': ('Navarra', []),
    '32': ('Ourense', []), '33': ('Asturias', []), '34': ('Palencia', []), '35': ('Las Palmas', ['palmas']),
    '36': ('Pontevedra', []), '37': ('Salamanca', []), '38': ('Santa Cruz de Tenerife', ['santa cruz']),
    '39': ('Cantabria', []), '40': ('Segovia', []), '41': ('Sevilla', []), '42': ('Soria', []),
    '43': ('Tarragona', []), '44': ('Teruel', []), '45': ('Toledo', []), '46': ('Valencia', ['valencia/valencia']),
    '47': ('Valladolid', []), '48': ('Bizkaia', []), '49': ('Zamora', []), '50': ('Zaragoza', []),
    '51': ('Ceuta', []), '52': ('Melilla', []),
}


def _norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower().strip()
    for sep in [' (', ',']:
        s = s.split(sep)[0]
    return s.strip()


_ALIAS = {}
for _cod, (_nom, _al) in PROVINCIAS.items():
    for a in [_nom] + _al:
        _ALIAS[_norm(a)] = _cod
        _ALIAS[_norm(a).split('/')[0]] = _cod


def codigo(nombre):
    n = _norm(nombre)
    return _ALIAS.get(n) or _ALIAS[n.split('/')[0]]


def nombre(cod):
    return PROVINCIAS[cod][0]


def leer_andalucia():
    """Tabla 1: hechos conocidos 2017 en las 8 provincias andaluzas (columnas resumen del .xls)."""
    raw = pd.read_excel(os.path.join(DATOS, 'Andalucía.xls'), header=None)
    filas, prov = {}, None
    mapa = {'Crímenes': 'Crímenes', 'Robos': 'Robos', 'Relaciondo con Vehículos': 'Vehículos',
            'Orden Público': 'Orden público', 'Drogas': 'Drogas'}
    for _, r in raw.iterrows():
        if isinstance(r[0], str) and pd.isna(r[1]) and r[0].strip() in [PROVINCIAS[c][0] for c in
                                                                         ['04', '11', '14', '18', '21', '23', '29', '41']]:
            prov = r[0].strip()
            filas[prov] = {}
        if prov and isinstance(r[2], str) and r[2].strip() in mapa:
            filas[prov][mapa[r[2].strip()]] = float(r[3])
    df = pd.DataFrame(filas).T[CRITERIOS]
    df.index.name = 'Provincia'
    return df


def leer_habitantes():
    raw = pd.read_excel(os.path.join(DATOS, 'habitantes.xlsx'), header=None).iloc[2:, 1:3]
    s = pd.Series(raw[2].astype(float).values, index=[codigo(n) for n in raw[1]], name='Habitantes')
    return s


def leer_nacional(anio):
    if anio == 2017:
        df = pd.read_excel(os.path.join(DATOS, 'Datos_nacionales_2017_por_10000_ES.xlsx'))
        df = df.rename(columns={'Provincia': 'Provincia'})
    else:
        df = pd.read_excel(os.path.join(DATOS, 'Datos_nacionales_2023_por_10000_ES.xlsx'), sheet_name='Datos 2023')
    df['cod'] = [codigo(n) for n in df['Provincia']]
    df['Provincia'] = [nombre(c) for c in df['cod']]
    return df.set_index('cod').sort_index()


def por_10000(abs_df, poblacion):
    return abs_df.div(poblacion, axis=0) * 10000


def rps_equidistantes(maximo, n=6):
    """n puntos de referencia simétricos y proporcionales: 0, max/(n-1), ..., max."""
    return np.linspace(0, maximo, n)


def cps_percentiles(datos, q=(0.20, 0.50, 0.80)):
    """Perfiles centrales como percentiles 20/50/80 de las tasas (Alto/Medio/Bajo nivel de seguridad)."""
    return pd.DataFrame({c: [np.quantile(datos[c], qq) for qq in q] for c in CRITERIOS},
                        index=['CP1 Alto', 'CP2 Medio', 'CP3 Bajo'])


def _cps(filas):
    return pd.DataFrame(filas, index=['CP1 Alto', 'CP2 Medio', 'CP3 Bajo'], columns=CRITERIOS)


# ----------------------------------------------------------------------------------------
# 4. Datos de calibración del enunciado (Tablas 3, 6, 8, 10, 11, 12, 13)
# ----------------------------------------------------------------------------------------
# Ejercicio 1 - Tabla 6: puntos (x, prioridad local) de RPs y CPs, valores absolutos
T6 = {
    'Crímenes':      ([0, 20, 41.2, 82.4, 100, 123.6, 160, 164.8, 206],
                      [0.336, 0.277, 0.132, 0.086, 0.058, 0.032, 0.029, 0.026, 0.023]),
    'Robos':         ([0, 5000, 50914.4, 95000, 101828.8, 152743.2, 180000, 203657.6, 254572],
                      [0.32, 0.261, 0.139, 0.075, 0.074, 0.047, 0.032, 0.031, 0.022]),
    'Vehículos':     ([0, 5000, 7025.4, 14050.8, 19000, 21076.2, 27000, 28101.6, 35127],
                      [0.459, 0.138, 0.124, 0.069, 0.063, 0.048, 0.04, 0.034, 0.026]),
    'Orden público': ([0, 300, 764.8, 1529.6, 1800, 2294.4, 2500, 3059.2, 3824],
                      [0.308, 0.259, 0.149, 0.071, 0.066, 0.06, 0.034, 0.032, 0.023]),
    'Drogas':        ([0, 300, 321.6, 600, 643.2, 900, 964.8, 1286.4, 1608],
                      [0.385, 0.262, 0.112, 0.07, 0.062, 0.035, 0.033, 0.023, 0.018]),
}
CP_E1 = _cps([[20, 5000, 5000, 300, 300], [100, 95000, 19000, 1800, 600], [160, 180000, 27000, 2500, 900]])

# Ejercicio 2 - Tablas 8 y 10 (tasas por 10.000 hab., Andalucía)
CP_E2 = _cps([[0.26, 194.29, 22.95, 2.87, 2.06], [0.34, 215.32, 28.44, 3.86, 3.23],
              [0.41, 261.32, 36.20, 5.32, 6.25]])
T10 = {
    'Crímenes':      ([0, 0.10, 0.20, 0.26, 0.30, 0.34, 0.39, 0.41, 0.49],
                      [0.34, 0.13, 0.09, 0.05, 0.03, 0.03, 0.03, 0.03, 0.02]),
    'Robos':         ([0, 56.59, 113.18, 169.77, 194.29, 215.32, 226.35, 261.32, 282.94],
                      [0.32, 0.14, 0.07, 0.05, 0.04, 0.03, 0.03, 0.03, 0.02]),
    'Vehículos':     ([0, 9.96, 19.93, 22.95, 28.44, 29.89, 36.20, 39.85, 49.82],
                      [0.46, 0.12, 0.07, 0.06, 0.05, 0.05, 0.04, 0.03, 0.03]),
    'Orden público': ([0, 1.17, 2.33, 2.87, 3.50, 3.86, 4.66, 5.32, 5.83],
                      [0.31, 0.15, 0.07, 0.07, 0.06, 0.05, 0.03, 0.03, 0.02]),
    'Drogas':        ([0, 1.36, 2.06, 2.73, 3.23, 4.09, 5.45, 6.25, 6.81],
                      [0.39, 0.11, 0.09, 0.06, 0.05, 0.03, 0.02, 0.02, 0.02]),
}
RP_E2_ENUNCIADO = pd.DataFrame({'Crímenes': [0, .10, .20, .30, .39, .49],
                                'Robos': [0, 56.59, 113.18, 169.77, 226.35, 282.94],
                                'Vehículos': [0, 9.96, 19.93, 29.89, 39.85, 49.82],
                                'Orden público': [0, 1.17, 2.33, 3.50, 4.66, 5.83],
                                'Drogas': [0, 1.36, 2.73, 4.09, 5.45, 6.81]},
                               index=[f'RP{i}' for i in range(1, 7)])

# Ejercicios 3 y 4 - Tablas 11, 12 y 13 (tasas nacionales 2017 por 10.000 hab.)
CP_E3 = _cps([[0.17, 129.18, 7.42, 2.73, 1.46], [0.25, 179.38, 13.69, 3.38, 2.00],
              [0.36, 268.99, 30.94, 4.52, 2.77]])
RP_E3 = pd.DataFrame({'Crímenes': [0, .15, .30, .44, .59, .74],
                      'Robos': [0, 81.29, 162.59, 243.88, 325.17, 406.46],
                      'Vehículos': [0, 10.80, 21.59, 32.39, 43.19, 53.98],
                      'Orden público': [0, 1.18, 2.35, 3.53, 4.70, 5.88],
                      'Drogas': [0, 1.36, 2.73, 4.09, 5.45, 6.81]},
                     index=[f'RP{i}' for i in range(1, 7)])
Y_E3 = pd.DataFrame({'Crímenes': [.336, .132, .086, .032, .026, .023],
                     'Robos': [.32, .139, .074, .047, .031, .022],
                     'Vehículos': [.459, .124, .069, .048, .034, .026],
                     'Orden público': [.308, .149, .071, .06, .032, .023],
                     'Drogas': [.385, .112, .062, .033, .023, .018]},
                    index=[f'RP{i}' for i in range(1, 7)])


def _funciones(tabla):
    return {c: FuncionPrioridad(*tabla[c]) for c in CRITERIOS}


# ----------------------------------------------------------------------------------------
# 5. Ejercicios
# ----------------------------------------------------------------------------------------
def ejercicio1():
    ahp = pesos_ahp(A_CRITERIOS)
    datos = leer_andalucia()
    modelo = ModeloAHPSort(ahp['pesos'], _funciones(T6), CP_E1)
    res = modelo.clasificar(datos)
    return {'ahp': ahp, 'datos': datos, 'modelo': modelo, 'resultado': res}


def ejercicio2():
    ahp = pesos_ahp(A_CRITERIOS)
    abs_ = leer_andalucia()
    hab = leer_habitantes()
    pob = pd.Series([hab[codigo(p)] for p in abs_.index], index=abs_.index)
    tasas = por_10000(abs_, pob)
    nac17 = leer_nacional(2017)
    tasas_fichero = nac17.set_index('Provincia').loc[tasas.index, CRITERIOS]
    rp_calc = pd.DataFrame({c: rps_equidistantes(tasas[c].max()) for c in CRITERIOS},
                           index=[f'RP{i}' for i in range(1, 7)])
    rp_calc_fichero = pd.DataFrame({c: rps_equidistantes(tasas_fichero[c].max()) for c in CRITERIOS},
                                   index=rp_calc.index)
    cp_calc = cps_percentiles(tasas)
    cp_calc_fichero = cps_percentiles(tasas_fichero)
    modelo = ModeloAHPSort(ahp['pesos'], _funciones(T10), CP_E2)
    res = modelo.clasificar(tasas)
    return {'ahp': ahp, 'poblacion': pob, 'tasas': tasas, 'tasas_fichero': tasas_fichero, 'modelo': modelo,
            'rp_calc': rp_calc, 'rp_calc_fichero': rp_calc_fichero, 'cp_calc': cp_calc,
            'cp_calc_fichero': cp_calc_fichero, 'resultado': res}


def modelo_nacional():
    ahp = pesos_ahp(A_CRITERIOS)
    funciones = {c: FuncionPrioridad(RP_E3[c].values, Y_E3[c].values) for c in CRITERIOS}
    return ModeloAHPSort(ahp['pesos'], funciones, CP_E3)


def ejercicio3():
    nac = leer_nacional(2017)
    tasas = nac.set_index('Provincia')[CRITERIOS]
    hab = leer_habitantes()
    # (a) verificación de la normalización con los datos absolutos andaluces
    abs_and = leer_andalucia()
    pob_implicita = pd.DataFrame({c: abs_and[c] / tasas.loc[abs_and.index, c] * 10000 for c in CRITERIOS})
    pob_implicita['Habitantes (tabla)'] = [hab[codigo(p)] for p in abs_and.index]
    # (b) RPs y CPs recalculados con las 50 provincias
    rp_calc = pd.DataFrame({c: rps_equidistantes(tasas[c].max()) for c in CRITERIOS}, index=RP_E3.index)
    cp_calc = cps_percentiles(tasas)
    modelo = modelo_nacional()
    res = modelo.clasificar(tasas)
    res.insert(0, 'cod', nac.index)
    return {'tasas': tasas, 'pob_implicita': pob_implicita, 'rp_calc': rp_calc, 'cp_calc': cp_calc,
            'modelo': modelo, 'resultado': res}


def ejercicio4():
    nac = leer_nacional(2023)
    abs_cols = ['Abs Crímenes', 'Abs Robos', 'Abs Vehículos', 'Abs Orden público', 'Abs Drogas']
    recalculo = nac[abs_cols].div(nac['Población 2023'], axis=0).values * 10000
    err_norm = float(np.abs(recalculo - nac[CRITERIOS].values).max())
    tasas = nac.set_index('Provincia')[CRITERIOS]
    modelo = modelo_nacional()
    fuera = pd.DataFrame({c: tasas[c] > RP_E3[c].iloc[-1] for c in CRITERIOS})
    res = modelo.clasificar(tasas)
    res.insert(0, 'cod', nac.index)
    r17 = ejercicio3()['resultado']
    trans = pd.DataFrame({'cod': res['cod'], 'Global 2017': r17['Global'], 'Clase 2017': r17['Nivel de seguridad'],
                          'Global 2023': res['Global'], 'Clase 2023': res['Nivel de seguridad']})
    orden = {c: i for i, c in enumerate(CLASES)}
    trans['Cambio'] = [('=' if a == b else ('mejora' if orden[b] < orden[a] else 'empeora'))
                       for a, b in zip(trans['Clase 2017'], trans['Clase 2023'])]
    matriz = pd.crosstab(pd.Categorical(trans['Clase 2017'], CLASES), pd.Categorical(trans['Clase 2023'], CLASES),
                         rownames=['2017'], colnames=['2023'], dropna=False)
    return {'tasas': tasas, 'err_norm': err_norm, 'fuera_rango': fuera, 'modelo': modelo, 'resultado': res,
            'transicion': trans, 'matriz': matriz}


# ----------------------------------------------------------------------------------------
# 6. Gráficas y mapas
# ----------------------------------------------------------------------------------------
def grafica_funciones(modelo, titulo, fichero=None, unidades=''):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.4))
    for ax, c in zip(axs, CRITERIOS):
        f = modelo.funciones[c]
        ax.plot(f.x, f.y, '-o', color='#1F4E79', ms=4, lw=1.5, label='función (RP)')
        ax.plot(modelo.cp_x[c], modelo.cp_local[c], 'o', color='#C62828', ms=6, label='CP1-CP3')
        for i, (x, y) in enumerate(zip(modelo.cp_x[c], modelo.cp_local[c])):
            ax.annotate(f'CP{i + 1}', (x, y), textcoords='offset points', xytext=(4, 5), fontsize=7, color='#C62828')
        ax.set_title(c, fontsize=10)
        ax.set_xlabel(unidades, fontsize=8)
        ax.grid(alpha=.3)
        ax.tick_params(labelsize=7)
    axs[0].set_ylabel('prioridad local')
    axs[0].legend(fontsize=7)
    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    if fichero:
        fig.savefig(fichero, dpi=160, bbox_inches='tight')
    return fig


def _leyenda(ax):
    from matplotlib.patches import Patch
    etiqueta = {'Alto': 'Alta (CP1)', 'Medio': 'Media (CP2)', 'Bajo': 'Baja (CP3)'}
    ax.legend(handles=[Patch(color=COLORES[c], label=etiqueta[c]) for c in CLASES], loc='upper center',
              bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9, frameon=False,
              title='Nivel de seguridad', title_fontsize=9)


def mapa_andalucia(resultado, titulo, ax=None, fichero=None):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    g = gpd.read_file(os.path.join(MAPAS, 'andalucia_provincias.geojson')).to_crs(25830)
    clase = {codigo(p): c for p, c in resultado['Nivel de seguridad'].items()}
    glob = {codigo(p): v for p, v in resultado['Global'].items()}
    g['clase'] = g['cod'].map(clase)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.6))
    g.plot(ax=ax, color=g['clase'].map(COLORES), edgecolor='white', linewidth=1)
    for _, r in g.iterrows():
        pt = r.geometry.representative_point()
        ax.annotate(f"{r['nombre']}\n{glob[r['cod']]:.4f}".replace('.', ','), (pt.x, pt.y), ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
    ax.set_title(titulo, fontsize=11)
    ax.set_axis_off()
    _leyenda(ax)
    if fichero:
        ax.figure.savefig(fichero, dpi=160, bbox_inches='tight')
    return ax


def mapa_espana(resultado, titulo, ax=None, fichero=None):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    g = gpd.read_file(os.path.join(MAPAS, 'espana_provincias.geojson'))
    clase = dict(zip(resultado['cod'], resultado['Nivel de seguridad']))
    g['clase'] = g['cod'].map(clase)
    can = g['cod'].isin(['35', '38'])
    pen = g[~can].to_crs(25830)
    ica = g[can].to_crs(25830)
    # Canarias desplazada como recuadro al suroeste de la península (práctica habitual en mapas de España)
    ica = ica.set_geometry(ica.translate(xoff=1_050_000, yoff=570_000))
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7.5))
    for parte in (pen, ica):
        parte.plot(ax=ax, color=parte['clase'].map(COLORES), edgecolor='white', linewidth=.6)
    minx, miny, maxx, maxy = ica.total_bounds
    ax.plot([minx - 3e4, maxx + 3e4, maxx + 3e4, minx - 3e4, minx - 3e4],
            [miny - 3e4, miny - 3e4, maxy + 3e4, maxy + 3e4, miny - 3e4], color='#888', lw=.8)
    ax.set_title(titulo, fontsize=11)
    ax.set_axis_off()
    _leyenda(ax)
    if fichero:
        ax.figure.savefig(fichero, dpi=160, bbox_inches='tight')
    return ax


# ----------------------------------------------------------------------------------------
# 7. Ejecución completa por consola
# ----------------------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(RESULTADOS, exist_ok=True)
    pd.set_option('display.width', 200)
    pd.set_option('display.precision', 4)

    e1 = ejercicio1()
    a = e1['ahp']
    print('=== Pesos de los criterios (Tabla 4) ===')
    print(pd.Series(a['pesos'], index=CRITERIOS).round(4).to_string())
    print(f"λmáx={a['lambda_max']:.4f}  IC={a['IC']:.4f}  IA={a['IA']:.3f}  RC={a['RC']:.4f}  "
          f"-> {'válido' if a['RC'] < 0.1 else 'NO válido'}")
    print('\n=== Ejercicio 1: CP (Tabla 7) ===')
    print(e1['modelo'].tabla_cp().round(4).to_string())
    print('\n=== Ejercicio 1: clasificación (datos absolutos) ===')
    print(e1['resultado'].round(4).to_string())
    e1['resultado'].to_csv(os.path.join(RESULTADOS, 'ej1_clasificacion.csv'))
    grafica_funciones(e1['modelo'], 'Ej. 1 - Funciones de prioridad local (hechos absolutos)',
                      os.path.join(RESULTADOS, 'ej1_funciones.png'), 'nº de hechos')
    mapa_andalucia(e1['resultado'], 'Ej. 1 - Andalucía 2017 (datos absolutos)',
                   fichero=os.path.join(RESULTADOS, 'ej1_mapa.png'))

    e2 = ejercicio2()
    print('\n=== Ejercicio 2: tasas por 10.000 hab. ===')
    print(e2['tasas'].round(3).to_string())
    print('\nRP recalculados (max/5):'); print(e2['rp_calc'].round(2).to_string())
    print('\nCP recalculados (P20/P50/P80):'); print(e2['cp_calc'].round(2).to_string())
    print('\nCP (Tabla 8) y prioridades:'); print(e2['modelo'].tabla_cp().round(4).to_string())
    print('\n=== Ejercicio 2: clasificación ===')
    print(e2['resultado'].round(4).to_string())
    e2['resultado'].to_csv(os.path.join(RESULTADOS, 'ej2_clasificacion.csv'))
    grafica_funciones(e2['modelo'], 'Ej. 2 - Funciones de prioridad local (Andalucía, por 10.000 hab.)',
                      os.path.join(RESULTADOS, 'ej2_funciones.png'), 'hechos / 10.000 hab.')
    mapa_andalucia(e2['resultado'], 'Ej. 2 - Andalucía 2017 (por 10.000 habitantes)',
                   fichero=os.path.join(RESULTADOS, 'ej2_mapa.png'))

    e3 = ejercicio3()
    print('\n=== Ejercicio 3: verificación normalización (población implícita) ===')
    print(e3['pob_implicita'].round(0).to_string())
    print('\nRP recalculados:'); print(e3['rp_calc'].round(2).to_string())
    print('\nCP recalculados (P20/P50/P80):'); print(e3['cp_calc'].round(2).to_string())
    print('\nCP (Tabla 14):'); print(e3['modelo'].tabla_cp().round(4).to_string())
    print('\n=== Ejercicio 3: clasificación nacional 2017 ===')
    print(e3['resultado'].sort_values('Global', ascending=False).round(4).to_string())
    e3['resultado'].to_csv(os.path.join(RESULTADOS, 'ej3_clasificacion.csv'))
    grafica_funciones(e3['modelo'], 'Ej. 3 - Funciones de prioridad local nacionales 2017',
                      os.path.join(RESULTADOS, 'ej3_funciones.png'), 'hechos / 10.000 hab.')
    mapa_espana(e3['resultado'], 'Ej. 3 - España 2017', fichero=os.path.join(RESULTADOS, 'ej3_mapa.png'))

    e4 = ejercicio4()
    print(f"\n=== Ejercicio 4: error máximo de la normalización 2023 = {e4['err_norm']:.2e}")
    print('Tasas fuera de RP6 (truncadas):')
    f = e4['fuera_rango']
    for c in CRITERIOS:
        if f[c].any():
            print(f'  {c}: ' + ', '.join(f.index[f[c]]))
    print('\n=== Ejercicio 4: clasificación 2023 ===')
    print(e4['resultado'].sort_values('Global', ascending=False).round(4).to_string())
    print('\nTabla de transición:'); print(e4['transicion'].round(4).to_string())
    print('\nMatriz de cambios:'); print(e4['matriz'].to_string())
    e4['resultado'].to_csv(os.path.join(RESULTADOS, 'ej4_clasificacion.csv'))
    e4['transicion'].to_csv(os.path.join(RESULTADOS, 'ej4_transicion.csv'))
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    mapa_espana(e3['resultado'], 'España 2017', ax=axs[0])
    mapa_espana(e4['resultado'], 'España 2023 (calibración 2017)', ax=axs[1])
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTADOS, 'ej4_mapas_2017_2023.png'), dpi=160, bbox_inches='tight')
    mapa_espana(e4['resultado'], 'Ej. 4 - España 2023', fichero=os.path.join(RESULTADOS, 'ej4_mapa.png'))
    plt.close('all')


if __name__ == '__main__':
    main()
