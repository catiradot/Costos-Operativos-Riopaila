# ============================================================
# RIOPAILA CASTILLA — Sistema de Analitica de Costos
# Streamlit App | INTEP Roldanillo Valle
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                              accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              classification_report, silhouette_score)

# ── Configuracion de pagina ───────────────────────────────────
st.set_page_config(
    page_title="Riopaila Castilla — Analitica de Costos",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personalizado ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 50%, #40916c 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
        text-align: center; color: white;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; font-weight: 700; }
    .main-header p  { font-size: 1rem; margin: 0.5rem 0 0 0; opacity: 0.9; }
    .metric-card {
        background: #1b4332; padding: 1.2rem; border-radius: 10px;
        border-left: 5px solid #52b788; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        text-align: center;
    }
    .metric-card h3 { color: #74c69d; font-size: 1.6rem; margin: 0; }
    .metric-card p  { color: #b7e4c7; font-size: 0.85rem; margin: 0.3rem 0 0 0; }
    .section-header {
        background: #f0f7f4; padding: 0.8rem 1.2rem; border-radius: 8px;
        border-left: 4px solid #2d6a4f; margin: 1.5rem 0 1rem 0;
    }
    .section-header h3 { color: #1a472a; margin: 0; font-size: 1.1rem; }
    .alert-box {
        background: #3d2b00; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #f4a261; margin: 1rem 0; color: #fde8c8;
    }
    .alert-box b { color: #f4a261; }
    .success-box {
        background: #1b4332; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #52b788; margin: 1rem 0; color: #d8f3dc;
    }
    .success-box b { color: #74c69d; }
    .info-box {
        background: #023047; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #48cae4; margin: 1rem 0; color: #caf0f8;
    }
    .info-box b { color: #90e0ef; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
    div[data-testid="stSidebarContent"] { background: #0d2818 !important; }
    div[data-testid="stSidebarContent"] label { color: #b7e4c7 !important; }
    div[data-testid="stSidebarContent"] .stMarkdown { color: #b7e4c7 !important; }
    div[data-testid="stSidebarContent"] h3 { color: #52b788 !important; }
    div[data-testid="stSidebarContent"] .stMetric { background: #1b4332; border-radius: 8px; padding: 0.5rem; }
    div[data-testid="stSidebarContent"] .stMetricLabel { color: #95d5b2 !important; }
    div[data-testid="stSidebarContent"] .stMetricValue { color: #ffffff !important; }
    div[data-testid="stSidebarContent"] hr { border-color: #2d6a4f; }
    .stMultiSelect span { background: #2d6a4f !important; color: white !important; }
    .stMultiSelect [data-baseweb="tag"] { background: #2d6a4f !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════

@st.cache_data
def cargar_y_limpiar(archivo):
    """Carga y limpia el dataset aplicando los 4 pasos del notebook."""
    df = pd.read_excel(archivo, sheet_name='Hoja1')
    df.columns = df.columns.str.strip()

    # Problema 1: #N/D
    df['GRUPO LABORES'] = df['GRUPO LABORES'].replace('#N/D', 'Sin Clasificar')

    # Problema 2: Numericos y negativos
    for col in ['Csts.real.cargo', 'Cant.producida real', 'Csts.unitarios real', 'Tarifa']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[df['Csts.real.cargo'] >= 0].copy()

    # Problema 3: Fecha serial Excel
    if 'Fecha' in df.columns:
        df['Fecha_Real'] = pd.to_datetime(df['Fecha'], origin='1899-12-30', unit='D', errors='coerce')
        if df['Mes'].isnull().any() or 'Mes' not in df.columns:
            df['Mes'] = df['Fecha_Real'].dt.month
        if df['Año'].isnull().any() or 'Año' not in df.columns:
            df['Año'] = df['Fecha_Real'].dt.year

    # Problema 4: columnas innecesarias
    cols_drop = ['Source.Name', 'Elemento PEP', 'Orden']
    df.drop(columns=[c for c in cols_drop if c in df.columns], inplace=True)

    # Labels tenencia
    ten_labels = {10: 'Propia', 20: 'Alquilada', 30: 'Participacion'}
    df['Tenencia_Label'] = df['Tenencia'].map(ten_labels).fillna('Otra')

    return df


@st.cache_data
def entrenar_modelos(df):
    """Entrena todos los modelos del proyecto."""
    resultados = {}

    # ── Datos para regresion ──────────────────────────────────
    min_year = df['Año'].min()
    df = df.copy()
    df['Mes_Continuo'] = (df['Año'] - min_year) * 12 + df['Mes']

    df_grouped = df.groupby(['Año', 'Mes', 'GRUPO LABORES', 'Mes_Continuo']).agg(
        Costo_Total=('Csts.real.cargo', 'sum'),
        Produccion_Total=('Cant.producida real', 'sum'),
        N_Labores=('Csts.real.cargo', 'count')
    ).reset_index()

    df_model = pd.get_dummies(df_grouped.copy(), columns=['GRUPO LABORES'], drop_first=True)
    feat_reg = ['Mes_Continuo'] + [c for c in df_model.columns if c.startswith('GRUPO LABORES_')]
    X_reg = df_model[feat_reg]
    y_reg = df_model['Costo_Total']

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)

    # Regresion simple
    X_s = df_grouped[['Mes_Continuo']]
    y_s = df_grouped['Costo_Total']
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
    mod_simple = LinearRegression()
    mod_simple.fit(X_tr_s, y_tr_s)
    y_pred_s = mod_simple.predict(X_te_s)

    # Regresion multiple
    mod_multiple = LinearRegression()
    mod_multiple.fit(X_train_m, y_train_m)
    y_pred_m = mod_multiple.predict(X_test_m)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_m, y_train_m)
    y_pred_rf = rf.predict(X_test_m)

    resultados['reg'] = {
        'mod_simple': mod_simple, 'mod_multiple': mod_multiple, 'rf': rf,
        'X_train_m': X_train_m, 'X_test_m': X_test_m,
        'y_train_m': y_train_m, 'y_test_m': y_test_m,
        'y_pred_s': y_pred_s, 'y_pred_m': y_pred_m, 'y_pred_rf': y_pred_rf,
        'X_te_s': X_te_s, 'y_te_s': y_te_s,
        'feat_reg': feat_reg, 'df_grouped': df_grouped, 'min_year': min_year
    }

    # ── Clasificacion ─────────────────────────────────────────
    umbral = df['Csts.real.cargo'].quantile(0.75)
    df_c = df.copy()
    df_c['Labor_Costosa'] = (df_c['Csts.real.cargo'] > umbral).astype(int)
    cols_base = [c for c in ['Mes', 'Tenencia', 'Cant.producida real'] if c in df_c.columns]
    cols_dum = [c for c in ['GRUPO LABORES', 'Tipo Labor'] if c in df_c.columns]
    df_c = pd.get_dummies(df_c, columns=cols_dum, drop_first=True)
    dum_cols = [c for c in df_c.columns if 'GRUPO LABORES_' in c or 'Tipo Labor_' in c]
    feat_clf = cols_base + dum_cols
    X_clf = df_c[feat_clf].dropna()
    y_clf = df_c.loc[X_clf.index, 'Labor_Costosa']
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    scaler_clf = StandardScaler()
    X_train_sc = scaler_clf.fit_transform(X_train)
    X_test_sc  = scaler_clf.transform(X_test)

    rl = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    rl.fit(X_train_sc, y_train)
    y_pred_rl  = rl.predict(X_test_sc)
    y_proba_rl = rl.predict_proba(X_test_sc)[:, 1]

    arbol = DecisionTreeClassifier(max_depth=4, random_state=42,
                                    min_samples_leaf=50, class_weight='balanced')
    arbol.fit(X_train, y_train)
    y_pred_arbol  = arbol.predict(X_test)
    y_proba_arbol = arbol.predict_proba(X_test)[:, 1]

    resultados['clf'] = {
        'rl': rl, 'arbol': arbol, 'scaler_clf': scaler_clf,
        'feat_clf': feat_clf, 'umbral': umbral,
        'X_train': X_train, 'X_test': X_test,
        'X_train_sc': X_train_sc, 'X_test_sc': X_test_sc,
        'y_train': y_train, 'y_test': y_test,
        'y_pred_rl': y_pred_rl, 'y_proba_rl': y_proba_rl,
        'y_pred_arbol': y_pred_arbol, 'y_proba_arbol': y_proba_arbol
    }

    # ── Clustering ────────────────────────────────────────────
    sector_col = ('Sector-suerte' if 'Sector-suerte' in df.columns
                  else ('Sector' if 'Sector' in df.columns else None))
    if sector_col:
        lotes = df.groupby(sector_col).agg(
            Costo_Total=('Csts.real.cargo', 'sum'),
            Produccion_Total=('Cant.producida real', 'sum'),
            N_Labores=('Csts.real.cargo', 'count'),
            Costo_Promedio=('Csts.real.cargo', 'mean')
        ).reset_index()
        lotes['Costo_x_Unidad'] = (lotes['Costo_Total'] /
                                    lotes['Produccion_Total'].replace(0, np.nan)).round(0)
        lotes = lotes.dropna()
        lotes_activos = lotes[lotes['N_Labores'] >= 20].copy()
        vars_cl = ['Costo_Total', 'Produccion_Total', 'N_Labores']
        X_cl = lotes_activos[vars_cl]
        scaler_cl = StandardScaler()
        X_scaled = scaler_cl.fit_transform(X_cl)

        # Silhouette scores
        siluetas = {}
        for k in range(2, 9):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            siluetas[k] = silhouette_score(X_scaled, labels)

        km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
        lotes_activos['Cluster'] = km4.fit_predict(X_scaled)
        nombres_cl = {
            0: 'Lotes Estandar',
            1: 'Alta Produccion',
            2: 'Lotes Ineficientes',
            3: 'Lotes de Elite'
        }
        lotes_activos['Nombre_Cluster'] = lotes_activos['Cluster'].map(nombres_cl)

        resultados['clust'] = {
            'lotes_activos': lotes_activos,
            'vars_cl': vars_cl,
            'X_scaled': X_scaled,
            'siluetas': siluetas,
            'sector_col': sector_col
        }

    return resultados


# ══════════════════════════════════════════════════════════════
# ENCABEZADO PRINCIPAL
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🌿 Ingenio Riopaila Castilla</h1>
    <p>Sistema de Analitica de Costos Operativos — Labores Agricolas 2021-2026</p>
    <p style="font-size:0.85rem; opacity:0.75;">Analitica Predictiva Aplicada a los Negocios | INTEP Roldanillo Valle</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR — CARGA DE DATOS
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("assets/logo_riopaila.png", use_column_width=True)
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a472a,#2d6a4f);
                padding:0.5rem;border-radius:0 0 10px 10px;text-align:center;margin-bottom:0.5rem">
        <div style="color:#b7e4c7;font-size:0.8rem;font-weight:600">
            Analitica de Costos 2021-2026
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📂 Cargar Dataset")
    archivo = st.file_uploader(
        "Sube el archivo Excel del proyecto",
        type=["xlsx"],
        help="Comportamiento historico labores 2021-2026"
    )
    st.markdown("---")
    st.markdown("### 🔧 Filtros Globales")

    if archivo:
        df_raw = cargar_y_limpiar(archivo)
        anos_disp = sorted(df_raw['Año'].unique().tolist())
        grupos_disp = sorted(df_raw['GRUPO LABORES'].dropna().unique().tolist())

        anos_sel = st.multiselect(
            "Años a analizar",
            options=anos_disp,
            default=anos_disp,
            help="Filtra el periodo de analisis"
        )
        grupos_sel = st.multiselect(
            "Grupos de labor",
            options=grupos_disp,
            default=[g for g in grupos_disp if g not in ['Sin Clasificar', 'DESCONOCIDO']],
            help="Tipos de labor a incluir"
        )


        st.markdown("---")
        st.markdown("### ℹ️ Dataset cargado")
        st.metric("Registros", f"{len(df_raw):,}")
        st.metric("Periodo", f"{df_raw['Año'].min()} – {df_raw['Año'].max()}")
        costo_total_raw = df_raw['Csts.real.cargo'].sum()
        if costo_total_raw >= 1e12:
            st.metric("Costo total", f"${costo_total_raw/1e12:.2f} Billones COP")
        else:
            st.metric("Costo total", f"${costo_total_raw/1e9:.0f} Mil Millones COP")

        st.markdown("---")
        st.markdown("### 💰 Top 3 Materiales")
        col_mat_sb = next((c for c in ['Texto breve de material', 'Numero de material']
                           if c in df_raw.columns), None)
        if col_mat_sb:
            top3_sb = (df_raw.groupby(col_mat_sb)['Csts.real.cargo']
                       .sum().sort_values(ascending=False).head(3).reset_index())
            top3_sb.columns = ['Material', 'Costo']
            top3_sb['%'] = (top3_sb['Costo'] / df_raw['Csts.real.cargo'].sum() * 100).round(1)
            iconos = ['🥇', '🥈', '🥉']
            for i, row in top3_sb.iterrows():
                nombre = row['Material'][:24] + '...' if len(row['Material']) > 24 else row['Material']
                st.markdown(f"""
<div style="background:#1b4332;border-left:4px solid #52b788;
            border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:0.5rem">
    <div style="color:#74c69d;font-size:0.75rem;font-weight:600">
        {iconos[i]} {nombre}
    </div>
    <div style="color:#ffffff;font-size:1rem;font-weight:700;margin-top:0.2rem">
        ${row['Costo']/1e9:.1f}B
    </div>
    <div style="color:#b7e4c7;font-size:0.72rem">{row['%']:.1f}% del presupuesto total</div>
</div>""", unsafe_allow_html=True)
    else:
        st.info("👆 Sube el archivo Excel para comenzar")
        st.stop()

# ── Aplicar filtros ───────────────────────────────────────────
df = df_raw.copy()
# Decision del grupo: solo tenencias propias 10/20/30
df = df[df['Tenencia'].isin([10, 20, 30])].copy()
df = df[df['Año'].isin(anos_sel)]
df = df[df['GRUPO LABORES'].isin(grupos_sel)]

if len(df) == 0:
    st.error("No hay datos con los filtros seleccionados. Ajusta los filtros del sidebar.")
    st.stop()

# ══════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Resumen Ejecutivo",
    "🔍 EDA & Hallazgos",
    "📈 Regresion & RF",
    "📅 SARIMA 2026",
    "🎯 Clasificacion",
    "🗺️ Clustering",
    "🧮 Simulador"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════

with tab1:
    st.markdown("## 📊 Resumen Ejecutivo")

    with st.expander("👥 Equipo de Trabajo", expanded=False):
        st.markdown("""
| Integrante | Rol |
|---|---|
| **Cesar Augusto Tirado** | Analista de datos / |
| **Eliana Villanueva** | Analista de datos /  |
| **Francisco Jaier Trejos** | Analista de datos / |

**Fecha:** Abril 2026 | **Repositorio:** https://github.com/INTEP-Analitica-2026/Costos-Operativos-Riopaila
        """)

    with st.expander("🏭 Problema de Negocio", expanded=False):
        st.markdown("""
### Pregunta central
> **¿Que factores determinan el costo de una labor agricola en Riopaila Castilla, y como podemos predecir si una labor sera costosa antes de ejecutarla?**

**Contexto:** Riopaila Castilla es uno de los ingenios azucareros mas grandes de Colombia. Su operacion agricola involucra miles de labores anuales registradas en SAP: fertilizacion, riego, preparacion de tierras, siembra y cosecha de cana.

**Por que importa:** Un error de presupuesto en esta escala representa miles de millones de pesos. Los costos crecieron 70% entre 2021-2023 sin un sistema de alerta temprana.

**Quien se beneficia:**
- 🌾 **Gerente de campo:** decide que lotes priorizar
- 💰 **Area financiera:** planifica presupuesto con mayor precision
- 📊 **Direccion general:** identifica oportunidades de ahorro
        """)

    st.markdown("**Objetivo:** Desarrollar un sistema de analitica de datos que permita predecir costos, identificar anomalias y segmentar patrones operativos a partir del comportamiento historico de labores 2021-2026.")

    # KPIs principales
    col1, col2, col3, col4, col5 = st.columns(5)
    costo_total = df['Csts.real.cargo'].sum()
    costo_fert = df[df['GRUPO LABORES'].str.contains('ertiliz', na=False)]['Csts.real.cargo'].sum()

    with col1:
        st.markdown(f"""<div class="metric-card">
            <h3>{len(df):,}</h3><p>Registros SAP</p></div>""", unsafe_allow_html=True)
    with col2:
        costo_label = (f"${costo_total/1e12:.2f}B" if costo_total >= 1e12
                       else f"${costo_total/1e9:.0f}MM")
        st.markdown(f"""<div class="metric-card">
            <h3>{costo_label}</h3><p>Costo Total COP</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <h3>${df['Csts.real.cargo'].mean():,.0f}</h3><p>Costo Promedio/Labor</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <h3>{costo_fert/costo_total*100:.0f}%</h3><p>% Fertilizacion</p></div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="metric-card">
            <h3>{df['GRUPO LABORES'].nunique()}</h3><p>Grupos de Labor</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 💰 Distribucion del Gasto por Grupo")
        costos_g = df.groupby('GRUPO LABORES')['Csts.real.cargo'].sum().sort_values(ascending=False).head(8)
        fig_pie = px.pie(
            values=costos_g.values,
            names=costos_g.index,
            color_discrete_sequence=px.colors.sequential.Greens_r,
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=False, height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("### 📅 Evolucion Anual de Costos")
        anual = df.groupby('Año')['Csts.real.cargo'].sum().reset_index()
        anual['variacion'] = anual['Csts.real.cargo'].pct_change() * 100
        fig_anual = go.Figure()
        fig_anual.add_trace(go.Bar(
            x=anual['Año'], y=anual['Csts.real.cargo']/1e9,
            marker_color=['#e74c3c' if v > 0 else '#2ecc71'
                          for v in anual['variacion'].fillna(0)],
            text=[f'${v:.1f}B' for v in anual['Csts.real.cargo']/1e9],
            textposition='outside', name='Costo Total'
        ))
        fig_anual.update_layout(
            xaxis_title='Año', yaxis_title='Costo (Miles de Millones $)',
            height=380, margin=dict(t=20, b=20), showlegend=False
        )
        st.plotly_chart(fig_anual, use_container_width=True)

    # Hallazgos clave
    st.markdown("### 🔑 Hallazgos Clave del Dataset")

    # Funcion auxiliar de formato — disponible para todo Tab1 y Tab2
    def fmt_b(val):
        if val >= 1e12: return f"${val/1e12:.1f} Bill"
        elif val >= 1e9: return f"${val/1e9:.1f}B"
        else: return f"${val/1e6:.0f}M"
    costo_2021 = df[df['Año'] == df['Año'].min()]['Csts.real.cargo'].sum()
    costo_max  = df.groupby('Año')['Csts.real.cargo'].sum().max()
    year_max   = df.groupby('Año')['Csts.real.cargo'].sum().idxmax()

    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(f"""<div class="alert-box">
            <b>🌱 Fertilizacion domina</b><br>
            {fmt_b(costo_fert)} en {df['Año'].nunique()} anos = {costo_fert/costo_total*100:.0f}% del presupuesto.
            ABONO APORQUE es el mayor gasto individual.
        </div>""", unsafe_allow_html=True)
    with h2:
        var = (costo_max - costo_2021)/costo_2021*100 if costo_2021 > 0 else 0
        st.markdown(f"""<div class="alert-box">
            <b>📈 Costos crecieron {var:.0f}%</b><br>
            Desde {df['Año'].min()} ({fmt_b(costo_2021)}) hasta {year_max} ({fmt_b(costo_max)}).
            Coincide con inflacion post-pandemia en insumos.
        </div>""", unsafe_allow_html=True)
    with h3:
        n_lotes_inef = 226
        st.markdown(f"""<div class="alert-box">
            <b>⚠️ 226 lotes ineficientes detectados</b><br>
            Costo/unidad de $145.140 vs $28.043 de los Lotes de Elite.
            Son <b>5x mas caros</b> por unidad producida — mayor oportunidad de ahorro.
        </div>""", unsafe_allow_html=True)

    st.markdown("### 📋 Variables del Modelo")
    vars_df = pd.DataFrame({
        'Variable': ['Csts.real.cargo', 'Cant.producida real', 'GRUPO LABORES',
                     'Mes', 'Año', 'Tenencia', 'Sector/Suerte'],
        'Rol': ['Y — Dependiente', 'X1 — Independiente', 'X2 — Independiente',
                'X3 — Independiente', 'X4 — Independiente',
                'X5 — Independiente', 'X6 — Independiente'],
        'Descripcion': [
            'Costo real total cargado a cada labor (objetivo a predecir)',
            'Cantidad producida — variable mas importante (importancia 0.818)',
            'Tipo de labor agricola — define estructura de costos',
            'Mes de ejecucion — captura estacionalidad anual',
            'Año — captura inflacion e incremento de insumos',
            'Tipo de tenencia de tierra (propia vs arrendada)',
            'Ubicacion geografica del lote — distancia y condiciones'
        ]
    })
    st.dataframe(vars_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════

with tab2:
    st.markdown("## 🔍 Analisis Exploratorio de Datos")

    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
        "Por Grupo", "Evolucion Anual", "Estacionalidad", "Top Materiales", "Tenencia & Correlacion"
    ])

    with subtab1:
        st.markdown("### 2.1 Costos por Grupo de Labor")
        costos_grupo = df.groupby('GRUPO LABORES').agg(
            N_Labores=('Csts.real.cargo', 'count'),
            Costo_Total=('Csts.real.cargo', 'sum'),
            Costo_Promedio=('Csts.real.cargo', 'mean'),
            Produccion_Total=('Cant.producida real', 'sum')
        ).sort_values('Costo_Total', ascending=False).reset_index()
        costos_grupo['% del Total'] = (costos_grupo['Costo_Total'] / costos_grupo['Costo_Total'].sum() * 100).round(1)
        costos_grupo['Costo_x_Unidad'] = (costos_grupo['Costo_Total'] / costos_grupo['Produccion_Total'].replace(0, np.nan)).round(0)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                costos_grupo, x='Costo_Total', y='GRUPO LABORES',
                orientation='h', color='Costo_Total',
                color_continuous_scale='Greens',
                labels={'Costo_Total': 'Costo Total ($)', 'GRUPO LABORES': ''},
                title='Costo Total por Grupo de Labor'
            )
            fig.update_layout(height=450, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(
                costos_grupo, x='Costo_Promedio', y='GRUPO LABORES',
                orientation='h', color='Costo_Promedio',
                color_continuous_scale='Blues',
                labels={'Costo_Promedio': 'Costo Promedio ($)', 'GRUPO LABORES': ''},
                title='Costo Promedio por Labor'
            )
            fig2.update_layout(height=450, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(
            costos_grupo[['GRUPO LABORES', 'N_Labores', 'Costo_Total', 'Costo_Promedio', '% del Total', 'Costo_x_Unidad']]
            .style.format({'Costo_Total': '${:,.0f}', 'Costo_Promedio': '${:,.0f}',
                           '% del Total': '{:.1f}%', 'Costo_x_Unidad': '${:,.0f}'}),
            use_container_width=True, hide_index=True
        )
        st.markdown("""
> **Interpretacion de negocio — 2.1:** La Fertilizacion domina el presupuesto con el 24% del costo total.
> Solo **ABONO APORQUE + EQUIPO ABONO APORQUE** suman aproximadamente **$70 mil millones en 5 anos**.
> Si Riopaila quiere reducir costos operativos, el primer lugar donde mirar es en los contratos
> con proveedores de abono. Riego y Control de Malezas suman otro 30% combinadas.
        """)

    with subtab2:
        st.markdown("### 2.2 Evolucion Anual de Costos")
        anual = df.groupby('Año').agg(
            Costo_Total=('Csts.real.cargo', 'sum'),
            Produccion_Total=('Cant.producida real', 'sum'),
            N_Labores=('Csts.real.cargo', 'count')
        ).reset_index()
        anual['Costo_Unitario'] = (anual['Costo_Total'] / anual['Produccion_Total'].replace(0, np.nan)).round(0)
        anual['Variacion_Pct']  = anual['Costo_Total'].pct_change().round(3) * 100

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=anual['Año'], y=anual['Costo_Total']/1e9,
                marker_color='#2d6a4f',
                text=[f'${v:.1f}B' for v in anual['Costo_Total']/1e9],
                textposition='outside', name='Costo Total'
            ))
            fig.update_layout(title='Costo Total Anual (Miles de Millones)', height=380)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=anual['Año'], y=anual['Costo_Unitario'],
                mode='lines+markers+text',
                text=[f'${v:,.0f}' for v in anual['Costo_Unitario']],
                textposition='top center',
                line=dict(color='#e74c3c', width=2.5),
                marker=dict(size=10)
            ))
            fig2.update_layout(title='Costo Unitario por Año', height=380)
            st.plotly_chart(fig2, use_container_width=True)

        anual_min_yr  = int(anual['Año'].min())
        anual_max_yr  = int(anual['Año'].max())
        costo_min_yr  = float(anual[anual['Año']==anual_min_yr]['Costo_Total'].values[0])
        costo_max_yr  = float(anual['Costo_Total'].max())
        yr_max_costo  = int(anual.loc[anual['Costo_Total'].idxmax(), 'Año'])
        costo_last_yr = float(anual[anual['Año']==anual_max_yr]['Costo_Total'].values[0])
        var_pct       = (costo_max_yr - costo_min_yr) / costo_min_yr * 100 if costo_min_yr > 0 else 0

        st.markdown(f"""
> **Interpretacion de negocio — 2.2:** Los costos crecieron **{var_pct:.0f}% entre {anual_min_yr} ({fmt_b(costo_min_yr)}) y {yr_max_costo} ({fmt_b(costo_max_yr)})**,
> coincidiendo con la inflacion post-pandemia en insumos agricolas. En {anual_max_yr} {'bajaron' if costo_last_yr < costo_max_yr else 'se mantuvieron'} a {fmt_b(costo_last_yr)} —
> señal de {'recuperacion de eficiencia' if costo_last_yr < costo_max_yr else 'estabilizacion'}.
> El costo unitario confirma que no es solo un efecto de volumen sino de precio real de los insumos.
        """)

        st.markdown("### 2.3 Costos Promedio por Año y Grupo")
        year_sel = st.selectbox("Selecciona el año", sorted(df['Año'].unique()), index=0)
        df_year = df[df['Año'] == year_sel]
        resumen_year = df_year.groupby('GRUPO LABORES')['Csts.real.cargo'].mean().sort_values(ascending=False).reset_index()
        fig3 = px.bar(
            resumen_year, x='GRUPO LABORES', y='Csts.real.cargo',
            color='Csts.real.cargo', color_continuous_scale='Viridis',
            title=f'Costo Promedio por Grupo — Año {year_sel}',
            labels={'Csts.real.cargo': 'Costo Promedio ($)'}
        )
        fig3.update_layout(height=420, coloraxis_showscale=False, xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(f"""
> **Interpretacion de negocio — 2.3:** En {year_sel}, **Siembra** lidera el costo promedio por labor,
> seguida de Adecuacion y Fertilizacion. Este patron se repite en todos los anos analizados,
> lo que confirma que la estructura de costos es **estable y predecible**.
> Las labores de preparacion y establecimiento del cultivo (Siembra, Adecuacion) son las mas costosas
> por labor individual, mientras que Fertilizacion domina en **costo total acumulado** por su alta frecuencia.
> El grupo de labor es el predictor mas importante del modelo.
        """)

    with subtab3:
        st.markdown("### 2.4 Estacionalidad Mensual")
        meses_n = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                   'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        mensual = df.groupby('Mes')['Csts.real.cargo'].sum().reset_index()
        mensual = mensual.sort_values('Mes').reset_index(drop=True)
        mensual['Costo_Prom'] = mensual['Csts.real.cargo'] / df['Año'].nunique()
        mensual['Mes_Nombre'] = mensual['Mes'].apply(lambda x: meses_n[x-1] if 1 <= x <= 12 else str(x))
        mediana_m = mensual['Costo_Prom'].median()
        mensual['Color'] = mensual['Costo_Prom'].apply(
            lambda x: 'Sobre mediana' if x > mediana_m else 'Bajo mediana')

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=mensual['Mes_Nombre'],
            y=mensual['Costo_Prom'],
            marker_color=['#e74c3c' if x > mediana_m else '#3498db'
                          for x in mensual['Costo_Prom']],
            text=[f'${v/1e9:.1f}B' for v in mensual['Costo_Prom']],
            textposition='outside',
            name='Costo Mensual'
        ))
        fig.add_hline(y=mediana_m, line_dash='dash', line_color='gray',
                      annotation_text='Mediana')
        fig.update_layout(
            title='Costo Mensual Promedio — Patron Estacional',
            xaxis_title='Mes',
            yaxis_title='Costo Mensual Promedio ($)',
            height=450,
            showlegend=False,
            xaxis=dict(categoryorder='array',
                       categoryarray=meses_n)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
> **Interpretacion de negocio — 2.4:** El patron estacional muestra dos picos
> de costo claramente definidos: **Febrero** es el mes mas costoso del año,
> seguido de **Julio y Diciembre**. Los meses mas economicos son Mayo, Junio
> y Noviembre. Este patron no es el tipico julio-septiembre — en Riopaila
> el primer trimestre (especialmente Febrero) tiene alta actividad de
> fertilizacion y preparacion. Esto es clave para la planificacion
> presupuestaria: el area financiera debe reservar mayor presupuesto para
> Febrero y el segundo semestre. El patron anual justifica el uso de
> **SARIMA con m=12** para el pronostico.
        """)

        st.markdown("### 2.5 Frecuencia de Labores Registradas")

        df_freq = df[~df['GRUPO LABORES'].isin(['Sin Clasificar', 'DESCONOCIDO'])].copy()
        freq_data = (df_freq['GRUPO LABORES'].value_counts().reset_index())
        freq_data.columns = ['GRUPO LABORES', 'Frecuencia']

        fig_freq = px.bar(
            freq_data,
            x='Frecuencia', y='GRUPO LABORES',
            orientation='h',
            color='Frecuencia',
            color_continuous_scale='Viridis',
            title=f'Frecuencia de Labores Registradas ({df["Año"].min()}-{df["Año"].max()})',
            labels={'Frecuencia': 'Numero de Registros', 'GRUPO LABORES': 'Grupo de Labor'},
            text=[f'{v:,}' for v in freq_data['Frecuencia']]
        )
        fig_freq.update_traces(textposition='outside')
        fig_freq.update_layout(
            height=500,
            coloraxis_showscale=False,
            yaxis=dict(categoryorder='total ascending')
        )
        st.plotly_chart(fig_freq, use_container_width=True)
        st.markdown("""
> **Interpretacion de negocio — 2.5:** **Control de malezas** es la labor mas frecuente del ingenio,
> seguida de Complementarias y Drenaje. Fertilizacion, aunque no es la mas frecuente,
> es la que mas costo total acumula por el alto precio unitario de sus insumos.
> Esta distincion es clave: frecuencia alta no significa necesariamente alto costo total,
> pero si indica alta demanda operativa de recursos humanos y maquinaria.
        """)

    with subtab4:
        st.markdown("### Top 10 Materiales mas Costosos")
        col_mat = next((c for c in ['Texto breve de material', 'Numero de material']
                        if c in df.columns), None)
        if col_mat:
            top_mat = (df.groupby(col_mat)['Csts.real.cargo']
                       .sum().sort_values(ascending=False).head(10).reset_index())
            top_mat.columns = ['Material', 'Costo_Total']
            top_mat['% Total'] = (top_mat['Costo_Total'] / df['Csts.real.cargo'].sum() * 100).round(1)

            fig = px.bar(
                top_mat, x='Costo_Total', y='Material',
                orientation='h', color='Costo_Total',
                color_continuous_scale='Reds',
                title='Top 10 Materiales por Costo Total',
                text=[f'${v/1e9:.1f}B ({p:.1f}%)' for v, p in zip(top_mat['Costo_Total'], top_mat['% Total'])],
                labels={'Costo_Total': 'Costo Total ($)', 'Material': ''}
            )
            fig.update_layout(height=480, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""<div class="alert-box">
                <b>💡 Hallazgo:</b> {top_mat.iloc[0]['Material']} + {top_mat.iloc[1]['Material']}
                suman ${(top_mat.iloc[0]['Costo_Total']+top_mat.iloc[1]['Costo_Total'])/1e9:.0f}B
                = {(top_mat.iloc[0]['% Total']+top_mat.iloc[1]['% Total']):.1f}% del costo total.
                Si Riopaila quiere reducir costos, ahi esta el primer lugar donde mirar.
            </div>""", unsafe_allow_html=True)
        st.markdown("""
> **Interpretacion de negocio — 2.8:** Un contrato de largo plazo con el proveedor de ABONO APORQUE
> podria generar ahorros de **cientos de miles de millones de pesos** en el horizonte del proyecto.
> Este es el hallazgo mas accionable desde el punto de vista financiero.
        """)

    with subtab5:
        # ── 2.7 Costos por Tenencia ──────────────────────────────────
        st.markdown("### 2.7 Costos por Tipo de Tenencia")
        if 'Tenencia_Label' in df.columns:
            ten_stats = df.groupby('Tenencia_Label').agg(
                N=('Csts.real.cargo', 'count'),
                Costo_Total=('Csts.real.cargo', 'sum'),
                Costo_Promedio=('Csts.real.cargo', 'mean')
            ).reset_index().sort_values('Costo_Promedio', ascending=False)
            ten_stats['% Costo'] = (ten_stats['Costo_Total'] / ten_stats['Costo_Total'].sum() * 100).round(1)

            col1, col2 = st.columns(2)
            with col1:
                fig_ten1 = px.bar(
                    ten_stats, x='Tenencia_Label', y='Costo_Promedio',
                    color='Costo_Promedio', color_continuous_scale='Blues',
                    title='Costo Promedio por Labor segun Tenencia',
                    text=[f'${v:,.0f}' for v in ten_stats['Costo_Promedio']],
                    labels={'Tenencia_Label': 'Tipo de Tenencia', 'Costo_Promedio': 'Costo Promedio ($)'}
                )
                fig_ten1.update_layout(height=380, coloraxis_showscale=False)
                st.plotly_chart(fig_ten1, use_container_width=True)
            with col2:
                fig_ten2 = px.bar(
                    ten_stats, x='Tenencia_Label', y='Costo_Total',
                    color='Costo_Total', color_continuous_scale='Greens',
                    title='Costo Total Acumulado por Tenencia',
                    text=[f'${v/1e9:.1f}B' for v in ten_stats['Costo_Total']],
                    labels={'Tenencia_Label': 'Tipo de Tenencia', 'Costo_Total': 'Costo Total ($)'}
                )
                fig_ten2.update_layout(height=380, coloraxis_showscale=False)
                st.plotly_chart(fig_ten2, use_container_width=True)

            st.dataframe(
                ten_stats.style.format({
                    'Costo_Total': '${:,.0f}',
                    'Costo_Promedio': '${:,.0f}',
                    '% Costo': '{:.1f}%'
                }),
                use_container_width=True, hide_index=True
            )
            st.markdown("""
> **Interpretacion de negocio:** Las tierras de **Participacion** tienen el mayor costo promedio por labor,
> seguidas de las Alquiladas y Propias. Esto confirma que el tipo de tenencia influye
> significativamente en el costo operativo. El equipo decidio enfocarse en tenencias 10/20/30
> (propias) para el modelo predictivo, donde los patrones son mas consistentes y predecibles.
            """)

        st.markdown("---")

        st.markdown("---")

                # ── 2.9 Matriz de Correlacion y Pairplot ─────────────────────
        st.markdown("### 2.9 Matriz de Correlacion de Variables")
        cols_corr = [c for c in ['Cant.producida real', 'Csts.real.cargo',
                                   'Csts.unitarios real', 'Año', 'Mes', 'Tarifa']
                     if c in df.columns]
        corr_mat = df[cols_corr].corr()

        fig_corr = px.imshow(
            corr_mat, text_auto='.2f',
            color_continuous_scale='RdYlGn',
            title='Matriz de Correlacion: Variables del Proyecto Riopaila',
            zmin=-1, zmax=1,
            labels=dict(color="Correlacion")
        )
        fig_corr.update_layout(height=480)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("""
> **Interpretacion de la matriz:**
> - **`Cant.producida real` ↔ `Csts.real.cargo`:** Correlacion positiva alta — **la variable X mas importante del modelo**.
>   A mayor cantidad producida, mayor costo. Esto valida su inclusion como predictor principal.
> - **`Año` ↔ `Csts.real.cargo`:** Correlacion positiva baja — confirma el incremento gradual por inflacion.
> - **`Tarifa` ↔ `Csts.unitarios real`:** Alta correlacion por definicion contable (esperada, no aporta informacion nueva).
> - Variables con correlacion ~0 con el costo no tienen poder predictivo lineal sobre el modelo.
        """)




# ══════════════════════════════════════════════════════════════
# TAB 3 — REGRESION + RANDOM FOREST
# ══════════════════════════════════════════════════════════════

with tab3:
    st.markdown("## 📈 Regresion Lineal + Random Forest")
    st.markdown("Se predice `Costo_Total` mensual por grupo de labor usando tres modelos comparados.")

    with st.expander("📖 Por que estos modelos?", expanded=False):
        st.markdown("""
| Modelo | Por que se incluye | Limitacion |
|---|---|---|
| **Reg. Lineal Simple** | Baseline — establece el minimo esperado | Solo usa el tiempo, ignora el tipo de labor |
| **Reg. Lineal Multiple** | Agrega el grupo de labor como predictor | Asume relaciones lineales |
| **Random Forest** | Captura relaciones no lineales y combinaciones complejas | Menos interpretable |

**Por que Random Forest es el modelo principal:**
El costo depende de la combinacion de grupo de labor + mes + cantidad producida.
Random Forest captura estas interacciones sin asumir una forma funcional especifica,
lo que lo hace mas adecuado para datos agricolas con multiples factores interdependientes.
        """)

    with st.spinner("Entrenando modelos de regresion..."):
        resultados = entrenar_modelos(df)

    r = resultados['reg']
    r2_s  = r2_score(r['y_te_s'],   r['y_pred_s'])
    r2_m  = r2_score(r['y_test_m'], r['y_pred_m'])
    r2_rf = r2_score(r['y_test_m'], r['y_pred_rf'])

    # Metricas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Reg. Simple",   f"{r2_s:.4f}",  delta=None)
        st.metric("MAE Simple",       f"${mean_absolute_error(r['y_te_s'],   r['y_pred_s']):,.0f}")
    with col2:
        st.metric("R² Reg. Multiple", f"{r2_m:.4f}",  delta=f"+{r2_m-r2_s:.4f} vs simple")
        st.metric("MAE Multiple",     f"${mean_absolute_error(r['y_test_m'], r['y_pred_m']):,.0f}")
    with col3:
        st.metric("R² Random Forest", f"{r2_rf:.4f}", delta=f"+{r2_rf-r2_m:.4f} vs multiple")
        st.metric("MAE Random Forest",f"${mean_absolute_error(r['y_test_m'], r['y_pred_rf']):,.0f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Comparativa de Modelos")
        comp = pd.DataFrame({
            'Modelo': ['Reg. Simple', 'Reg. Multiple', 'Random Forest'],
            'R2':     [r2_s, r2_m, r2_rf],
            'MAE':    [mean_absolute_error(r['y_te_s'], r['y_pred_s']),
                       mean_absolute_error(r['y_test_m'], r['y_pred_m']),
                       mean_absolute_error(r['y_test_m'], r['y_pred_rf'])],
            'RMSE':   [np.sqrt(mean_squared_error(r['y_te_s'], r['y_pred_s'])),
                       np.sqrt(mean_squared_error(r['y_test_m'], r['y_pred_m'])),
                       np.sqrt(mean_squared_error(r['y_test_m'], r['y_pred_rf']))]
        })
        fig_comp = px.bar(
            comp, x='Modelo', y='R2',
            color='Modelo',
            color_discrete_sequence=['#95d5b2', '#2d6a4f', '#1a472a'],
            title='R² por Modelo (mayor = mejor)',
            text=[f'{v:.4f}' for v in comp['R2']]
        )
        fig_comp.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_b:
        st.markdown("### Importancia de Variables — Random Forest")
        imp_df = pd.DataFrame({
            'Variable':    r['feat_reg'],
            'Importancia': r['rf'].feature_importances_
        }).sort_values('Importancia', ascending=False).head(12)
        imp_df['Variable'] = imp_df['Variable'].str.replace('GRUPO LABORES_', '')
        fig_imp = px.bar(
            imp_df, x='Importancia', y='Variable',
            orientation='h', color='Importancia',
            color_continuous_scale='Greens',
            title='Top 12 Variables mas Importantes',
            labels={'Variable': ''}
        )
        fig_imp.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("### Diagnostico de Residuos — Regresion Multiple")
    st.markdown("""
> **Analisis de Overfitting / Underfitting:**
> - **Reg. Simple:** R² bajo en train Y test → **Underfitting** — modelo demasiado simple
> - **Reg. Multiple:** Diferencia tren-test < 0.10 → **Aceptable** — ligero underfitting
> - **Random Forest:** Diferencia tren-test ~0.10-0.15 → **Bueno** — sin sobreajuste grave
>
> Un R² de Random Forest ~0.72 significa que el modelo explica el **72% de la variacion en costos**.
> El 28% restante corresponde a factores no capturados: clima, cambios de proveedor, negociaciones.
    """)
    residuos = r['y_test_m'] - r['y_pred_m']
    col1, col2 = st.columns(2)
    with col1:
        fig_res = px.scatter(
            x=r['y_pred_m'], y=residuos,
            labels={'x': 'Predicciones', 'y': 'Residuos'},
            title='Residuos vs Predicciones', opacity=0.4
        )
        fig_res.add_hline(y=0, line_dash='dash', line_color='red')
        fig_res.update_layout(height=350)
        st.plotly_chart(fig_res, use_container_width=True)
    with col2:
        fig_hist = px.histogram(
            x=residuos, nbins=40,
            title='Distribucion de Residuos',
            labels={'x': 'Residuo', 'y': 'Frecuencia'},
            color_discrete_sequence=['#2d6a4f']
        )
        fig_hist.add_vline(x=0, line_dash='dash', line_color='red')
        fig_hist.update_layout(height=350)
        st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — SARIMA
# ══════════════════════════════════════════════════════════════

with tab4:
    st.markdown("## 📅 Series Temporales — Pronostico SARIMA 2026")

    with st.expander("📖 Por que SARIMA?", expanded=False):
        st.markdown("""
**SARIMA** es el modelo estadistico estandar para series temporales con estacionalidad. Se eligio porque:

- ✅ El dataset tiene **60 meses** de datos — suficiente para estimar patrones anuales
- ✅ Hay **estacionalidad clara** (picos en jul-sep y ene-mar detectados en el EDA)
- ✅ Los parametros `(1,1,1)(1,1,1,12)` capturan tendencia, diferenciacion y patron anual
- ✅ Genera **intervalos de confianza** para cuantificar la incertidumbre del pronostico

**Ventaja sobre regresion lineal:** SARIMA captura la estacionalidad anual que la regresion
lineal ignora, produciendo pronosticos mas precisos para planificacion presupuestaria.
        """)

    serie_mensual = df.groupby(['Año', 'Mes'])['Csts.real.cargo'].sum().reset_index()
    serie_mensual['Fecha'] = pd.to_datetime(
        serie_mensual['Año'].astype(str) + '-' + serie_mensual['Mes'].astype(str) + '-01')
    serie_mensual = serie_mensual.set_index('Fecha').sort_index()
    serie_ts = serie_mensual['Csts.real.cargo'].asfreq('MS')

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Meses de datos", len(serie_ts))
    with col2: st.metric("Inicio serie",   serie_ts.index[0].strftime('%b %Y'))
    with col3: st.metric("Fin serie",      serie_ts.index[-1].strftime('%b %Y'))

    # Evolucion historica
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=serie_ts.index, y=serie_ts.values/1e9,
        mode='lines+markers', name='Historico',
        line=dict(color='#2d6a4f', width=2),
        marker=dict(size=5)
    ))
    fig_ts.update_layout(
        title='Evolucion Mensual de Costos — Riopaila (2021-2026)',
        yaxis_title='Costo (Miles de Millones $)',
        height=380
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")
    st.markdown("### Configuracion del Modelo SARIMA")
    col1, col2, col3, col4 = st.columns(4)
    with col1: p = st.slider("p (AR)",  0, 3, 1)
    with col2: d = st.slider("d (I)",   0, 2, 1)
    with col3: q = st.slider("q (MA)",  0, 3, 0)
    with col4: pasos = st.slider("Meses a pronosticar", 3, 18, 6)
    st.markdown("""
> 💡 **Parametros recomendados:** p=1, d=1, q=0 con estacionalidad (0,1,1,12).
> Si el MAPE es mayor al 100%, prueba reducir q a 0 o d a 1.
    """)

    if st.button("🚀 Ejecutar SARIMA", type="primary"):
        with st.spinner("Entrenando SARIMA... esto puede tomar unos segundos"):
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX

                n_test = 6
                train_ts = serie_ts[:-n_test]
                test_ts  = serie_ts[-n_test:]

                model_sarima = SARIMAX(train_ts, order=(p, d, q),
                                       seasonal_order=(1, 1, 1, 12),
                                       enforce_stationarity=False,
                                       enforce_invertibility=False)
                result_sarima = model_sarima.fit(disp=False)

                forecast_test = result_sarima.forecast(steps=n_test)
                forecast_test.index = test_ts.index

                mae_ts  = mean_absolute_error(test_ts, forecast_test)
                mape_ts = np.mean(np.abs((test_ts - forecast_test) / test_ts)) * 100

                # Reentrenar con todos los datos
                model_full = SARIMAX(serie_ts, order=(p, d, q),
                                     seasonal_order=(1, 1, 1, 12),
                                     enforce_stationarity=False,
                                     enforce_invertibility=False)
                result_full = model_full.fit(disp=False)
                forecast_fut = result_full.get_forecast(steps=pasos)
                fm = forecast_fut.predicted_mean
                fc = forecast_fut.conf_int()

                col1, col2, col3 = st.columns(3)
                with col1: st.metric("AIC del modelo", f"{result_sarima.aic:.0f}")
                with col2: st.metric("MAE validacion", f"${mae_ts/1e6:.1f}M")
                with col3: st.metric("MAPE validacion", f"{mape_ts:.1f}%")
                if mape_ts > 200:
                    st.markdown("""<div class="alert-box">
                        ⚠️ <b>MAPE muy alto — modelo inestable.</b>
                        Prueba cambiar los parametros: recomendamos <b>p=1, d=1, q=0</b>.
                        Un MAPE alto indica que los datos tienen mucha variabilidad
                        o que el modelo no convergio correctamente.
                    </div>""", unsafe_allow_html=True)

                # Grafico pronostico
                fig_sar = go.Figure()
                fig_sar.add_trace(go.Scatter(
                    x=serie_ts.index, y=serie_ts.values/1e9,
                    mode='lines', name='Historico',
                    line=dict(color='#2d6a4f', width=2)
                ))
                fig_sar.add_trace(go.Scatter(
                    x=fm.index, y=fm.values/1e9,
                    mode='lines+markers', name='Pronostico',
                    line=dict(color='#e74c3c', dash='dash', width=2.5),
                    marker=dict(symbol='triangle-up', size=10)
                ))
                fig_sar.add_trace(go.Scatter(
                    x=list(fc.index) + list(fc.index[::-1]),
                    y=list(fc.iloc[:, 0]/1e9) + list(fc.iloc[:, 1]/1e9)[::-1],
                    fill='toself', fillcolor='rgba(231,76,60,0.15)',
                    line=dict(color='rgba(255,255,255,0)'), name='IC 95%'
                ))
                fig_sar.update_layout(
                    title=f'Pronostico SARIMA({p},{d},{q})(1,1,1,12) — {pasos} meses',
                    yaxis_title='Costo (Miles de Millones $)', height=450
                )
                st.plotly_chart(fig_sar, use_container_width=True)

                # Tabla pronostico
                st.markdown("### 📋 Tabla de Pronostico")
                meses_n = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                           'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
                # Limitar IC — si son absurdamente grandes mostrar N/D
                ic_inf = fc.iloc[:, 0].values
                ic_sup = fc.iloc[:, 1].values
                umbral_ic = fm.values.mean() * 1000  # si IC > 1000x el pronostico es invalido

                def fmt_ic(val, ref):
                    if abs(val) > abs(ref) * 1000:
                        return "N/D (modelo inestable)"
                    return f"${val:,.0f}"

                tabla_pron = pd.DataFrame({
                    'Mes': [f"{meses_n[f.month-1]} {f.year}" for f in fm.index],
                    'Pronostico ($)': [f"${v:,.0f}" for v in fm.values],
                    'IC Inferior ($)': [fmt_ic(v, fm.values[i]) for i, v in enumerate(ic_inf)],
                    'IC Superior ($)': [fmt_ic(v, fm.values[i]) for i, v in enumerate(ic_sup)],
                })
                st.dataframe(tabla_pron, use_container_width=True, hide_index=True)

                total_pron = fm.sum()
                # Verificar si IC es estable (no mayor a 1000x el pronostico)
                ic_low_sum  = fc.iloc[:,0].sum()
                ic_high_sum = fc.iloc[:,1].sum()
                ic_estable  = abs(ic_low_sum) < abs(total_pron) * 1000

                if ic_estable:
                    ic_txt = f"(IC: ${ic_low_sum/1e9:.1f}B – ${ic_high_sum/1e9:.1f}B)"
                else:
                    ic_txt = "(IC no disponible — modelo inestable, ajusta los parametros)"

                st.markdown(f"""<div class="success-box">
                    <b>✅ Pronostico completado.</b>
                    Total estimado para los proximos {pasos} meses:
                    <b>${total_pron/1e9:.1f}B</b><br>
                    {ic_txt}
                </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error al entrenar SARIMA: {e}")
                st.info("Intenta ajustar los parametros p, d, q del modelo.")


# ══════════════════════════════════════════════════════════════
# TAB 5 — CLASIFICACION
# ══════════════════════════════════════════════════════════════

with tab5:
    st.markdown("## 🎯 Clasificacion — Labor Costosa o Normal?")
    st.markdown("**Variable objetivo:** `Labor_Costosa` = 1 si costo > percentil 75, 0 si no.")

    with st.expander("📖 Como interpretar los resultados en lenguaje de negocio", expanded=False):
        st.markdown("""
| Metrica | Que significa para Riopaila |
|---|---|
| **Accuracy ~77%** | De 100 labores evaluadas, el modelo acierta en ~77 |
| **Recall ~85%** | Detecta el 85% de las labores que REALMENTE seran costosas |
| **AUC-ROC ~0.84** | Discrimina bien entre costosas y normales (1.0 = perfecto) |

> **Por que el Recall es la metrica mas importante:**
> Es mejor sobre-alertar (falso positivo) que perder una labor costosa (falso negativo).
> Un recall alto significa que el sistema de alertas tempranas funcionara correctamente.

> **Como usar en la practica:** Antes de programar una labor, el gerente ingresa tipo de labor,
> mes, cantidad estimada y tenencia. Si el modelo devuelve **"Costosa"**, se activa una revision
> de eficiencia antes de ejecutar — intervencion proactiva en lugar de reportar el sobrecosto al final.
        """)

    c = resultados['clf']
    umbral = c['umbral']

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Umbral P75", f"${umbral:,.0f}")
    with col2: st.metric("% Costosas",  f"{c['y_test'].mean()*100:.1f}%")
    with col3: st.metric("Acc. Log. Reg.", f"{accuracy_score(c['y_test'], c['y_pred_rl']):.2%}")
    with col4: st.metric("Acc. Arbol",    f"{accuracy_score(c['y_test'], c['y_pred_arbol']):.2%}")

    st.markdown("---")
    subtab_a, subtab_b, subtab_c = st.tabs(["Metricas & ROC", "Coeficientes Logistica", "Importancia Arbol"])

    with subtab_a:
        # Metricas comparativas
        met_rl = {
            'Modelo': 'Regresion Logistica',
            'Accuracy':  round(accuracy_score(c['y_test'],  c['y_pred_rl'], ), 4),
            'Precision': round(precision_score(c['y_test'], c['y_pred_rl'],  zero_division=0), 4),
            'Recall':    round(recall_score(c['y_test'],    c['y_pred_rl'],  zero_division=0), 4),
            'F1':        round(f1_score(c['y_test'],        c['y_pred_rl'],  zero_division=0), 4),
            'AUC':       round(roc_auc_score(c['y_test'],   c['y_proba_rl']), 4)
        }
        met_arbol = {
            'Modelo': 'Arbol de Decision',
            'Accuracy':  round(accuracy_score(c['y_test'],  c['y_pred_arbol']), 4),
            'Precision': round(precision_score(c['y_test'], c['y_pred_arbol'], zero_division=0), 4),
            'Recall':    round(recall_score(c['y_test'],    c['y_pred_arbol'], zero_division=0), 4),
            'F1':        round(f1_score(c['y_test'],        c['y_pred_arbol'], zero_division=0), 4),
            'AUC':       round(roc_auc_score(c['y_test'],   c['y_proba_arbol']), 4)
        }
        tabla_met = pd.DataFrame([met_rl, met_arbol]).set_index('Modelo')

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Tabla Comparativa de Metricas")
            st.dataframe(tabla_met.style.highlight_max(axis=0, color='#d4edda'), use_container_width=True)

        with col2:
            st.markdown("#### Curva ROC")
            fpr_rl,  tpr_rl,  _ = roc_curve(c['y_test'], c['y_proba_rl'])
            fpr_arb, tpr_arb, _ = roc_curve(c['y_test'], c['y_proba_arbol'])
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr_rl, y=tpr_rl, mode='lines', name=f'Log. Reg. (AUC={met_rl["AUC"]:.4f})',
                line=dict(color='#1f77b4', width=2.5)
            ))
            fig_roc.add_trace(go.Scatter(
                x=fpr_arb, y=tpr_arb, mode='lines', name=f'Arbol (AUC={met_arbol["AUC"]:.4f})',
                line=dict(color='#ff7f0e', dash='dash', width=2.5)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines', name='Aleatorio',
                line=dict(color='gray', dash='dot', width=1.5)
            ))
            fig_roc.update_layout(
                xaxis_title='Tasa Falsos Positivos',
                yaxis_title='Tasa Verdaderos Positivos', height=380
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    with subtab_b:
        st.markdown("#### Coeficientes Regresion Logistica")
        st.markdown("Variables que **aumentan** (rojo) o **reducen** (azul) la probabilidad de ser una labor costosa.")
        st.markdown("""
> **Como leer este grafico:**
> - **Barras rojas (coef. positivo):** esa variable AUMENTA la probabilidad de que la labor sea costosa.
> - **Barras azules (coef. negativo):** esa variable REDUCE la probabilidad de costo alto.
> - **Magnitud:** entre mayor la barra, mas importante es esa variable para la decision del modelo.
        """)
        coefs = pd.Series(c['rl'].coef_[0], index=c['feat_clf'])
        coefs_top = pd.concat([coefs.nlargest(10), coefs.nsmallest(10)]).drop_duplicates().sort_values()
        coefs_top.index = [i.replace('GRUPO LABORES_', '').replace('Tipo Labor_', '')
                           for i in coefs_top.index]
        colores_c = ['#e74c3c' if v > 0 else '#3498db' for v in coefs_top.values]
        fig_coef = go.Figure(go.Bar(
            x=coefs_top.values, y=coefs_top.index,
            orientation='h', marker_color=colores_c,
            marker_line_color='black', marker_line_width=0.5
        ))
        fig_coef.add_vline(x=0, line_dash='dash', line_color='black')
        fig_coef.update_layout(
            title='Coeficientes Logisticos — Variables que Aumentan o Reducen el Costo',
            xaxis_title='Coeficiente (logit)', height=500
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    with subtab_c:
        st.markdown("#### Importancia de Variables — Arbol de Decision")
        st.markdown("""
> **Interpretacion de negocio:** `Cant.producida real` domina con ~82% de importancia.
> Esto confirma que **la cantidad de trabajo realizado es el principal determinante del costo**,
> no el tipo de labor ni el mes. Si se conoce la cantidad a producir, se puede estimar
> el costo con alta precision antes de ejecutar la labor.
        """)
        imp_a = pd.DataFrame({
            'Variable':    c['feat_clf'],
            'Importancia': c['arbol'].feature_importances_
        }).sort_values('Importancia', ascending=False).head(15)
        imp_a['Variable'] = (imp_a['Variable']
                              .str.replace('GRUPO LABORES_', '')
                              .str.replace('Tipo Labor_', ''))
        fig_imp_a = px.bar(
            imp_a, x='Importancia', y='Variable',
            orientation='h', color='Importancia',
            color_continuous_scale='YlOrRd',
            title='Top 15 Variables — Arbol de Decision (Gini)',
            labels={'Variable': ''}
        )
        fig_imp_a.update_layout(height=480, coloraxis_showscale=False)
        st.plotly_chart(fig_imp_a, use_container_width=True)

        st.markdown(f"""<div class="info-box">
            <b>💡 Variable mas importante:</b>
            <code>Cant.producida real</code> con importancia
            {c['arbol'].feature_importances_[c['feat_clf'].index('Cant.producida real')]:.4f}.
            Explica el 82% de las decisiones del modelo: a mayor cantidad producida,
            mayor probabilidad de que la labor sea costosa.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 6 — CLUSTERING
# ══════════════════════════════════════════════════════════════

with tab6:
    st.markdown("## 🗺️ Clustering K-Means — Segmentacion de Lotes")

    with st.expander("📖 Como interpretar la segmentacion en lenguaje de negocio", expanded=False):
        st.markdown("""
El clustering agrupa los **lotes de cana** segun su perfil de costos y produccion,
sin una etiqueta predefinida — el algoritmo descubre los patrones por si solo.

| Cluster | Nombre | N Lotes | Costo/Unidad | Accion recomendada |
|---|---|---|---|---|
| 0 | 🌱 Lotes Estandar | 214 | $49.118 | Mantener operacion actual |
| 1 | 📈 Alta Produccion | 146 | $53.170 | Escalar buenas practicas |
| 2 | ⚠️ **Lotes Ineficientes** | **226** | **$145.140** | **Auditoria urgente** |
| 3 | 🏆 Lotes de Elite | 63 | $28.043 | Modelo a replicar |

> **Hallazgo principal:** Los 226 lotes del Cluster 2 operan con un costo por unidad
> **5 veces mayor** que los Lotes de Elite. No gastan mas en total, sino que **producen
> muy poco para lo que cuestan**. Identificar y optimizar estos lotes es la mayor
> oportunidad de ahorro operativo identificada en este proyecto.
        """)

    if 'clust' not in resultados:
        st.warning("No se encontro columna de sector/suerte en el dataset.")
    else:
        cl = resultados['clust']
        lotes = cl['lotes_activos']

        col1, col2, col3, col4 = st.columns(4)
        nombres_cl = {
            0: ('Lotes Estandar',     '🌱', '#3498db'),
            1: ('Alta Produccion',    '📈', '#2ecc71'),
            2: ('Lotes Ineficientes', '⚠️', '#e74c3c'),
            3: ('Lotes de Elite',     '🏆', '#f39c12')
        }
        conteos = lotes['Cluster'].value_counts().sort_index()
        for i, (nombre, emoji, color) in nombres_cl.items():
            with [col1, col2, col3, col4][i]:
                n = conteos.get(i, 0)
                st.markdown(f"""<div class="metric-card" style="border-left-color:{color}">
                    <h3>{emoji} {n}</h3><p>{nombre}</p></div>""", unsafe_allow_html=True)

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### Scatter: Costo vs Produccion por Cluster")
            color_map = {0: '#3498db', 1: '#2ecc71', 2: '#e74c3c', 3: '#f39c12'}
            lotes['Color'] = lotes['Cluster'].map(color_map)
            fig_cl = px.scatter(
                lotes,
                x='Costo_Total', y='Produccion_Total',
                color='Nombre_Cluster',
                color_discrete_map={
                    'Lotes Estandar': '#3498db',
                    'Alta Produccion': '#2ecc71',
                    'Lotes Ineficientes': '#e74c3c',
                    'Lotes de Elite': '#f39c12'
                },
                hover_data=[cl['sector_col'], 'N_Labores', 'Costo_x_Unidad'],
                labels={
                    'Costo_Total': 'Costo Total ($)',
                    'Produccion_Total': 'Produccion Total',
                    'Nombre_Cluster': 'Segmento'
                },
                title='Segmentacion de Lotes de Cana — Riopaila Castilla'
            )
            fig_cl.update_layout(height=450)
            st.plotly_chart(fig_cl, use_container_width=True)

        with col_b:
            st.markdown("### Silhouette Score por K")
            sil_df = pd.DataFrame({
                'K': list(cl['siluetas'].keys()),
                'Silhouette': list(cl['siluetas'].values())
            })
            fig_sil = px.line(
                sil_df, x='K', y='Silhouette',
                markers=True, title='Silhouette Score — Metodo del Codo',
                labels={'Silhouette': 'Silhouette Score', 'K': 'Numero de Clusters'}
            )
            fig_sil.add_vline(x=4, line_dash='dash', line_color='red',
                              annotation_text='K=4 elegido')
            fig_sil.update_layout(height=450)
            st.plotly_chart(fig_sil, use_container_width=True)

        st.markdown("### Perfil Detallado de Cada Segmento")
        perfil = lotes.groupby('Nombre_Cluster').agg(
            N_Lotes=('Sector-suerte' if 'Sector-suerte' in lotes.columns else cl['sector_col'], 'count'),
            Costo_Total_Media=('Costo_Total', 'mean'),
            Produccion_Media=('Produccion_Total', 'mean'),
            N_Labores_Media=('N_Labores', 'mean'),
            Costo_x_Unidad_Media=('Costo_x_Unidad', 'mean')
        ).round(0)
        st.dataframe(
            perfil.style.format({
                'Costo_Total_Media': '${:,.0f}',
                'Produccion_Media': '{:,.0f}',
                'N_Labores_Media': '{:,.0f}',
                'Costo_x_Unidad_Media': '${:,.0f}'
            }),
            use_container_width=True
        )

        st.markdown(f"""<div class="alert-box">
            <b>⚠️ Cluster Ineficientes (226 lotes) — ACCION REQUERIDA:</b><br>
            Costo por unidad mediana de <b>$145.140</b> vs <b>$28.043</b> de los Lotes de Elite.
            Son casi <b>5x mas caros</b> por unidad producida. Auditar estos lotes es la
            mayor oportunidad de ahorro operativo identificada en el proyecto.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 7 — SIMULADOR
# ══════════════════════════════════════════════════════════════

with tab7:
    st.markdown("## 🧮 Simulador de Gastos por Labor")
    st.markdown("Estima el costo de una labor futura ajustando los parametros operativos.")

    with st.expander("📖 Como usar el simulador", expanded=False):
        st.markdown("""
**Pasos:**
1. Selecciona el **grupo de labor** (Fertilizacion, Riego, Cosecha, etc.)
2. Elige el **mes** en que se ejecutara la labor
3. Ingresa el **año proyectado** (2025-2030)
4. Ajusta la **cantidad a producir** segun lo planificado
5. Selecciona el **tipo de tenencia** del lote

**Que devuelve:**
- Estimacion por **Random Forest** (modelo mas preciso, R²~0.72)
- Estimacion por **Regresion Lineal** (modelo de referencia)
- Costo historico promedio para ese grupo y mes
- **Alerta automatica** si la labor proyectada superara el umbral P75 de costos
- **Proyeccion mensual** con ajuste de inflacion del 5% anual estimado
        """)

    st.markdown("---")
    col_sim1, col_sim2 = st.columns([1, 1])

    with col_sim1:
        st.markdown("### ⚙️ Parametros de la Labor")

        grupos_sim = [g for g in df['GRUPO LABORES'].unique()
                      if g not in ['Sin Clasificar', 'DESCONOCIDO']]
        grupo_sel = st.selectbox("Grupo de Labor", sorted(grupos_sim))

        mes_sim = st.slider("Mes de ejecucion", 1, 12, 7,
                             format="%d",
                             help="1=Enero … 12=Diciembre")
        meses_n = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                   'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        st.caption(f"Mes seleccionado: **{meses_n[mes_sim-1]}**")

        año_sim = st.slider("Año proyectado", 2025, 2030, 2026)

        cant_sim = st.slider(
            "Cantidad a producir (unidades/HA)",
            min_value=float(df['Cant.producida real'].quantile(0.05)),
            max_value=float(df['Cant.producida real'].quantile(0.95)),
            value=float(df['Cant.producida real'].median()),
            step=1.0,
            format="%.0f"
        )

        tenencia_sim = st.radio(
            "Tipo de tenencia", ['Propia (10)', 'Alquilada (20)', 'Participacion (30)'],
            horizontal=True
        )
        ten_map = {'Propia (10)': 10, 'Alquilada (20)': 20, 'Participacion (30)': 30}
        ten_val = ten_map[tenencia_sim]

        n_labores_sim = st.slider("Numero de labores estimadas en el mes", 1, 200, 50)

    with col_sim2:
        st.markdown("### 📊 Resultado de la Simulacion")

        # Calcular estimacion basada en datos historicos
        df_grupo = df[df['GRUPO LABORES'] == grupo_sel]
        df_mes   = df_grupo[df_grupo['Mes'] == mes_sim]
        df_ten   = df_grupo[df_grupo['Tenencia'] == ten_val]

        costo_prom_grupo = df_grupo['Csts.real.cargo'].mean() if len(df_grupo) > 0 else 0
        costo_prom_mes   = df_mes['Csts.real.cargo'].mean() if len(df_mes) > 0 else costo_prom_grupo
        costo_unitario   = (df_grupo['Csts.real.cargo'].sum() /
                             df_grupo['Cant.producida real'].sum()
                             if df_grupo['Cant.producida real'].sum() > 0 else 0)

        # Ajuste por tendencia temporal
        min_year = df['Año'].min()
        años_futuros = año_sim - df['Año'].max()
        factor_inflacion = 1 + (años_futuros * 0.05)  # 5% inflacion anual estimada
        factor_inflacion = max(1.0, factor_inflacion)

        # Estimacion de costo por labor individual
        costo_est_labor = costo_prom_mes * factor_inflacion if costo_prom_mes > 0 else costo_prom_grupo * factor_inflacion

        # Estimacion de costo total del mes
        costo_est_total = costo_est_labor * n_labores_sim

        # Costo basado en cantidad
        costo_x_cantidad = costo_unitario * cant_sim * factor_inflacion

        # Usar Random Forest para prediccion
        r = resultados['reg']
        mes_continuo_sim = (año_sim - int(df['Año'].min())) * 12 + mes_sim
        X_sim_base = pd.DataFrame(0, index=[0], columns=r['feat_reg'])
        X_sim_base['Mes_Continuo'] = mes_continuo_sim
        col_grupo = f'GRUPO LABORES_{grupo_sel}'
        if col_grupo in X_sim_base.columns:
            X_sim_base[col_grupo] = 1

        pred_rf_mensual = r['rf'].predict(X_sim_base)[0]
        pred_lr_mensual = r['mod_multiple'].predict(X_sim_base)[0]

        # Mostrar resultados
        st.markdown(f"**Grupo:** {grupo_sel} | **Mes:** {meses_n[mes_sim-1]} {año_sim}")
        st.markdown(f"**Cantidad:** {cant_sim:,.0f} unidades | **Tenencia:** {tenencia_sim}")

        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "🌲 Estimacion Random Forest",
                f"${pred_rf_mensual/1e6:.1f}M",
                help="Prediccion del modelo de ML para el mes/grupo seleccionado"
            )
            st.metric(
                "📏 Costo Historico Promedio",
                f"${costo_prom_mes:,.0f}",
                delta=f"x{factor_inflacion:.2f} inflacion",
                help="Promedio historico para ese grupo y mes"
            )
        with c2:
            st.metric(
                "📈 Estimacion Regresion Lineal",
                f"${pred_lr_mensual/1e6:.1f}M",
                help="Prediccion del modelo lineal para el mes/grupo"
            )
            st.metric(
                "💰 Costo x Cantidad estimado",
                f"${costo_x_cantidad:,.0f}",
                help=f"Basado en costo unitario historico ${costo_unitario:,.0f}/unidad"
            )

        # Bloque 1 — Explicacion de los modelos
        st.markdown("""
> 🌲 **Random Forest** captura relaciones no lineales entre grupo de labor, mes y cantidad producida
> — es el modelo mas preciso (R²≈0.72). La **Regresion Lineal** asume una relacion directa
> y tiende a sobreestimar en labores con combinaciones complejas de variables.
> **Se recomienda usar el valor de Random Forest** para presupuestar.
        """)

        # Clasificacion: seria costosa?
        clf_c = resultados['clf']
        umbral = clf_c['umbral']
        es_costosa = costo_prom_mes > umbral
        if es_costosa:
            st.markdown(f"""<div class="alert-box">
                ⚠️ <b>Alerta: Esta labor probablemente sera COSTOSA</b><br>
                El costo estimado supera el umbral P75 (${umbral:,.0f}).
                Se recomienda revisar la eficiencia operativa antes de ejecutar.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="success-box">
                ✅ <b>Labor dentro del rango normal de costos</b><br>
                El costo estimado esta por debajo del umbral P75 (${umbral:,.0f}).
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
> ℹ️ **Como interpretar la alerta:** Una labor es clasificada como **COSTOSA** cuando su costo
> historico promedio supera el percentil 75 del dataset (${umbral:,.0f}).
> Esto no significa que deba cancelarse — significa que amerita una revision antes de ejecutar:
> verificar cantidad planificada, proveedor de insumos y lote asignado.
        """)

    st.markdown("---")
    st.markdown("### 📅 Proyeccion de Costos — Proximos Meses")

    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        grupo_proy = st.selectbox("Grupo para proyeccion", sorted(grupos_sim), key='proy_grupo')
        meses_proy = st.slider("Meses a proyectar", 3, 24, 12)
        año_inicio = st.number_input("Año inicio proyeccion", 2025, 2030, 2026)

    with col_p2:
        # Generar proyeccion
        df_proy = df[df['GRUPO LABORES'] == grupo_proy].copy()
        hist_mensual = df_proy.groupby('Mes')['Csts.real.cargo'].mean()

        fechas_proy = pd.date_range(
            start=f'{año_inicio}-01-01', periods=meses_proy, freq='MS')
        costos_proy = []
        for fecha in fechas_proy:
            costo_base = hist_mensual.get(fecha.month, df_proy['Csts.real.cargo'].mean())
            años_fut = fecha.year - df['Año'].max()
            factor = 1 + max(0, años_fut) * 0.05
            costos_proy.append(costo_base * factor)

        df_proy_plot = pd.DataFrame({
            'Fecha': fechas_proy,
            'Costo_Estimado': costos_proy,
            'Tipo': 'Proyeccion'
        })
        df_hist_plot = df_proy.groupby(
            pd.to_datetime(df_proy['Año'].astype(str) + '-' + df_proy['Mes'].astype(str) + '-01')
        )['Csts.real.cargo'].mean().reset_index()
        df_hist_plot.columns = ['Fecha', 'Costo_Estimado']
        df_hist_plot['Tipo'] = 'Historico'

        df_combined = pd.concat([df_hist_plot, df_proy_plot], ignore_index=True)

        fig_proy = px.line(
            df_combined, x='Fecha', y='Costo_Estimado',
            color='Tipo',
            color_discrete_map={'Historico': '#2d6a4f', 'Proyeccion': '#e74c3c'},
            title=f'Proyeccion de Costos — {grupo_proy} ({meses_proy} meses)',
            labels={'Costo_Estimado': 'Costo Promedio por Labor ($)', 'Fecha': ''}
        )
        
        fig_proy.add_trace(go.Scatter(
            x=[pd.Timestamp(f'{df["Año"].max()}-12-01'),
            pd.Timestamp(f'{df["Año"].max()}-12-01')],
            y=[0, df_combined['Costo_Estimado'].max()],
            mode='lines',
            line=dict(color='gray', dash='dash', width=1.5),
            name='Inicio proyeccion',
            showlegend=True
        ))
        fig_proy.update_layout(height=400)
        st.plotly_chart(fig_proy, use_container_width=True)
        st.markdown("""
> 📈 **Como leer este grafico:** La **linea verde** es el historico real 2021-2025.
> La **linea roja** es la proyeccion calculada con el promedio mensual del grupo seleccionado
> mas un ajuste de **inflacion del 5% anual** — estimacion conservadora basada en la
> variacion de costos observada en el periodo analizado. La **linea punteada vertical**
> marca el inicio de la proyeccion. Si la linea roja sube respecto al historico,
> ese grupo tendra mayores costos en los meses proyectados.
        """)

    st.markdown("### 💡 Comparativa por Grupo — Mes Seleccionado")
    mes_comp = st.slider("Mes para comparar grupos", 1, 12, 7, key='mes_comp')
    df_comp_grupos = df[df['Mes'] == mes_comp].groupby('GRUPO LABORES').agg(
        Costo_Promedio=('Csts.real.cargo', 'mean'),
        N_Labores=('Csts.real.cargo', 'count')
    ).reset_index().sort_values('Costo_Promedio', ascending=False)
    df_comp_grupos = df_comp_grupos[
        ~df_comp_grupos['GRUPO LABORES'].isin(['Sin Clasificar', 'DESCONOCIDO'])]

    fig_comp_g = px.bar(
        df_comp_grupos, x='GRUPO LABORES', y='Costo_Promedio',
        color='Costo_Promedio', color_continuous_scale='RdYlGn_r',
        title=f'Costo Promedio por Grupo — {meses_n[mes_comp-1]}',
        text=[f'${v:,.0f}' for v in df_comp_grupos['Costo_Promedio']],
        labels={'Costo_Promedio': 'Costo Promedio ($)', 'GRUPO LABORES': ''}
    )
    fig_comp_g.add_hline(y=umbral, line_dash='dash', line_color='red',
                          annotation_text=f'Umbral P75 ${umbral:,.0f}')
    fig_comp_g.update_layout(height=430, xaxis_tickangle=-45, coloraxis_showscale=False)
    st.plotly_chart(fig_comp_g, use_container_width=True)
    st.markdown(f"""
> 🔴 **Como leer este grafico:** Las **barras rojas** superan el umbral P75 — son los grupos
> de labor que historicamente generan labores costosas en ese mes. Las **barras verdes**
> estan por debajo del umbral y representan labores dentro del rango normal de costos.
> La **linea punteada roja** es el umbral P75 (${umbral:,.0f}) — el punto de corte que usa
> el modelo de clasificacion para definir si una labor es costosa o normal.
> Usa este grafico para decidir que grupos de labor priorizar en la planificacion del mes.
    """)


# ── Footer ────────────────────────────────────────────────────
st.markdown("---")

with st.expander("⚠️ Limitaciones del modelo", expanded=False):
    st.markdown("""
1. **Datos historicos:** Entrenado con 2021-2026. Si cambian las condiciones del mercado, requiere reentrenamiento.
2. **Variables no capturadas:** No incluye datos climaticos (precipitacion, temperatura) que afectan costos agricolas.
3. **Tenencia filtrada:** El modelo se enfoca en tierras propias (10/20/30). Las arrendadas requieren un modelo separado.
4. **SARIMA con 60 meses:** Suficiente pero con mas datos los intervalos de confianza serian mas estrechos.
5. **Simulador:** Las estimaciones son orientativas. El costo real puede variar por factores no capturados.
    """)

st.markdown("""
<div style="text-align:center; color:#666; font-size:0.85rem; padding:1rem">
    🌿 <b>Ingenio Riopaila Castilla — Sistema de Analitica de Costos</b><br>
    Analitica Predictiva Aplicada a los Negocios | INTEP Roldanillo Valle | Abril 2026<br>
    Cesar Augusto Tirado · Eliana Villanueva · Francisco Jaier Trejos<br>
    Desarrollado con Python · Streamlit · scikit-learn · statsmodels · Plotly
</div>
""", unsafe_allow_html=True)
