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
        background: white; padding: 1.2rem; border-radius: 10px;
        border-left: 5px solid #2d6a4f; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-card h3 { color: #2d6a4f; font-size: 1.6rem; margin: 0; }
    .metric-card p  { color: #666; font-size: 0.85rem; margin: 0.3rem 0 0 0; }
    .section-header {
        background: #f0f7f4; padding: 0.8rem 1.2rem; border-radius: 8px;
        border-left: 4px solid #2d6a4f; margin: 1.5rem 0 1rem 0;
    }
    .section-header h3 { color: #1a472a; margin: 0; font-size: 1.1rem; }
    .alert-box {
        background: #1a472a; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #52b788; margin: 1rem 0;
        color: #ffffff;
    }
    .success-box {
        background: #1b4332; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #52b788; margin: 1rem 0;
        color: #ffffff;
    }
    .info-box {
        background: #023e8a; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #48cae4; margin: 1rem 0;
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
    div[data-testid="stSidebarContent"] { background: #f8fffe; }
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

    # Filtro Tenencia 10/20/30 (decision del grupo)
    df = df[df['Tenencia'].isin([10, 20, 30])].copy()

    # Labels tenencia
    ten_labels = {10: 'Propia Baja', 20: 'Propia Media', 30: 'Propia Alta'}
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
    st.image("https://via.placeholder.com/280x80/1a472a/ffffff?text=Riopaila+Castilla",
             use_column_width=True)
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
        if 'Centro' in df_raw.columns:
            centros_disp = sorted(df_raw['Centro'].unique().tolist())
            centros_sel  = st.multiselect("Centro operativo", centros_disp, default=centros_disp)
        else:
            centros_sel = []

        st.markdown("---")
        st.markdown("### ℹ️ Dataset cargado")
        st.metric("Registros", f"{len(df_raw):,}")
        st.metric("Periodo", f"{df_raw['Año'].min()} – {df_raw['Año'].max()}")
        st.metric("Costo total", f"${df_raw['Csts.real.cargo'].sum()/1e12:.2f}B COP")
    else:
        st.info("👆 Sube el archivo Excel para comenzar")
        st.stop()

# ── Aplicar filtros ───────────────────────────────────────────
df = df_raw.copy()
df = df[df['Año'].isin(anos_sel)]
df = df[df['GRUPO LABORES'].isin(grupos_sel)]
if centros_sel and 'Centro' in df.columns:
    df = df[df['Centro'].isin(centros_sel)]

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
    st.markdown("**Objetivo:** Desarrollar un sistema de analitica de datos que permita predecir costos, identificar anomalias y segmentar patrones operativos a partir del comportamiento historico de labores 2021-2026.")

    # KPIs principales
    col1, col2, col3, col4, col5 = st.columns(5)
    costo_total = df['Csts.real.cargo'].sum()
    costo_fert = df[df['GRUPO LABORES'].str.contains('ertiliz', na=False)]['Csts.real.cargo'].sum()

    with col1:
        st.markdown(f"""<div class="metric-card">
            <h3>{len(df):,}</h3><p>Registros SAP</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <h3>${costo_total/1e12:.2f}B</h3><p>Costo Total COP</p></div>""", unsafe_allow_html=True)
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
    costo_2021 = df[df['Año'] == df['Año'].min()]['Csts.real.cargo'].sum()
    costo_max  = df.groupby('Año')['Csts.real.cargo'].sum().max()
    year_max   = df.groupby('Año')['Csts.real.cargo'].sum().idxmax()

    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(f"""<div class="alert-box">
            <b>🌱 Fertilizacion domina</b><br>
            ${costo_fert/1e9:.0f}B en 5 anos = {costo_fert/costo_total*100:.0f}% del presupuesto.
            ABONO APORQUE es el mayor gasto individual.
        </div>""", unsafe_allow_html=True)
    with h2:
        var = (costo_max - costo_2021)/costo_2021*100 if costo_2021 > 0 else 0
        st.markdown(f"""<div class="alert-box">
            <b>📈 Costos crecieron {var:.0f}%</b><br>
            Desde {df['Año'].min()} (${costo_2021/1e9:.1f}B) hasta {year_max} (${costo_max/1e9:.1f}B).
            Coincide con inflacion post-pandemia en insumos.
        </div>""", unsafe_allow_html=True)
    with h3:
        if 'Centro' in df.columns:
            pct_gi01 = df[df['Centro']=='GI01']['Csts.real.cargo'].sum()/costo_total*100
            st.markdown(f"""<div class="alert-box">
                <b>🏭 GI01 = {pct_gi01:.0f}% del costo</b><br>
                Centro Riopaila concentra casi todo el presupuesto.
                Cualquier optimizacion debe enfocarse ahi.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="info-box">
                <b>📊 {df['Sector-suerte'].nunique() if 'Sector-suerte' in df.columns else 'N/A'} lotes activos</b><br>
                Distribuidos en {df['Sector'].nunique() if 'Sector' in df.columns else 'N/A'} sectores geograficos.
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
        "Por Grupo", "Evolucion Anual", "Estacionalidad", "Top Materiales", "Centros & Tenencia"
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

    with subtab3:
        st.markdown("### 2.4 Estacionalidad Mensual")
        meses_n = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                   'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        mensual = df.groupby('Mes')['Csts.real.cargo'].sum().reset_index()
        mensual['Costo_Prom'] = mensual['Csts.real.cargo'] / df['Año'].nunique()
        mensual['Mes_Nombre'] = mensual['Mes'].apply(lambda x: meses_n[x-1] if 1 <= x <= 12 else str(x))
        mediana_m = mensual['Costo_Prom'].median()
        mensual['Color'] = mensual['Costo_Prom'].apply(
            lambda x: 'Sobre mediana' if x > mediana_m else 'Bajo mediana')

        fig = px.bar(
            mensual, x='Mes_Nombre', y='Costo_Prom',
            color='Color',
            color_discrete_map={'Sobre mediana': '#e74c3c', 'Bajo mediana': '#3498db'},
            title='Costo Mensual Promedio — Patron Estacional',
            labels={'Costo_Prom': 'Costo Mensual Promedio ($)', 'Mes_Nombre': 'Mes'},
            text=[f'${v/1e9:.1f}B' for v in mensual['Costo_Prom']]
        )
        fig.add_hline(y=mediana_m, line_dash='dash', line_color='gray',
                      annotation_text='Mediana')
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 2.5 Scatter: Cantidad vs Costo")
        year_sc = st.selectbox("Año para scatter", sorted(df['Año'].unique()), key='sc_year')
        df_sc = df[(df['Año'] == year_sc) &
                   ~df['GRUPO LABORES'].isin(['Sin Clasificar', 'DESCONOCIDO'])].dropna(
            subset=['Cant.producida real', 'Csts.real.cargo'])
        fig_sc = px.scatter(
            df_sc.sample(min(5000, len(df_sc)), random_state=42),
            x='Cant.producida real', y='Csts.real.cargo',
            color='GRUPO LABORES', opacity=0.6,
            title=f'Relacion Cantidad vs Costo — {year_sc}',
            labels={'Csts.real.cargo': 'Costo ($)', 'Cant.producida real': 'Cantidad Producida'}
        )
        fig_sc.update_layout(height=450)
        st.plotly_chart(fig_sc, use_container_width=True)

    with subtab4:
        st.markdown("### 2.8 Top 10 Materiales mas Costosos")
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

        st.markdown("### 2.9 Matriz de Correlacion")
        cols_corr = [c for c in ['Cant.producida real', 'Csts.real.cargo',
                                   'Csts.unitarios real', 'Año', 'Mes', 'Tarifa']
                     if c in df.columns]
        corr_mat = df[cols_corr].corr()
        fig_corr = px.imshow(
            corr_mat, text_auto='.2f', color_continuous_scale='RdYlGn',
            title='Matriz de Correlacion: Variables del Proyecto',
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(height=450)
        st.plotly_chart(fig_corr, use_container_width=True)

    with subtab5:
        st.markdown("### 2.7 Costos por Tipo de Tenencia")
        if 'Tenencia_Label' in df.columns:
            ten_stats = df.groupby('Tenencia_Label').agg(
                N=('Csts.real.cargo', 'count'),
                Costo_Total=('Csts.real.cargo', 'sum'),
                Costo_Promedio=('Csts.real.cargo', 'mean')
            ).reset_index().sort_values('Costo_Promedio', ascending=False)

            fig = px.bar(
                ten_stats, x='Tenencia_Label', y='Costo_Promedio',
                color='Costo_Promedio', color_continuous_scale='Blues',
                title='Costo Promedio por Tipo de Tenencia',
                text=[f'${v:,.0f}' for v in ten_stats['Costo_Promedio']],
                labels={'Tenencia_Label': 'Tipo de Tenencia', 'Costo_Promedio': 'Costo Promedio ($)'}
            )
            fig.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        if 'Centro' in df.columns:
            st.markdown("### 2.10 Costos por Centro Operativo (GI01 vs GI03)")
            centro_stats = df.groupby('Centro').agg(
                N_Labores=('Csts.real.cargo', 'count'),
                Costo_Total=('Csts.real.cargo', 'sum'),
                Costo_Promedio=('Csts.real.cargo', 'mean')
            ).reset_index()
            centro_stats['% del Total'] = (centro_stats['Costo_Total'] /
                                            centro_stats['Costo_Total'].sum() * 100).round(1)

            col1, col2 = st.columns(2)
            with col1:
                fig_c1 = px.pie(
                    centro_stats, values='Costo_Total', names='Centro',
                    title='Distribucion del Gasto por Centro',
                    color_discrete_sequence=['#2d6a4f', '#95d5b2']
                )
                fig_c1.update_layout(height=350)
                st.plotly_chart(fig_c1, use_container_width=True)
            with col2:
                fig_c2 = px.bar(
                    centro_stats, x='Centro', y='Costo_Promedio',
                    color='Centro',
                    color_discrete_sequence=['#2d6a4f', '#95d5b2'],
                    title='Costo Promedio por Labor segun Centro',
                    text=[f'${v:,.0f}' for v in centro_stats['Costo_Promedio']]
                )
                fig_c2.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_c2, use_container_width=True)

            if 'GI01' in centro_stats['Centro'].values:
                pct = centro_stats[centro_stats['Centro'] == 'GI01']['% del Total'].values[0]
                costo_gi01 = centro_stats[centro_stats['Centro'] == 'GI01']['Costo_Total'].values[0]
                st.markdown(f"""<div class="alert-box">
                    <b>🏭 GI01 concentra el {pct:.0f}% del presupuesto operativo.</b><br>
                    Un 5% de ahorro en GI01 = <b>${costo_gi01*0.05/1e9:.1f}B</b> en el periodo analizado.
                    Cualquier iniciativa de optimizacion debe enfocarse en este centro.
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — REGRESION + RANDOM FOREST
# ══════════════════════════════════════════════════════════════

with tab3:
    st.markdown("## 📈 Regresion Lineal + Random Forest")
    st.markdown("Se predice `Costo_Total` mensual por grupo de labor usando tres modelos comparados.")

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
    with col3: q = st.slider("q (MA)",  0, 3, 1)
    with col4: pasos = st.slider("Meses a pronosticar", 3, 18, 6)

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
                with col2: st.metric("MAE validacion", f"${mae_ts:,.0f}")
                with col3: st.metric("MAPE validacion", f"{mape_ts:.1f}%")

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
                tabla_pron = pd.DataFrame({
                    'Mes': [f"{meses_n[f.month-1]} {f.year}" for f in fm.index],
                    'Pronostico ($)': fm.values,
                    'IC Inferior ($)': fc.iloc[:, 0].values,
                    'IC Superior ($)': fc.iloc[:, 1].values
                })
                st.dataframe(
                    tabla_pron.style.format({
                        'Pronostico ($)': '${:,.0f}',
                        'IC Inferior ($)': '${:,.0f}',
                        'IC Superior ($)': '${:,.0f}'
                    }),
                    use_container_width=True, hide_index=True
                )

                total_pron = fm.sum()
                st.markdown(f"""<div class="success-box">
                    <b>✅ Pronostico completado.</b>
                    Total estimado para los proximos {pasos} meses:
                    <b>${total_pron/1e9:.1f}B</b>
                    (IC: ${fc.iloc[:,0].sum()/1e9:.1f}B – ${fc.iloc[:,1].sum()/1e9:.1f}B)
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
            "Tipo de tenencia", ['Propia Baja (10)', 'Propia Media (20)', 'Propia Alta (30)'],
            horizontal=True
        )
        ten_map = {'Propia Baja (10)': 10, 'Propia Media (20)': 20, 'Propia Alta (30)': 30}
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
        fig_proy.add_vline(
            x=str(pd.Timestamp(f'{df["Año"].max()}-12-01')),
            line_dash='dash', line_color='gray',
            annotation_text='Inicio proyeccion'
        )
        fig_proy.update_layout(height=400)
        st.plotly_chart(fig_proy, use_container_width=True)

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


# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.85rem; padding:1rem">
    🌿 <b>Ingenio Riopaila Castilla — Sistema de Analitica de Costos</b><br>
    Analitica Predictiva Aplicada a los Negocios | INTEP Roldanillo Valle | 2024<br>
    Desarrollado con Python · Streamlit · scikit-learn · statsmodels · Plotly
</div>
""", unsafe_allow_html=True)
