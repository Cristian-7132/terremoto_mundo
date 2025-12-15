#NOTA SOBRE USO DE LAMBDA:
#En este codigo, use funciones lambda, las funciones lambda se utilizan para aplicar transformaciones rápidas y concisas a columnas de DataFrames,
#por ejemplo, para formatear fechas, convertir valores o crear nuevas columnas de manera eficiente y legible.
#Ej.: df["col"].apply(lambda x: funcion(x))

import streamlit as st
from quakefeeds import QuakeFeed
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone
import numpy as np

# -------------------------
# Constantes
# -------------------------
PR_BBOX = {
    "lat_min": 17.6,
    "lat_max": 19.0,
    "lon_min": -67.9,
    "lon_max": -65.0,
}

SEVERITY_OPTIONS = {
    "todos": "all",
    "significativo": "significant",
    "4.5": "4.5",
    "2.5": "2.5",
    "1.0": "1.0",
}

PERIOD_OPTIONS = {
    "mes": "month",
    "semana": "week",
    "día": "day",
}
PR_SIZE_FACTOR = 0.02
WORLD_SIZE_FACTOR = 0.02

MESES_ES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]

def formato_fecha_dia(dt):
    if pd.isna(dt) or dt is None:
        return ""
    try:
        ts = pd.to_datetime(dt)
        return f"{ts.day} de {MESES_ES[ts.month - 1]} de {ts.year}"
    except Exception:
        return ""

def clasificacion_mag(mag):
    try:
        m = float(mag)
    except Exception:
        return "desconocido"
    for bound, label in [(2, "micro"), (4, "menor"), (5, "ligero"), (6, "moderado"), (7, "fuerte"), (8, "mayor"), (10, "épico")]:
        if m < bound:
            return label
    return "legendario"

#  En este codigo, se obtiene eventos del feed de terremotos según severidad y periodo.
# - Llama a `QuakeFeed` y convierte cada evento en una fila del DataFrame
# - Parsea tiempo (ms -> datetime UTC), coordenadas, profundidad y magnitud
# - Añade columna `clasificacion` usando `clasificacion_mag` y ordena por fecha
# - Está cacheada (`ttl=300`) para reducir llamadas repetidas al feed

@st.cache_data(ttl=300)
def obtener_feed(severity_feed_key: str, period_feed_key: str):
    feed = QuakeFeed(severity_feed_key, period_feed_key)
    def parse_event(event):
        props = event.get("properties", {}) or {}
        coords = (event.get("geometry") or {}).get("coordinates", [None, None, None])
        try:
            t_ms = props.get("time")
            time_dt = pd.to_datetime(int(t_ms), unit="ms", utc=True) if t_ms is not None else None
        except Exception:
            time_dt = None
        return {
            "time": time_dt,
            "place": props.get("place", props.get("title", "")),
            "latitude": coords[1],
            "longitude": coords[0],
            "depth_km": coords[2],
            "magnitude": props.get("mag"),
        }
    df = pd.DataFrame([parse_event(e) for e in feed])
    df["clasificacion"] = df["magnitude"].apply(clasificacion_mag)
    if "time" in df.columns:
        df.sort_values(by="time", ascending=False, inplace=True, na_position="last")
        df.reset_index(drop=True, inplace=True)
    return df, feed

# -------------------------
# Sidebar
# -------------------------
st.set_page_config(layout="wide", page_title="Terremotos - Puerto Rico y Mundo")

with st.sidebar:
    st.header("Severidad")
    sev_label = st.selectbox("Selecciona severidad", options=list(SEVERITY_OPTIONS.keys()), index=0)
    severity = SEVERITY_OPTIONS[sev_label]

    st.divider()

    st.header("Periodo")
    per_label = st.selectbox("Selecciona periodo", options=list(PERIOD_OPTIONS.keys()), index=0)
    period = PERIOD_OPTIONS[per_label]

    st.divider()

    st.header("Zona Geográfica")
    zone = st.selectbox("Selecciona zona", options=["Puerto Rico", "Mundo"], index=0)

    st.divider()

    show_map = st.checkbox("Mostrar mapa", value=True)
    st.write("")

    st.divider()

    show_table_checkbox = st.checkbox("Mostrar tabla con 5 eventos", value=False)
    num_events_for_table = None
    if show_table_checkbox:
        num_events_for_table = st.slider("Cantidad de eventos", min_value=5, max_value=20, value=5)

    st.divider()
    st.markdown("Aplicación desarrollada por:")
    st.markdown("Cristian Santell")
    st.markdown("**INGE3016**")
    st.markdown("Universidad de Puerto Rico, Recinto de Humacao")

    st.divider()    

# -------------------------
# Obtener datos 
# -------------------------
# NOTA: En este bloque se obtiene el DataFrame principal de eventos sísmicos desde el feed seleccionado,
# aplicando filtros de severidad y periodo. Se maneja el error de conexión y se obtiene la hora de la petición.
with st.spinner("Obteniendo datos del feed..."):
    def _obtener_feed_seguro(sev, per):
        try:
            df, feed = obtener_feed(sev, per)
        except Exception as e:
            st.error(f"No se pudo obtener el feed: {e}")
            st.stop()
        return df, feed, getattr(feed, "time", None)

    df, feed, request_time = _obtener_feed_seguro(severity, period)

    # Filtrar por severidad numérica (ej: '4.5', '2.5', '1.0')
    if severity in ["4.5", "2.5", "1.0"]:
        try:
            thresh = float(severity)
            df = df[pd.to_numeric(df['magnitude'], errors='coerce') >= thresh].reset_index(drop=True)
        except Exception:
            pass

# -------------------------
 # Filtrar por zona
# -------------------------
if zone == "Puerto Rico":
    df_zone = df[
        (df["latitude"].notna()) &
        (df["longitude"].notna()) &
        (df["latitude"] >= PR_BBOX["lat_min"]) &
        (df["latitude"] <= PR_BBOX["lat_max"]) &
        (df["longitude"] >= PR_BBOX["lon_min"]) &
        (df["longitude"] <= PR_BBOX["lon_max"])
    ].reset_index(drop=True)
else:
    df_zone = df.copy()

# -------------------------
# Panel derecho - contenido
# -------------------------
st.title("Datos en Tiempo Real de los Terremotos en Puerto Rico y el Mundo")
st.divider()

# ---------- BLOQUE DE (FECHA / EVENTOS / PROMEDIOS) ----------
fecha_text = formato_fecha_dia(request_time)

cantidad_eventos = len(df_zone)
avg_mag = df_zone["magnitude"].dropna().astype(float).mean() if not df_zone["magnitude"].dropna().empty else np.nan
avg_depth = df_zone["depth_km"].dropna().astype(float).mean() if not df_zone["depth_km"].dropna().empty else np.nan
avg_mag_str = f"{avg_mag:.2f}" if not np.isnan(avg_mag) else "N/A"
avg_depth_str = f"{avg_depth:.2f} km" if not np.isnan(avg_depth) else "N/A"
st.markdown(
        f"<div style='text-align:center;font-size:14px;color:#DDDDDD;padding:6px;'>"
        f"<strong>Fecha de petición</strong><br>{fecha_text}<br>"
        f"<strong>Cantidad de eventos</strong><br>{cantidad_eventos}<br>"
        f"<strong>Promedios</strong><br>Magnitud: {avg_mag_str} — Profundidad: {avg_depth_str}</div>",
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------------
# TABLA DE EVENTOS
# [Fecha (día/mes/año), Localización, Magnitud y Clasificación]
# -------------------------------------------------------------------------
df_display = df_zone.copy()
if "time" in df_display.columns:
    df_display["time"] = df_display["time"].apply(lambda x: formato_fecha_dia(x) if pd.notnull(x) else "")
    cols = [c for c in ["time", "place", "magnitude", "clasificacion"] if c in df_display.columns]
    if show_table_checkbox and num_events_for_table is not None:
        st.divider()
        table_to_show = df_display[cols].head(num_events_for_table)
        if "magnitude" in table_to_show.columns:
            table_to_show["magnitude"] = table_to_show["magnitude"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
        st.table(table_to_show.rename({
            "time": "Fecha",
            "place": "Localización",
            "magnitude": "Magnitud",
            "clasificacion": "Clasificación"
        }, axis=1))

# -------------------------
# Histogramas y Mapa
# -------------------------
# Preparar datos para gráficos
map_df = df_zone.assign(
    time_str=lambda d: d["time"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(x) else ""),
    time_date=lambda d: d["time"].apply(lambda x: formato_fecha_dia(x) if pd.notnull(x) else ""),
    magnitude=lambda d: pd.to_numeric(d["magnitude"], errors="coerce"),
    depth_km=lambda d: pd.to_numeric(d.get("depth_km", d.get("profundidad", None)), errors="coerce"),
)
size_base = map_df["magnitude"].fillna(0.0).clip(lower=0.0)

# -------------------------
# Puntos de los Terremotos
# -------------------------
base_factor = 1.0
size_factor = base_factor * (float(WORLD_SIZE_FACTOR) if zone == "Mundo" else float(PR_SIZE_FACTOR))
default_zoom = 0 if zone == "Mundo" else 7
size_max, sizemin_value = 3.5, 0.5

# -------------------------
# Fórmula de tamaño: escala por magnitud con pequeño offset (Para que todos los puntos sean visibles)
# ------------------------
map_df["size_plot"] = ((size_base * size_factor) + 0.15).clip(lower=0.05, upper=size_max)


if zone == "Puerto Rico":
    center_lat = (PR_BBOX["lat_min"] + PR_BBOX["lat_max"]) / 2
    center_lon = (PR_BBOX["lon_min"] + PR_BBOX["lon_max"]) / 2
else:
    center_lat = 0
    center_lon = -30

# -------------------------
# Altura común para gráficos y función para crear el histograma
# -------------------------
common_height = 520
def _hist(x, title, color, range_x=None):
    x_title = "Magnitudes" if x == "magnitude" else ("Profundidades" if x == "depth_km" else title.split()[1].capitalize())
    return px.histogram(map_df, x=x, nbins=30, color_discrete_sequence=[color], title=title, range_x=range_x).update_layout(
        yaxis_title="Conteo", xaxis_title=x_title, height=common_height,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )


# Escala fija de magnitudes según la escala de Richter definida:
# micro: <2, menor: 2-3.9, ligero: 4-4.9, moderado: 5-5.9, fuerte: 6-6.9, mayor: 7-7.9, épico: 8-9.9, legendario: 10+
mag_min, mag_max = 0.0, 10.0

fig_mag = _hist("magnitude", "Histograma de Magnitudes", "crimson", range_x=(mag_min, mag_max))
fig_depth = _hist("depth_km", "Histograma de Profundidades", "darkred")

if show_map:
    fig_map = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color="magnitude",
        size="size_plot",
        color_continuous_scale="Turbo",
        range_color=(mag_min, mag_max),
        size_max=size_max,
        zoom=default_zoom,
        center=dict(lat=center_lat, lon=center_lon),
        custom_data=["place", "magnitude", "longitude", "latitude", "time_date", "depth_km"],
        height=common_height,
    )
    hover_template = (
        "<b>%{customdata[0]}</b><br>"
        "Magnitud: %{customdata[1]:.2f}<br>"
        "Latitud: %{customdata[3]:.3f}<br>"
        "Longitud: %{customdata[2]:.3f}<br>"
        "Fecha: %{customdata[4]}<br>"
        "Profundidad: %{customdata[5]:.2f} km<extra></extra>"
    )
    fig_map.update_traces(hovertemplate=hover_template, marker=dict(sizemode="diameter", sizemin=sizemin_value))
    fig_map.update_layout(mapbox_style="carto-darkmatter", margin={"r":0,"t":30,"l":0,"b":0},
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         coloraxis_colorbar=dict(title="Magnitudes"), title=dict(text="Mapa de Terremotos", x=0.5))
    a, b, c = st.columns([1, 1, 1.6], gap="large")
    with a: st.plotly_chart(fig_mag, use_container_width=True)
    with b: st.plotly_chart(fig_depth, use_container_width=True)
    with c: st.plotly_chart(fig_map, use_container_width=True)
else:
    a, b = st.columns([1, 1], gap="large")
    with a: st.plotly_chart(fig_mag, use_container_width=True)
    with b: st.plotly_chart(fig_depth, use_container_width=True)

