from pathlib import Path  # Maneja rutas de forma portable.
from html import escape  # Escapa nombres de equipos antes de insertarlos en HTML.
import itertools  # Genera combinaciones de equipos para las tablas de grupo.
import json  # Lee los grupos por defecto y serializa el estado de grupos para la cache.
import time as time_module  # Mide tiempo transcurrido y ETA de las simulaciones.

import joblib  # Carga el scaler y los metadatos guardados en el notebook 02.
import matplotlib.pyplot as plt  # Dibuja el grafico de barras de campeones probables.
import numpy as np  # Operaciones numericas (normalizar probabilidades, clip de goles, etc).
import pandas as pd  # Manipula tablas de datos.
import streamlit as st  # Framework del dashboard.
from tensorflow import keras  # Carga el modelo de Deep Learning entrenado (MLP multi-salida).


st.set_page_config(page_title="Simulador Mundial 2026", page_icon="🏆", layout="wide")  # Configura pestana y layout ancho.

ROOT = Path(__file__).resolve().parent  # Carpeta raiz del proyecto (donde vive este archivo).
DATA_DIR = ROOT / "artifacts" / "data"  # Carpeta con los CSV/JSON generados por el notebook 01.
MODEL_DIR = ROOT / "artifacts" / "models"  # Carpeta con los modelos entrenados en el notebook 02.

# ---------------------------------------------------------------------------
# Estilos globales (CSS incrustado para tipografia, hero, chips y tarjetas)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Sora', sans-serif !important; letter-spacing: -0.01em; }

    .hero {
        background: linear-gradient(135deg, #064e3b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.6rem 2rem 1.4rem;
        color: #f8fafc;
        margin-bottom: 1rem;
    }
    .hero-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #6ee7b7;
        margin-bottom: 0.3rem;
    }
    .hero h1 {
        color: #ffffff !important;
        font-size: 1.9rem;
        margin: 0 0 0.35rem;
    }
    .hero p { color: #cbd5e1; margin: 0 0 0.8rem; font-size: 0.92rem; }
    .hero-chips { display: flex; flex-wrap: wrap; gap: 0.45rem; }
    .chip {
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 999px;
        padding: 0.28rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 600;
        color: #e2e8f0;
    }

    .team-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        margin: 0.16rem;
        font-weight: 600;
        font-size: 0.84rem;
    }
    .team-chip.direct { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .team-chip.third { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }

    .champ-card {
        background: linear-gradient(135deg, #065f46 0%, #064e3b 100%);
        color: #ffffff;
        border-radius: 14px;
        padding: 1.1rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.1rem;
        margin: 0.6rem 0 1rem;
    }
    .champ-trophy { font-size: 2.8rem; line-height: 1; }
    .champ-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #a7f3d0;
    }
    .champ-name { font-family: 'Sora', sans-serif; font-size: 1.65rem; font-weight: 800; }

    .legend-row { display: flex; flex-wrap: wrap; gap: 0.8rem; margin: 0.2rem 0 0.6rem; font-size: 0.8rem; color: #475569; }
    .legend-dot { display: inline-block; width: 10px; height: 10px; border-radius: 999px; margin-right: 0.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)  # Inyecta el CSS una sola vez al iniciar la app.

# ---------------------------------------------------------------------------
# Banderas (fallback: balon) para que la interfaz sea mas visual
# ---------------------------------------------------------------------------
FLAGS = {  # Diccionario nombre de seleccion -> emoji de bandera para mostrar en la interfaz.
    "Argentina": "🇦🇷", "Brasil": "🇧🇷", "Brazil": "🇧🇷", "España": "🇪🇸", "Spain": "🇪🇸",
    "Francia": "🇫🇷", "France": "🇫🇷", "Inglaterra": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Alemania": "🇩🇪", "Germany": "🇩🇪", "Portugal": "🇵🇹", "Países Bajos": "🇳🇱",
    "Paises Bajos": "🇳🇱", "Netherlands": "🇳🇱", "Holanda": "🇳🇱", "Italia": "🇮🇹", "Italy": "🇮🇹",
    "Bélgica": "🇧🇪", "Belgica": "🇧🇪", "Belgium": "🇧🇪", "Croacia": "🇭🇷", "Croatia": "🇭🇷",
    "Uruguay": "🇺🇾", "Colombia": "🇨🇴", "México": "🇲🇽", "Mexico": "🇲🇽",
    "Estados Unidos": "🇺🇸", "United States": "🇺🇸", "USA": "🇺🇸", "Canadá": "🇨🇦", "Canada": "🇨🇦",
    "Marruecos": "🇲🇦", "Morocco": "🇲🇦", "Senegal": "🇸🇳", "Japón": "🇯🇵", "Japon": "🇯🇵", "Japan": "🇯🇵",
    "Corea del Sur": "🇰🇷", "South Korea": "🇰🇷", "Korea Republic": "🇰🇷", "Australia": "🇦🇺",
    "Suiza": "🇨🇭", "Switzerland": "🇨🇭", "Dinamarca": "🇩🇰", "Denmark": "🇩🇰",
    "Polonia": "🇵🇱", "Poland": "🇵🇱", "Ecuador": "🇪🇨", "Perú": "🇵🇪", "Peru": "🇵🇪",
    "Chile": "🇨🇱", "Paraguay": "🇵🇾", "Venezuela": "🇻🇪", "Bolivia": "🇧🇴",
    "Costa Rica": "🇨🇷", "Panamá": "🇵🇦", "Panama": "🇵🇦", "Honduras": "🇭🇳", "Jamaica": "🇯🇲",
    "Ghana": "🇬🇭", "Nigeria": "🇳🇬", "Camerún": "🇨🇲", "Camerun": "🇨🇲", "Cameroon": "🇨🇲",
    "Egipto": "🇪🇬", "Egypt": "🇪🇬", "Túnez": "🇹🇳", "Tunez": "🇹🇳", "Tunisia": "🇹🇳",
    "Argelia": "🇩🇿", "Algeria": "🇩🇿", "Costa de Marfil": "🇨🇮", "Ivory Coast": "🇨🇮", "Côte d'Ivoire": "🇨🇮",
    "Malí": "🇲🇱", "Mali": "🇲🇱", "Sudáfrica": "🇿🇦", "South Africa": "🇿🇦",
    "Arabia Saudita": "🇸🇦", "Saudi Arabia": "🇸🇦", "Irán": "🇮🇷", "Iran": "🇮🇷", "IR Iran": "🇮🇷",
    "Irak": "🇮🇶", "Iraq": "🇮🇶", "Catar": "🇶🇦", "Qatar": "🇶🇦", "Jordania": "🇯🇴", "Jordan": "🇯🇴",
    "Uzbekistán": "🇺🇿", "Uzbekistan": "🇺🇿", "China": "🇨🇳", "Nueva Zelanda": "🇳🇿", "New Zealand": "🇳🇿",
    "Escocia": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Gales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
    "Austria": "🇦🇹", "Serbia": "🇷🇸", "Ucrania": "🇺🇦", "Ukraine": "🇺🇦",
    "Turquía": "🇹🇷", "Turquia": "🇹🇷", "Turkey": "🇹🇷", "Türkiye": "🇹🇷", "Suecia": "🇸🇪", "Sweden": "🇸🇪",
    "Noruega": "🇳🇴", "Norway": "🇳🇴", "Grecia": "🇬🇷", "Greece": "🇬🇷",
    "República Checa": "🇨🇿", "Republica Checa": "🇨🇿", "Czech Republic": "🇨🇿", "Chequia": "🇨🇿", "Czechia": "🇨🇿",
    "Hungría": "🇭🇺", "Hungria": "🇭🇺", "Hungary": "🇭🇺", "Rumania": "🇷🇴", "Romania": "🇷🇴",
    "Eslovaquia": "🇸🇰", "Slovakia": "🇸🇰", "Eslovenia": "🇸🇮", "Slovenia": "🇸🇮",
    "Albania": "🇦🇱", "Georgia": "🇬🇪", "Finlandia": "🇫🇮", "Finland": "🇫🇮",
    "Irlanda": "🇮🇪", "Ireland": "🇮🇪", "Rusia": "🇷🇺", "Russia": "🇷🇺",
    "Curaçao": "🇨🇼", "Curacao": "🇨🇼", "Congo DR": "🇨🇩", "Cape Verde": "🇨🇻", "Haiti": "🇭🇹",
}


def flag(team):  # Devuelve la bandera de una seleccion o un balon si no esta en el diccionario.
    return FLAGS.get(str(team), "⚽")  # Evita romper la interfaz con selecciones sin bandera mapeada.


# ---------------------------------------------------------------------------
# Carga de datos y del modelo de Deep Learning entrenado en el notebook 02
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_and_bundle():  # Carga una sola vez el modelo Keras y su bundle de preprocesamiento.
    bundle = joblib.load(MODEL_DIR / "model_bundle.pkl")  # Scaler, columnas de features y metricas del entrenamiento.
    model = keras.models.load_model(MODEL_DIR / "dashboard_model.keras")  # Red neuronal (MLP) elegida para el simulador.
    return bundle, model  # Devuelve ambos objetos para reusarlos en toda la app.


@st.cache_data
def load_tables():  # Carga los datos tabulares que no cambian durante la sesion.
    groups = json.loads((DATA_DIR / "groups_2026.json").read_text(encoding="utf-8"))  # Grupos oficiales por defecto.
    team_state = pd.read_csv(DATA_DIR / "team_state_2026.csv")  # Ranking, forma y atributos FIFA vigentes por seleccion.
    team_history = pd.read_csv(DATA_DIR / "team_match_history.csv", parse_dates=["date"])  # Historial de partidos (para H2H).
    champions = pd.read_csv(DATA_DIR / "champion_history.csv")  # Cuantos mundiales gano cada seleccion historicamente.
    return groups, team_state, team_history, champions  # Devuelve las cuatro tablas.


bundle, model = load_model_and_bundle()  # Trae el modelo entrenado y su bundle.
scaler = bundle["scaler"]  # StandardScaler ajustado en el notebook 02 (mismas medias/varianzas que en entrenamiento).
feature_cols = bundle["feature_cols"]  # Orden exacto de columnas que espera el modelo.
groups_default, team_state, team_history, champion_history = load_tables()  # Trae los datos ya procesados.
all_teams = sorted(team_state["team"].tolist())  # Lista de las 48 selecciones del Mundial 2026.
state_by_team = team_state.set_index("team").to_dict(orient="index")  # Acceso rapido al estado de cada seleccion por nombre.


# ---------------------------------------------------------------------------
# Logica de prediccion: usa el MLP entrenado (no un modelo alterno)
# ---------------------------------------------------------------------------
def h2h_stats(team_a, team_b):  # Calcula el historial de enfrentamientos directos de A visto desde A.
    pair = " vs ".join(sorted([team_a, team_b]))  # Llave simetrica del cruce (mismo orden sin importar quien es A o B).
    recent = team_history[(team_history["pair"] == pair) & (team_history["team"] == team_a)].sort_values("date").tail(10)  # Ultimos 10 duelos directos.
    if recent.empty:  # Si nunca se enfrentaron (o no hay historial suficiente).
        return {"win": 0.0, "draw": 0.0, "loss": 0.0, "goal_diff": 0.0}  # Devuelve un historial neutro.
    return {
        "win": float(recent["win"].sum()),  # Victorias de A sobre B en esos duelos.
        "draw": float(recent["draw"].sum()),  # Empates entre A y B.
        "loss": float(recent["loss"].sum()),  # Derrotas de A ante B.
        "goal_diff": float(recent["goal_diff"].sum()),  # Diferencia de goles acumulada de A en esos duelos.
    }


def match_feature_values(team_a, team_b):  # Arma el diccionario de features crudas para un cruce A vs B.
    a = state_by_team[team_a]  # Estado actual (ranking, forma, jugadores) de A.
    b = state_by_team[team_b]  # Estado actual de B.
    h2h_a = h2h_stats(team_a, team_b)  # Historial directo de A contra B.
    h2h_b = h2h_stats(team_b, team_a)  # Historial directo de B contra A.
    values = {  # Features base identicas a las usadas en el notebook de entrenamiento.
        "home_rank": a["rank"],
        "away_rank": b["rank"],
        "home_ranking_points": a["ranking_points"],
        "away_ranking_points": b["ranking_points"],
        "home_goals_for_last10": a["goals_for_last10"],
        "away_goals_for_last10": b["goals_for_last10"],
        "home_goals_against_last10": a["goals_against_last10"],
        "away_goals_against_last10": b["goals_against_last10"],
        "home_points_last10": a["points_last10"],
        "away_points_last10": b["points_last10"],
        "home_goal_diff_last10": a["goal_diff_last10"],
        "away_goal_diff_last10": b["goal_diff_last10"],
        "home_h2h_win_last10": h2h_a["win"],
        "away_h2h_win_last10": h2h_b["win"],
        "home_h2h_draw_last10": h2h_a["draw"],
        "away_h2h_draw_last10": h2h_b["draw"],
        "home_h2h_loss_last10": h2h_a["loss"],
        "away_h2h_loss_last10": h2h_b["loss"],
        "home_h2h_goal_diff_last10": h2h_a["goal_diff"],
        "away_h2h_goal_diff_last10": h2h_b["goal_diff"],
        "home_player_overall": a["overall"],
        "away_player_overall": b["overall"],
        "home_player_pace": a["pace"],
        "away_player_pace": b["pace"],
        "home_player_shooting": a["shooting"],
        "away_player_shooting": b["shooting"],
        "home_player_defending": a["defending"],
        "away_player_defending": b["defending"],
        "home_player_physical": a["physical"],
        "away_player_physical": b["physical"],
        "neutral": 1.0,  # Los cruces del simulador se tratan como cancha neutral.
    }
    for name, left, right in [  # Agrega las mismas variables de diferencia (A - B) usadas en el entrenamiento.
        ("rank", "home_rank", "away_rank"),
        ("ranking_points", "home_ranking_points", "away_ranking_points"),
        ("goals_for_last10", "home_goals_for_last10", "away_goals_for_last10"),
        ("goals_against_last10", "home_goals_against_last10", "away_goals_against_last10"),
        ("points_last10", "home_points_last10", "away_points_last10"),
        ("goal_diff_last10", "home_goal_diff_last10", "away_goal_diff_last10"),
        ("player_overall", "home_player_overall", "away_player_overall"),
        ("player_pace", "home_player_pace", "away_player_pace"),
        ("player_shooting", "home_player_shooting", "away_player_shooting"),
        ("player_defending", "home_player_defending", "away_player_defending"),
        ("player_physical", "home_player_physical", "away_player_physical"),
    ]:
        values[f"diff_{name}"] = values[left] - values[right]  # Calcula la ventaja relativa de A sobre B.
    return values  # Devuelve el diccionario completo de features para este cruce.


def build_feature_frame(pairs):  # Construye la tabla de features para una lista de cruces (team_a, team_b).
    rows = [match_feature_values(team_a, team_b) for team_a, team_b in pairs]  # Calcula una fila por cruce.
    frame = pd.DataFrame(rows)  # Convierte la lista de diccionarios en tabla.
    for col in feature_cols:  # Recorre las columnas que el modelo espera en ese orden.
        if col not in frame.columns:  # Si faltara alguna (no deberia ocurrir con los datos actuales).
            frame[col] = 0.0  # Rellena por seguridad para no romper el escalado.
    return frame[feature_cols].astype(float)  # Reordena columnas exactamente como en el entrenamiento.


@st.cache_resource(show_spinner="Calculando predicciones del modelo para todos los cruces posibles...")
def build_prediction_cache(_model, _scaler, feature_cols_tuple, all_teams_tuple):  # Precalcula todas las predicciones en un solo batch.
    teams = list(all_teams_tuple)  # Convierte la tupla (hashable) de vuelta a lista.
    cols = list(feature_cols_tuple)  # Orden de columnas que define la clave de cache (debe coincidir con el del scaler).
    pairs = [(team_a, team_b) for team_a in teams for team_b in teams if team_a != team_b]  # Todos los cruces ordenados posibles.
    frame = build_feature_frame(pairs)[cols]  # Features de todos los cruces a la vez, reordenadas segun feature_cols_tuple.
    scaled = _scaler.transform(frame.values)  # Escala con el mismo scaler ajustado en el entrenamiento.
    preds_batch = _model.predict(scaled, batch_size=512, verbose=0)  # Una sola llamada batched al MLP (rapido incluso en CPU).
    probs_batch, goals_batch = preds_batch["resultado"], preds_batch["goles"]  # El modelo tiene salidas nombradas; predict() devuelve un diccionario.
    cache = {}  # Diccionario final cruce -> prediccion.
    for (team_a, team_b), probs, goals in zip(pairs, probs_batch, goals_batch):  # Recorre cada cruce con su prediccion.
        probs_norm = probs / probs.sum()  # Renormaliza por seguridad numerica.
        cache[(team_a, team_b)] = {  # Guarda el resultado con el mismo formato que usa el resto de la app.
            "team_a": team_a,
            "team_b": team_b,
            "p_a": float(probs_norm[0]),  # Probabilidad de victoria de A.
            "p_draw": float(probs_norm[1]),  # Probabilidad de empate.
            "p_b": float(probs_norm[2]),  # Probabilidad de victoria de B.
            "g_a": float(np.clip(goals[0], 0, 6)),  # Marcador esperado de A (acotado a un rango razonable).
            "g_b": float(np.clip(goals[1], 0, 6)),  # Marcador esperado de B.
        }
    return cache  # Devuelve la cache completa (se calcula una sola vez por sesion).


PREDICTION_CACHE = build_prediction_cache(model, scaler, tuple(feature_cols), tuple(all_teams))  # Cache lista antes de renderizar la interfaz.


def predict_match(team_a, team_b):  # Devuelve la prediccion de un cruce usando el modelo de Deep Learning.
    result = PREDICTION_CACHE.get((team_a, team_b))  # Busca el cruce ya precalculado.
    if result is None:  # Caso limite: un equipo fuera de la lista original de 48 selecciones.
        frame = build_feature_frame([(team_a, team_b)])  # Calcula la fila de features al vuelo.
        scaled = scaler.transform(frame.values)  # Escala con el mismo scaler del entrenamiento.
        preds = model.predict(scaled, verbose=0)  # Predice con el MLP entrenado (dict con ambas salidas).
        probs, goals = preds["resultado"], preds["goles"]  # Extrae probabilidades y marcador del diccionario.
        probs_norm = probs[0] / probs[0].sum()  # Normaliza probabilidades.
        result = {
            "team_a": team_a,
            "team_b": team_b,
            "p_a": float(probs_norm[0]),
            "p_draw": float(probs_norm[1]),
            "p_b": float(probs_norm[2]),
            "g_a": float(np.clip(goals[0][0], 0, 6)),
            "g_b": float(np.clip(goals[0][1], 0, 6)),
        }
    return result  # Devuelve el diccionario con las 3 probabilidades y el marcador esperado.


def group_table(teams):  # Calcula la tabla proyectada de un grupo de 4 selecciones.
    rows = {team: {"seleccion": team, "pts": 0.0, "gf": 0.0, "gc": 0.0} for team in teams}  # Acumuladores por seleccion.
    for team_a, team_b in itertools.combinations(teams, 2):  # Recorre los 6 cruces posibles del grupo.
        pred = predict_match(team_a, team_b)  # Prediccion del modelo para ese cruce.
        rows[team_a]["pts"] += 3 * pred["p_a"] + pred["p_draw"]  # Puntos esperados de A (valor esperado, no resultado fijo).
        rows[team_b]["pts"] += 3 * pred["p_b"] + pred["p_draw"]  # Puntos esperados de B.
        rows[team_a]["gf"] += pred["g_a"]  # Goles a favor esperados de A.
        rows[team_a]["gc"] += pred["g_b"]  # Goles en contra esperados de A.
        rows[team_b]["gf"] += pred["g_b"]  # Goles a favor esperados de B.
        rows[team_b]["gc"] += pred["g_a"]  # Goles en contra esperados de B.
    table = pd.DataFrame(rows.values())  # Convierte los acumuladores en tabla.
    table["dg"] = table["gf"] - table["gc"]  # Diferencia de goles esperada.
    table = table.sort_values(["pts", "dg", "gf"], ascending=False).reset_index(drop=True)  # Ordena como una tabla real de posiciones.
    table["pos"] = np.arange(1, len(table) + 1)  # Asigna posicion 1 a 4.
    table["clasificacion_%"] = np.select([table["pos"] <= 2, table["pos"] == 3], [92.0, 55.0], default=12.0)  # Probabilidad heuristica de avanzar segun la posicion.
    return table.round({"pts": 2, "gf": 2, "gc": 2, "dg": 2, "clasificacion_%": 1})  # Redondea para mostrar en pantalla.


def qualified_teams(groups):  # Calcula los 32 clasificados a la fase eliminatoria.
    tables = {group: group_table(teams) for group, teams in groups.items()}  # Tabla proyectada de cada uno de los 12 grupos.
    direct = []  # Clasificados directos (1o y 2o de cada grupo).
    thirds = []  # Terceros lugares (compiten entre si por 8 cupos).
    for group, table in tables.items():  # Recorre cada grupo.
        top_two = table.head(2).copy()  # Primero y segundo lugar.
        direct.extend(top_two["seleccion"].tolist())  # Los agrega como clasificados directos.
        third = table.iloc[2].copy()  # Tercer lugar del grupo.
        third["grupo"] = group  # Registra de que grupo viene.
        thirds.append(third)  # Lo guarda para comparar contra los demas terceros.
    best_thirds = pd.DataFrame(thirds).sort_values(["pts", "dg", "gf"], ascending=False).head(8)  # Los 8 mejores terceros clasifican.
    return (direct + best_thirds["seleccion"].tolist())[:32], tables  # Devuelve los 32 clasificados y las tablas por grupo.


def build_bracket(teams, forced_team=None):  # Simula el cuadro de eliminacion directa completo.
    current = list(teams)  # Lista de selecciones que siguen en competencia.
    matches = []  # Guarda cada partido simulado.
    round_names = ["Ronda de 32", "Octavos", "Cuartos", "Semifinal", "Final"]  # Nombres de las 5 rondas para 32 equipos.
    for round_name in round_names:  # Recorre ronda por ronda.
        winners = []  # Ganadores de esta ronda.
        for idx in range(0, len(current), 2):  # Empareja de dos en dos.
            team_a, team_b = current[idx], current[idx + 1]  # Cruce de la ronda.
            pred = predict_match(team_a, team_b)  # Prediccion del modelo para el cruce.
            if forced_team in [team_a, team_b]:  # Si el usuario forzo un ganador y aparece en este cruce.
                winner = forced_team  # Ese equipo avanza sin importar la probabilidad.
            else:  # Caso normal: gana quien tenga mayor probabilidad de victoria.
                winner = team_a if pred["p_a"] >= pred["p_b"] else team_b  # Compara victoria A vs victoria B.
            winners.append(winner)  # Registra el ganador de este cruce.
            matches.append({**pred, "ronda": round_name, "ganador": winner})  # Guarda el partido con su resultado.
        current = winners  # Pasa a la siguiente ronda con los ganadores.
        if len(current) == 1:  # Si ya queda un solo equipo.
            break  # Termina la simulacion (ya hay campeon del bracket).
    return pd.DataFrame(matches), current[0]  # Devuelve todos los partidos y el campeon final.


def seed_bracket_by_top_champions(qualified, top10_teams):  # Ordena el bracket para que el top 10 quede repartido y visible.
    top_seeded = [team for team in top10_teams if team in qualified]  # Favoritos del top 10 que si clasificaron.
    remaining = [team for team in qualified if team not in top_seeded]  # El resto de clasificados.
    size = len(qualified)  # Tamano del cuadro (32).
    slots = [None] * size  # Posiciones vacias del cuadro.
    preferred_slots = [  # Posiciones separadas entre si para que los favoritos no se crucen demasiado pronto.
        0,
        size - 1,
        size // 2,
        size // 2 - 1,
        size // 4,
        3 * size // 4 - 1,
        size // 4 - 1,
        3 * size // 4,
        size // 8,
        7 * size // 8 - 1,
    ]
    for team, slot in zip(top_seeded, preferred_slots):  # Coloca cada favorito en una posicion separada.
        slots[slot] = team  # Asigna el equipo a esa posicion del cuadro.
    fill_teams = iter(remaining)  # Iterador con el resto de equipos.
    return [team if team is not None else next(fill_teams) for team in slots]  # Completa el resto de posiciones.


def champion_simulation_table(groups, simulations, progress_bar=None, status_box=None):  # Simula el torneo completo muchas veces.
    counts = {}  # Cuenta cuantas veces gano cada seleccion.
    qualified, _ = qualified_teams(groups)  # Los 32 clasificados segun los grupos actuales.
    rng = np.random.default_rng(42)  # Generador aleatorio con semilla fija (resultados reproducibles).
    started_at = time_module.perf_counter()  # Marca de tiempo inicial para el ETA.
    update_every = max(1, simulations // 200)  # Actualiza la barra de progreso cada cierto numero de simulaciones.
    for sim_idx in range(simulations):  # Repite el torneo la cantidad de veces pedida.
        current = list(rng.permutation(qualified))  # Empareja a los 32 clasificados en orden aleatorio.
        while len(current) > 1:  # Mientras quede mas de un equipo en pie.
            next_round = []  # Ganadores de esta ronda de la simulacion.
            for idx in range(0, len(current), 2):  # Recorre los cruces de la ronda.
                pred = predict_match(current[idx], current[idx + 1])  # Prediccion del modelo para el cruce.
                probs = np.array([pred["p_a"] + 0.5 * pred["p_draw"], pred["p_b"] + 0.5 * pred["p_draw"]])  # Reparte el empate entre ambos para poder sortear un unico ganador.
                probs = probs / probs.sum()  # Normaliza para que sumen 1.
                next_round.append(rng.choice([current[idx], current[idx + 1]], p=probs))  # Sortea el ganador segun esas probabilidades.
            current = next_round  # Avanza a la siguiente ronda.
        counts[current[0]] = counts.get(current[0], 0) + 1  # Suma un titulo al campeon de esta simulacion.
        if progress_bar is not None and ((sim_idx + 1) % update_every == 0 or sim_idx + 1 == simulations):  # Actualiza la UI de vez en cuando (no en cada simulacion).
            progress = (sim_idx + 1) / simulations  # Porcentaje completado.
            elapsed = time_module.perf_counter() - started_at  # Tiempo transcurrido.
            eta_seconds = (elapsed / progress) - elapsed if progress > 0 else 0  # Estimacion del tiempo restante.
            progress_bar.progress(progress)  # Actualiza la barra visual.
            if status_box is not None:  # Si hay un cuadro de texto para el estado.
                status_box.caption(
                    f"Simulaciones completadas: {sim_idx + 1:,} / {simulations:,} | "
                    f"Transcurrido: {format_seconds(elapsed)} | ETA: {format_seconds(eta_seconds)}"
                )  # Muestra progreso, tiempo transcurrido y tiempo estimado restante.
    table = pd.DataFrame({"seleccion": list(counts), "veces_campeon": list(counts.values())})  # Tabla con el conteo de titulos.
    table["simulaciones"] = simulations  # Registra el total de simulaciones usadas.
    table["winrate_%"] = 100 * table["veces_campeon"] / simulations  # Calcula el porcentaje de veces campeon.
    table = table.merge(champion_history, on="seleccion", how="left")  # Agrega el historico real de titulos como contexto.
    table["mundiales_ganados"] = table["mundiales_ganados"].fillna(0).astype(int)  # Selecciones sin titulos historicos quedan en 0.
    return table.sort_values(["veces_campeon", "winrate_%"], ascending=False).reset_index(drop=True)  # Ordena de mayor a menor probabilidad.


def cached_champion_table(groups_json, simulations, progress_bar=None, status_box=None):  # Reutiliza la simulacion si los grupos y el numero de simulaciones no cambiaron.
    cache_key = (groups_json, simulations)  # Clave que identifica un mismo escenario de simulacion.
    if st.session_state.get("champion_cache_key") == cache_key:  # Si ya se simulo exactamente este escenario antes.
        return st.session_state["champion_cache_value"]  # Reutiliza el resultado sin recalcular ni tocar la barra de progreso.
    groups = json.loads(groups_json)  # Reconstruye el diccionario de grupos desde el JSON.
    result = champion_simulation_table(groups, simulations, progress_bar, status_box)  # Corre la simulacion completa (con progreso visible).
    st.session_state["champion_cache_key"] = cache_key  # Guarda la clave de este escenario.
    st.session_state["champion_cache_value"] = result  # Guarda el resultado para reutilizarlo mientras no cambie el escenario.
    return result  # Devuelve el resultado (recien calculado o reutilizado).


def format_seconds(seconds):  # Formatea segundos como texto legible (h/m/s).
    seconds = max(0, int(seconds))  # Evita valores negativos por redondeo.
    hours, remainder = divmod(seconds, 3600)  # Separa horas.
    minutes, seconds = divmod(remainder, 60)  # Separa minutos y segundos restantes.
    if hours:  # Si hay al menos una hora.
        return f"{hours}h {minutes:02d}m {seconds:02d}s"  # Formato con horas.
    if minutes:  # Si hay al menos un minuto.
        return f"{minutes}m {seconds:02d}s"  # Formato con minutos.
    return f"{seconds}s"  # Formato solo con segundos.


# ---------------------------------------------------------------------------
# Componentes visuales
# ---------------------------------------------------------------------------
def show_group_table(table):  # Muestra la tabla de un grupo con formato visual (colores, banderas, barra de progreso).
    df = table.copy()  # Copia para no modificar la tabla original.
    df["estado"] = np.select(
        [df["pos"] <= 2, df["pos"] == 3],
        ["🟢 Clasifica directo", "🟡 Puede ir como mejor tercero"],
        default="🔴 Eliminado",
    )  # Etiqueta el escenario de cada seleccion segun su posicion.
    df["seleccion"] = df["seleccion"].map(lambda team: f"{flag(team)} {team}")  # Agrega la bandera al nombre.
    df = df[["pos", "seleccion", "pts", "dg", "gf", "gc", "clasificacion_%", "estado"]]  # Ordena columnas para mostrar.
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "pos": st.column_config.NumberColumn("Pos", width="small"),
            "seleccion": st.column_config.TextColumn("Selección"),
            "pts": st.column_config.NumberColumn("Pts esperados", format="%.2f"),
            "dg": st.column_config.NumberColumn("Dif. goles", format="%.2f"),
            "gf": st.column_config.NumberColumn("GF", format="%.2f"),
            "gc": st.column_config.NumberColumn("GC", format="%.2f"),
            "clasificacion_%": st.column_config.ProgressColumn(
                "Prob. de clasificar", format="%.0f%%", min_value=0, max_value=100
            ),
            "estado": st.column_config.TextColumn("Escenario"),
        },
    )  # Renderiza la tabla interactiva de Streamlit.


def champion_bar_chart(top10):  # Dibuja el grafico de barras horizontal del top 10 de campeones probables.
    fig, ax = plt.subplots(figsize=(8, 5))  # Crea la figura.
    chart_data = top10.sort_values("winrate_%", ascending=True).reset_index(drop=True)  # Ordena ascendente para que el 1o quede arriba en un barh.
    n = len(chart_data)  # Cantidad de selecciones a graficar.
    colors = ["#10b981"] * n  # Color base para todas las barras.
    if n >= 1:
        colors[-1] = "#f59e0b"  # Oro para el primer lugar.
    if n >= 2:
        colors[-2] = "#94a3b8"  # Plata para el segundo lugar.
    if n >= 3:
        colors[-3] = "#b45309"  # Bronce para el tercer lugar.
    labels = list(chart_data["seleccion"])  # Etiquetas con el nombre (matplotlib no renderiza bien los emoji de bandera).
    bars = ax.barh(labels, chart_data["winrate_%"], color=colors, height=0.62)  # Dibuja las barras horizontales.
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in chart_data["winrate_%"]], padding=5, fontsize=9, fontweight="bold")  # Muestra el porcentaje al final de cada barra.
    ax.set_xlabel("Probabilidad de ser campeón (%)", fontsize=10)  # Etiqueta del eje X.
    ax.set_xlim(0, max(5, chart_data["winrate_%"].max() * 1.2))  # Deja margen para las etiquetas.
    ax.spines[["top", "right"]].set_visible(False)  # Quita bordes innecesarios.
    ax.xaxis.grid(True, color="#e2e8f0", linewidth=0.8)  # Agrega guias verticales suaves.
    ax.set_axisbelow(True)  # Las guias quedan detras de las barras.
    ax.tick_params(axis="both", labelsize=9)  # Ajusta tamano de las etiquetas de los ejes.
    fig.tight_layout()  # Ajusta margenes de la figura.
    return fig  # Devuelve la figura para mostrarla con st.pyplot.


def show_champion_table(champion_results):  # Muestra la tabla completa de campeones probables con medallas.
    df = champion_results.copy().reset_index(drop=True)  # Copia para no modificar el original.
    medals = {0: "🥇", 1: "🥈", 2: "🥉"}  # Medalla para los 3 primeros puestos.
    df.insert(0, "puesto", [medals.get(i, f"{i + 1}") for i in df.index])  # Agrega columna de puesto con medalla o numero.
    df["seleccion"] = df["seleccion"].map(lambda team: f"{flag(team)} {team}")  # Agrega bandera al nombre.
    df = df[["puesto", "seleccion", "winrate_%", "veces_campeon", "mundiales_ganados"]]  # Ordena columnas a mostrar.
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "puesto": st.column_config.TextColumn("#", width="small"),
            "seleccion": st.column_config.TextColumn("Selección"),
            "winrate_%": st.column_config.ProgressColumn(
                "Prob. de campeonar", format="%.2f%%", min_value=0, max_value=float(max(5.0, df["winrate_%"].max()))
            ),
            "veces_campeon": st.column_config.NumberColumn("Veces campeón"),
            "mundiales_ganados": st.column_config.NumberColumn("Mundiales reales 🏆"),
        },
    )  # Renderiza la tabla interactiva.


def render_bracket(bracket, top10):  # Dibuja el cuadro de eliminacion directa completo con estilo de bracket.
    round_order = ["Ronda de 32", "Octavos", "Cuartos", "Semifinal", "Final"]  # Orden de las columnas del bracket.
    top10_lookup = {  # Diccionario para resaltar rapidamente si un equipo esta en el top 10 de campeones probables.
        row["seleccion"]: {
            "rank": idx + 1,
            "winrate": row["winrate_%"],
            "wins": row["veces_campeon"],
        }
        for idx, row in top10.reset_index(drop=True).iterrows()
    }

    def team_slot(team, probability, winner):  # Genera el HTML de una selección dentro de un cruce.
        safe_team = f"{flag(team)} {escape(str(team))}"  # Nombre con bandera, escapado por seguridad.
        is_winner = team == winner  # Indica si esta seleccion gano el cruce.
        top_info = top10_lookup.get(team)  # Info del top 10 si aplica.
        classes = "team-slot winner-slot" if is_winner else "team-slot"  # Clase CSS segun si gano.
        if top_info:  # Si la seleccion esta en el top 10 de campeones probables.
            classes += " top-team"  # Le agrega el resalte dorado.
            rank_html = f"<span class='seed-badge'>Top #{top_info['rank']}</span>"  # Insignia con su puesto.
            champ_html = f"<span class='metric-chip'>Campeón: {top_info['winrate']:.1f}%</span>"  # Chip con su probabilidad de campeonar.
        else:  # Si no esta en el top 10.
            rank_html = ""  # Sin insignia.
            champ_html = ""  # Sin chip adicional.
        bar_width = max(4, probability * 100)  # Ancho minimo visible de la barra de avance.
        return (
            f'<div class="{classes}">'
            f'<div class="team-main">{rank_html}<span class="team-name">{safe_team}</span></div>'
            f'<div class="team-side">{champ_html}<span class="metric-chip advance">Avanza: {probability:.0%}</span></div>'
            f'<div class="prob-track"><span style="width: {bar_width:.1f}%"></span></div>'
            "</div>"
        )  # Devuelve el bloque HTML de la seleccion.

    st.markdown(
        """
        <style>
        .bracket-board {
            display: grid;
            grid-template-columns: repeat(5, minmax(260px, 1fr));
            gap: 0.85rem;
            overflow-x: auto;
            padding: 0.25rem 0.1rem 0.85rem;
        }
        .bracket-round { min-width: 260px; }
        .bracket-title {
            font-family: 'Sora', sans-serif;
            font-weight: 700;
            margin-bottom: 0.65rem;
            text-align: center;
            color: #0f172a;
            font-size: 0.92rem;
            padding: 0.3rem 0;
            background: #f1f5f9;
            border-radius: 999px;
        }
        .match-card {
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            padding: 0.45rem;
            margin-bottom: 0.75rem;
            background: #f8fafc;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        }
        .team-slot {
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            gap: 0.32rem;
            min-height: 60px;
            padding: 0.48rem 0.55rem 0.7rem;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            background: #ffffff;
            color: #334155;
            font-size: 0.82rem;
        }
        .team-slot + .team-slot { margin-top: 0.35rem; }
        .winner-slot {
            border-color: #16a34a;
            background: #f0fdf4;
            color: #0f172a;
        }
        .top-team { box-shadow: inset 4px 0 0 #f59e0b; }
        .team-main {
            min-width: 0;
            display: flex;
            align-items: center;
            gap: 0.36rem;
        }
        .team-name {
            min-width: 0;
            overflow-wrap: anywhere;
            font-weight: 750;
            line-height: 1.05rem;
        }
        .team-side {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.3rem;
            font-size: 0.72rem;
        }
        .seed-badge {
            flex: 0 0 auto;
            border-radius: 999px;
            padding: 0.1rem 0.38rem;
            background: #fef3c7;
            color: #92400e;
            text-align: center;
            font-size: 0.68rem;
            font-weight: 800;
        }
        .metric-chip {
            border-radius: 999px;
            padding: 0.1rem 0.42rem;
            background: #dcfce7;
            color: #166534;
            font-size: 0.68rem;
            font-weight: 800;
        }
        .metric-chip.advance {
            background: #e0f2fe;
            color: #0369a1;
        }
        .prob-track {
            position: absolute;
            left: 0.48rem;
            right: 0.48rem;
            bottom: 0.28rem;
            height: 3px;
            border-radius: 999px;
            background: #e2e8f0;
        }
        .prob-track span {
            display: block;
            height: 100%;
            border-radius: inherit;
            background: #16a34a;
        }
        .prob-triple {
            display: flex;
            justify-content: space-between;
            gap: 0.3rem;
            margin-top: 0.35rem;
            font-size: 0.68rem;
            font-weight: 700;
            color: #475569;
            background: #f1f5f9;
            border-radius: 6px;
            padding: 0.22rem 0.4rem;
        }
        .match-footer {
            display: flex;
            justify-content: space-between;
            gap: 0.4rem;
            color: #166534;
            margin-top: 0.42rem;
            font-size: 0.74rem;
            font-weight: 750;
            line-height: 1.1rem;
        }
        .round-2 .match-card { margin-top: 2.15rem; margin-bottom: 2.1rem; }
        .round-3 .match-card { margin-top: 5.3rem; margin-bottom: 5.2rem; }
        .round-4 .match-card { margin-top: 11.7rem; margin-bottom: 11.6rem; }
        .round-5 .match-card { margin-top: 24.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )  # Inyecta el CSS especifico del bracket.
    board_parts = ["<div class='bracket-board'>"]  # Acumula el HTML del bracket completo.
    for round_idx, round_name in enumerate(round_order, start=1):  # Recorre cada ronda en orden.
        round_matches = bracket[bracket["ronda"] == round_name].reset_index(drop=True)  # Partidos de esa ronda.
        board_parts.append(f"<div class='bracket-round round-{round_idx}'>")  # Abre la columna de la ronda.
        board_parts.append(f"<div class='bracket-title'>{escape(round_name)}</div>")  # Titulo de la ronda.
        for _, row in round_matches.iterrows():  # Recorre cada partido de la ronda.
            p_a_adv = row["p_a"] + 0.5 * row["p_draw"]  # Probabilidad de avanzar de A (victoria + mitad del empate).
            p_b_adv = row["p_b"] + 0.5 * row["p_draw"]  # Probabilidad de avanzar de B.
            winner = f"{flag(row['ganador'])} {escape(str(row['ganador']))}"  # Nombre del ganador con bandera.
            prob_triple = (  # Muestra las 3 probabilidades crudas del modelo tal como las pide el enunciado.
                f"<div class='prob-triple'>"
                f"<span>🅰️ Victoria A: {row['p_a']:.0%}</span>"
                f"<span>🤝 Empate: {row['p_draw']:.0%}</span>"
                f"<span>🅱️ Victoria B: {row['p_b']:.0%}</span>"
                f"</div>"
            )
            board_parts.append(
                "<div class='match-card'>"
                f"{team_slot(row['team_a'], p_a_adv, row['ganador'])}"
                f"{team_slot(row['team_b'], p_b_adv, row['ganador'])}"
                f"{prob_triple}"
                "<div class='match-footer'>"
                f"<span>✅ Avanza: {winner}</span>"
                "</div>"
                "</div>"
            )  # Arma la tarjeta completa del partido.
        board_parts.append("</div>")  # Cierra la columna de la ronda.
    board_parts.append("</div>")  # Cierra el bracket completo.
    st.markdown("".join(board_parts), unsafe_allow_html=True)  # Renderiza todo el bracket de una sola vez.


# ---------------------------------------------------------------------------
# Sidebar: controles globales
# ---------------------------------------------------------------------------
with st.sidebar:  # Barra lateral con los controles que afectan a toda la app.
    st.header("⚙️ Controles")  # Titulo de la barra lateral.
    sims = st.number_input(
        "Número de simulaciones",
        min_value=1,
        max_value=1_000_000,
        value=1_000,
        step=1_000,
        help="Más simulaciones dan resultados más estables, pero tardan más.",
    )  # Numero de torneos a simular para el ranking de campeones probables.
    forced = st.selectbox(
        "Forzar ganador en el bracket",
        ["Sin forzar"] + all_teams,
        help="La selección elegida ganará todos sus cruces de eliminación directa.",
    )  # Selector para forzar el ganador de todos los cruces donde aparezca.
    forced_team = None if forced == "Sin forzar" else forced  # None si no se eligio forzar a nadie.
    st.divider()  # Linea separadora visual.
    st.caption(
        "Los grupos se editan en la pestaña **Fase de grupos**. "
        "Cada cambio recalcula tablas, clasificados y bracket automáticamente."
    )  # Ayuda contextual para el usuario.
    st.caption(f"Modelo activo: **{bundle.get('dashboard_model', 'MLP')}** (red neuronal entrenada en el notebook 02).")  # Deja explicito que el modelo usado es el de Deep Learning.

# ---------------------------------------------------------------------------
# Hero (encabezado principal)
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-eyebrow">Copa Mundial de la FIFA · Edición 2026</div>
        <h1>🏆 Simulador Mundial 2026</h1>
        <p>Predice grupos, cruces de eliminación y campeones probables usando un modelo de Deep Learning
        entrenado con historial de partidos, ranking FIFA y atributos de jugadores.</p>
        <div class="hero-chips">
            <span class="chip">🌎 {len(all_teams)} selecciones disponibles</span>
            <span class="chip">🏟️ {len(groups_default)} grupos</span>
            <span class="chip">🎲 {int(sims):,} simulaciones</span>
            <span class="chip">🤖 MLP multi-salida (resultado + goles)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)  # Muestra el encabezado con contexto general del sistema.

# ---------------------------------------------------------------------------
# Pestañas principales: grupos, eliminatorias y campeones probables
# ---------------------------------------------------------------------------
tab_groups, tab_bracket, tab_champions = st.tabs(
    ["🏟️ Fase de grupos", "🥅 Eliminatorias", "🏆 Campeones probables"]
)  # Crea las 3 vistas obligatorias del enunciado.

with tab_groups:  # Vista de fase de grupos.
    with st.expander("✏️ Editar los grupos", expanded=False):  # Editor de grupos colapsado por defecto.
        st.caption("Cambia cualquier selección; una selección usada en un grupo no puede repetirse en otro.")  # Ayuda al usuario.
        groups_selected = {}  # Grupos definidos por el usuario en esta ejecucion.
        used = set()  # Selecciones ya asignadas a algun grupo (para no duplicar).
        group_names = list(groups_default.keys())  # Nombres de los 12 grupos (A-L).
        editor_cols = st.columns(3)  # Distribuye los grupos en 3 columnas.
        for idx, group in enumerate(group_names):  # Recorre cada grupo.
            default_teams = groups_default[group]  # Selecciones por defecto de ese grupo.
            with editor_cols[idx % 3]:  # Reparte los grupos entre las 3 columnas.
                st.markdown(f"**Grupo {group}**")  # Titulo del grupo.
                selected = []  # Selecciones elegidas para este grupo.
                for i, default in enumerate(default_teams):  # Recorre las 4 posiciones del grupo.
                    options = [team for team in all_teams if team not in used or team == default]  # Evita repetir selecciones ya usadas.
                    if default not in options:  # Si el valor por defecto ya fue tomado por otro grupo.
                        default = options[0]  # Usa la primera opcion disponible como respaldo.
                    value = st.selectbox(
                        f"{group}{i + 1}",
                        options=options,
                        index=options.index(default),
                        key=f"{group}-{i}",
                        label_visibility="collapsed",
                    )  # Selector de la seleccion para esa posicion del grupo.
                    selected.append(value)  # Guarda la eleccion.
                    used.add(value)  # Marca la seleccion como usada.
                groups_selected[group] = selected  # Guarda el grupo completo ya editado.

    with st.spinner("Calculando probabilidades de grupos..."):  # Indicador de carga mientras se calculan las tablas.
        qualified, tables = qualified_teams(groups_selected)  # Recalcula clasificados y tablas con los grupos actuales.

    st.subheader("Tablas proyectadas por grupo")  # Titulo de la seccion de tablas.
    st.markdown(
        """
        <div class="legend-row">
            <span><span class="legend-dot" style="background:#22c55e"></span>1º y 2º clasifican directo</span>
            <span><span class="legend-dot" style="background:#eab308"></span>3º puede clasificar entre los 8 mejores terceros</span>
            <span><span class="legend-dot" style="background:#ef4444"></span>4º queda eliminado</span>
        </div>
        """,
        unsafe_allow_html=True,
    )  # Leyenda de colores para interpretar las tablas.
    group_tabs = st.tabs([f"Grupo {group}" for group in groups_selected.keys()])  # Una pestaña interna por grupo.
    for tab, group in zip(group_tabs, groups_selected.keys()):  # Recorre cada pestana de grupo.
        with tab:
            show_group_table(tables[group])  # Muestra la tabla de ese grupo.

    st.subheader("✅ Clasificados proyectados a la fase eliminatoria")  # Titulo de la seccion de clasificados.
    direct_qualified = qualified[: 2 * len(groups_selected)]  # Los clasificados directos (1o y 2o de cada grupo).
    third_qualified = qualified[2 * len(groups_selected):]  # Los mejores terceros clasificados.
    chips = "".join(
        f"<span class='team-chip direct'>{flag(team)}&nbsp;{escape(str(team))}</span>" for team in direct_qualified
    ) + "".join(
        f"<span class='team-chip third'>{flag(team)}&nbsp;{escape(str(team))}</span>" for team in third_qualified
    )  # Arma las etiquetas visuales de clasificados directos y terceros.
    st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)  # Muestra la lista de clasificados.
    st.caption("Verde: clasificados directos (1º y 2º). Amarillo: mejores terceros.")  # Explica los colores usados.

# ---------------------------------------------------------------------------
# Simulacion de campeones (se calcula una vez y se reutiliza en dos pestañas)
# ---------------------------------------------------------------------------
groups_json = json.dumps(groups_selected, sort_keys=True)  # Serializa los grupos actuales para poder cachear la simulacion.
sim_progress = st.empty()  # Contenedor donde vivira la barra de progreso.
sim_status = st.empty()  # Contenedor donde vivira el texto de estado.
progress_widget = sim_progress.progress(0.0)  # Barra de progreso inicial en 0%.
with st.spinner("Ejecutando simulaciones de campeón..."):  # Indicador de carga mientras corren las simulaciones.
    champion_results = cached_champion_table(groups_json, int(sims), progress_widget, sim_status)  # Corre (o recupera de cache) la simulacion Monte Carlo del torneo.
sim_progress.empty()  # Limpia la barra de progreso al terminar.
sim_status.empty()  # Limpia el texto de estado al terminar.
top10 = champion_results.head(10)  # Top 10 de selecciones con mayor probabilidad de ser campeonas.

with tab_champions:  # Vista de campeones probables.
    st.subheader("Top 10 selecciones con mayor probabilidad de campeonar")  # Titulo de la seccion.
    st.caption(f"Basado en {int(sims):,} torneos simulados con emparejamientos aleatorios entre los 32 clasificados.")  # Contexto de la simulacion.
    chart_col, table_col = st.columns([0.5, 0.5])  # Divide en grafico y tabla lado a lado.
    with chart_col:
        st.pyplot(champion_bar_chart(top10), use_container_width=True)  # Grafico de barras horizontal del top 10.
    with table_col:
        show_champion_table(champion_results.head(15))  # Tabla ampliada (top 15) con historico de titulos reales.

with tab_bracket:  # Vista de cuadro de eliminacion directa.
    seeded_qualified = seed_bracket_by_top_champions(qualified, top10["seleccion"].tolist())  # Ordena el cuadro usando el top 10 de campeones probables.
    with st.spinner("Armando bracket de eliminación..."):  # Indicador de carga mientras se arma el bracket.
        bracket, champion = build_bracket(seeded_qualified, forced_team)  # Simula todas las rondas del bracket.

    st.markdown(
        f"""
        <div class="champ-card">
            <div class="champ-trophy">🏆</div>
            <div>
                <div class="champ-label">Campeón proyectado del bracket</div>
                <div class="champ-name">{flag(champion)} {escape(str(champion))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )  # Destaca al campeon proyectado del bracket actual.
    if forced_team:  # Si el usuario forzo un ganador desde la barra lateral.
        st.info(f"Estás forzando a {flag(forced_team)} **{forced_team}** como ganador de todos sus cruces.")  # Aviso explicito del forzado.
    st.caption(
        "El cuadro se siembra con el top 10 de la simulación (borde dorado). "
        "En verde, el ganador proyectado de cada cruce; cada partido muestra las tres probabilidades del modelo "
        "(victoria A, empate, victoria B)."
    )  # Explica como leer el bracket.
    render_bracket(bracket, top10)  # Dibuja el cuadro completo de eliminacion directa.
