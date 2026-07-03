from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
APP_PATH = ROOT / "app_streamlit.py"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def write_nb(path: Path, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


app_streamlit = r'''
from pathlib import Path
import itertools
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Simulador Mundial 2026", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "artifacts" / "data"
MODEL_DIR = ROOT / "artifacts" / "models"


@st.cache_resource
def load_model_and_bundle():
    bundle = joblib.load(MODEL_DIR / "dashboard_sklearn.pkl")
    return bundle


@st.cache_data
def load_tables():
    groups = json.loads((DATA_DIR / "groups_2026.json").read_text(encoding="utf-8"))
    team_state = pd.read_csv(DATA_DIR / "team_state_2026.csv")
    team_history = pd.read_csv(DATA_DIR / "team_match_history.csv", parse_dates=["date"])
    champions = pd.read_csv(DATA_DIR / "champion_history.csv")
    return groups, team_state, team_history, champions


bundle = load_model_and_bundle()
groups_default, team_state, team_history, champion_history = load_tables()
feature_cols = bundle["feature_cols"]
all_teams = sorted(team_state["team"].tolist())
state_by_team = team_state.set_index("team").to_dict(orient="index")
PREDICTION_CACHE = {}


def h2h_stats(team_a, team_b):
    pair = " vs ".join(sorted([team_a, team_b]))
    recent = team_history[(team_history["pair"] == pair) & (team_history["team"] == team_a)].sort_values("date").tail(10)
    if recent.empty:
        return {"win": 0.0, "draw": 0.0, "loss": 0.0, "goal_diff": 0.0}
    return {
        "win": float(recent["win"].sum()),
        "draw": float(recent["draw"].sum()),
        "loss": float(recent["loss"].sum()),
        "goal_diff": float(recent["goal_diff"].sum()),
    }


def make_features(team_a, team_b):
    a = state_by_team[team_a]
    b = state_by_team[team_b]
    h2h_a = h2h_stats(team_a, team_b)
    h2h_b = h2h_stats(team_b, team_a)
    values = {
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
        "neutral": 1.0,
    }
    for name, left, right in [
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
        values[f"diff_{name}"] = values[left] - values[right]
    return pd.DataFrame([{col: values.get(col, 0.0) for col in feature_cols}])


def predict_match(team_a, team_b):
    cache_key = (team_a, team_b)
    if cache_key in PREDICTION_CACHE:
        return PREDICTION_CACHE[cache_key]
    row = make_features(team_a, team_b)
    probs = bundle["classifier"].predict_proba(row)[0]
    goals = np.array([bundle["home_goals"].predict(row)[0], bundle["away_goals"].predict(row)[0]])
    probs = probs / probs.sum()
    result = {
        "team_a": team_a,
        "team_b": team_b,
        "p_a": float(probs[0]),
        "p_draw": float(probs[1]),
        "p_b": float(probs[2]),
        "g_a": float(np.clip(goals[0], 0, 6)),
        "g_b": float(np.clip(goals[1], 0, 6)),
    }
    PREDICTION_CACHE[cache_key] = result
    return result


def group_table(teams):
    rows = {team: {"seleccion": team, "pts": 0.0, "gf": 0.0, "gc": 0.0} for team in teams}
    for team_a, team_b in itertools.combinations(teams, 2):
        pred = predict_match(team_a, team_b)
        rows[team_a]["pts"] += 3 * pred["p_a"] + pred["p_draw"]
        rows[team_b]["pts"] += 3 * pred["p_b"] + pred["p_draw"]
        rows[team_a]["gf"] += pred["g_a"]
        rows[team_a]["gc"] += pred["g_b"]
        rows[team_b]["gf"] += pred["g_b"]
        rows[team_b]["gc"] += pred["g_a"]
    table = pd.DataFrame(rows.values())
    table["dg"] = table["gf"] - table["gc"]
    table = table.sort_values(["pts", "dg", "gf"], ascending=False).reset_index(drop=True)
    table["pos"] = np.arange(1, len(table) + 1)
    table["clasificacion_%"] = np.select([table["pos"] <= 2, table["pos"] == 3], [92.0, 55.0], default=12.0)
    return table.round({"pts": 2, "gf": 2, "gc": 2, "dg": 2, "clasificacion_%": 1})


def qualified_teams(groups):
    tables = {group: group_table(teams) for group, teams in groups.items()}
    direct = []
    thirds = []
    for group, table in tables.items():
        top_two = table.head(2).copy()
        direct.extend(top_two["seleccion"].tolist())
        third = table.iloc[2].copy()
        third["grupo"] = group
        thirds.append(third)
    best_thirds = pd.DataFrame(thirds).sort_values(["pts", "dg", "gf"], ascending=False).head(8)
    return (direct + best_thirds["seleccion"].tolist())[:32], tables


def build_bracket(teams, forced_team=None):
    current = list(teams)
    matches = []
    round_names = ["Ronda de 32", "Octavos", "Cuartos", "Semifinal", "Final"]
    for round_name in round_names:
        winners = []
        for idx in range(0, len(current), 2):
            team_a, team_b = current[idx], current[idx + 1]
            pred = predict_match(team_a, team_b)
            if forced_team in [team_a, team_b]:
                winner = forced_team
            else:
                winner = team_a if pred["p_a"] >= pred["p_b"] else team_b
            winners.append(winner)
            matches.append({**pred, "ronda": round_name, "ganador": winner})
        current = winners
        if len(current) == 1:
            break
    return pd.DataFrame(matches), current[0]


def champion_simulation_table(groups, simulations):
    counts = {}
    qualified, _ = qualified_teams(groups)
    rng = np.random.default_rng(42)
    for _ in range(simulations):
        current = list(rng.permutation(qualified))
        while len(current) > 1:
            next_round = []
            for idx in range(0, len(current), 2):
                pred = predict_match(current[idx], current[idx + 1])
                probs = np.array([pred["p_a"] + 0.5 * pred["p_draw"], pred["p_b"] + 0.5 * pred["p_draw"]])
                probs = probs / probs.sum()
                next_round.append(rng.choice([current[idx], current[idx + 1]], p=probs))
            current = next_round
        counts[current[0]] = counts.get(current[0], 0) + 1
    table = pd.DataFrame({"seleccion": list(counts), "veces_campeon": list(counts.values())})
    table["simulaciones"] = simulations
    table["winrate_%"] = 100 * table["veces_campeon"] / simulations
    table = table.merge(champion_history, on="seleccion", how="left")
    table["mundiales_ganados"] = table["mundiales_ganados"].fillna(0).astype(int)
    return table.sort_values(["veces_campeon", "winrate_%"], ascending=False).reset_index(drop=True)


def champion_bar_chart(top10):
    fig, ax = plt.subplots(figsize=(8, 5))
    chart_data = top10.sort_values("winrate_%", ascending=True)
    bars = ax.barh(chart_data["seleccion"], chart_data["winrate_%"], color="#3b82f6")
    ax.set_xlabel("Winrate (%)")
    ax.set_ylabel("Seleccion")
    ax.set_title("Top 10 selecciones con mayor probabilidad de campeonar")
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in chart_data["winrate_%"]], padding=4)
    ax.set_xlim(0, max(5, chart_data["winrate_%"].max() * 1.18))
    fig.tight_layout()
    return fig


def render_bracket(bracket):
    round_order = ["Ronda de 32", "Octavos", "Cuartos", "Semifinal", "Final"]
    st.markdown(
        """
        <style>
        .bracket-title {
            font-weight: 700;
            margin-bottom: 0.55rem;
            text-align: center;
        }
        .match-card {
            border: 1px solid #d7dde8;
            border-left: 5px solid #2563eb;
            border-radius: 8px;
            padding: 0.55rem 0.65rem;
            margin-bottom: 0.55rem;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
            min-height: 92px;
        }
        .team-line {
            display: flex;
            justify-content: space-between;
            gap: 0.4rem;
            font-size: 0.88rem;
            line-height: 1.25rem;
        }
        .winner {
            color: #0f766e;
            font-weight: 700;
            margin-top: 0.35rem;
            font-size: 0.86rem;
        }
        .prob-line {
            color: #475569;
            font-size: 0.75rem;
            margin-top: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(len(round_order))
    for col, round_name in zip(cols, round_order):
        round_matches = bracket[bracket["ronda"] == round_name].reset_index(drop=True)
        with col:
            st.markdown(f"<div class='bracket-title'>{round_name}</div>", unsafe_allow_html=True)
            for idx, row in round_matches.iterrows():
                st.markdown(
                    f"""
                    <div class="match-card">
                        <div class="team-line"><span>{row['team_a']}</span><strong>{row['p_a']:.0%}</strong></div>
                        <div class="team-line"><span>{row['team_b']}</span><strong>{row['p_b']:.0%}</strong></div>
                        <div class="prob-line">Empate: {row['p_draw']:.0%}</div>
                        <div class="winner">Avanza: {row['ganador']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


st.title("Sistema de Prediccion e Inteligencia Deportiva - Mundial 2026")
st.caption("Dashboard funcional para simular grupos, eliminatorias y campeones probables.")

groups_selected = {}
used = set()
left, right = st.columns([0.35, 0.65])

with left:
    st.subheader("Grupos editables")
    for group, default_teams in groups_default.items():
        with st.expander(f"Grupo {group}", expanded=group in ["A", "B"]):
            selected = []
            for i, default in enumerate(default_teams):
                options = [team for team in all_teams if team not in used or team == default]
                if default not in options:
                    default = options[0]
                value = st.selectbox(f"{group}{i + 1}", options=options, index=options.index(default), key=f"{group}-{i}")
                selected.append(value)
                used.add(value)
            groups_selected[group] = selected

with right:
    st.subheader("Tablas proyectadas")
    qualified, tables = qualified_teams(groups_selected)
    tabs = st.tabs(list(groups_selected.keys()))
    for tab, group in zip(tabs, groups_selected.keys()):
        with tab:
            st.dataframe(tables[group], use_container_width=True, hide_index=True)

st.divider()
forced = st.selectbox("Forzar ganador si aparece en un cruce", ["Sin forzar"] + all_teams)
forced_team = None if forced == "Sin forzar" else forced
bracket, champion = build_bracket(qualified, forced_team)

col1, col2 = st.columns([0.58, 0.42])
with col1:
    st.subheader("Cuadro de eliminacion")
    render_bracket(bracket)
    st.metric("Campeon proyectado del bracket", champion)

with col2:
    st.subheader("Top 10 campeones probables")
    sims = st.number_input("Numero de simulaciones", min_value=1, max_value=2000, value=100, step=25)
    champion_results = champion_simulation_table(groups_selected, int(sims))
    top10 = champion_results.head(10)
    st.pyplot(champion_bar_chart(top10), use_container_width=True)
    st.dataframe(
        champion_results.rename(
            columns={
                "seleccion": "Seleccion",
                "veces_campeon": "Veces campeon",
                "simulaciones": "Simulaciones",
                "winrate_%": "Winrate %",
                "mundiales_ganados": "Mundiales ganados",
            }
        ).round({"Winrate %": 2}),
        use_container_width=True,
        hide_index=True,
    )

'''


nb1 = [
    md(
        """
        # Notebook 01 - Preparacion de datos y features

        Este notebook limpia los datasets obligatorios, construye variables de ranking, forma reciente,
        enfrentamientos directos y atributos agregados de jugadores. Tambien reconstruye los 12 grupos
        del Mundial 2026 desde el calendario.
        """
    ),
    code(
        """
        from pathlib import Path  # Permite usar rutas compatibles con Windows y Colab.
        import json  # Guarda estructuras como los grupos en formato reutilizable.
        import warnings  # Controla avisos menores durante la ejecucion.

        import numpy as np  # Realiza calculos numericos y manejo de arreglos.
        import pandas as pd  # Carga y transforma los archivos CSV.
        import matplotlib.pyplot as plt  # Genera graficos de verificacion.

        warnings.filterwarnings("ignore")  # Evita que avisos repetidos oculten la salida importante.
        pd.set_option("display.max_columns", 120)  # Muestra suficientes columnas al inspeccionar tablas.
        plt.style.use("ggplot")  # Usa un estilo grafico disponible sin dependencias extra.
        """
    ),
    code(
        """
        ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()  # Detecta raiz del proyecto.
        DATA_DIR = ROOT  # Define donde estan las carpetas archive.
        OUT_DATA = ROOT / "artifacts" / "data"  # Define carpeta de datos procesados.
        OUT_DATA.mkdir(parents=True, exist_ok=True)  # Crea carpeta de salida si no existe.

        PATH_RESULTS = DATA_DIR / "archive (4)" / "results.csv"  # Historial internacional de partidos.
        PATH_RANKING_HISTORY = DATA_DIR / "archive (5)" / "fifa_ranking-2024-06-20.csv"  # Ranking FIFA historico.
        PATH_RANKING_2026 = DATA_DIR / "archive (3)" / "fifa_ranking_2026-06-08.csv"  # Ranking usado para simular 2026.
        PATH_PLAYERS = DATA_DIR / "archive (6)" / "players_21.csv"  # Atributos FIFA de jugadores.
        PATH_SCHEDULE = DATA_DIR / "archive (3)" / "schedule_2026.csv"  # Calendario de grupos 2026.
        PATH_WORLD_CUP = DATA_DIR / "archive (3)" / "world_cup.csv"  # Campeones historicos.
        """
    ),
    code(
        """
        results_raw = pd.read_csv(PATH_RESULTS)  # Carga el historial de partidos.
        ranking_history_raw = pd.read_csv(PATH_RANKING_HISTORY)  # Carga ranking FIFA por fecha.
        ranking_2026_raw = pd.read_csv(PATH_RANKING_2026)  # Carga ranking de referencia para 2026.
        players_raw = pd.read_csv(PATH_PLAYERS)  # Carga jugadores y atributos.
        schedule_raw = pd.read_csv(PATH_SCHEDULE)  # Carga partidos programados de 2026.
        world_cup_raw = pd.read_csv(PATH_WORLD_CUP)  # Carga campeones por edicion.

        print("Partidos:", results_raw.shape)  # Muestra tamano del historial.
        print("Ranking historico:", ranking_history_raw.shape)  # Muestra tamano del ranking.
        print("Jugadores:", players_raw.shape)  # Muestra tamano de jugadores.
        print("Calendario 2026:", schedule_raw.shape)  # Muestra partidos 2026.
        """
    ),
    code(
        """
        NAME_MAP = {  # Unifica nombres distintos entre fuentes.
            "Bosnia Herzegovina": "Bosnia-Herzegovina",  # Une jugadores con calendario.
            "Bosnia and Herzegovina": "Bosnia-Herzegovina",  # Une ranking/resultados con calendario.
            "Czech Republic": "Czechia",  # Usa nombre moderno.
            "DR Congo": "Congo DR",  # Usa forma del calendario.
            "Iran": "IR Iran",  # Usa forma FIFA.
            "Ivory Coast": "Côte d'Ivoire",  # Usa forma FIFA.
            "South Korea": "Korea Republic",  # Usa forma FIFA.
            "Turkey": "Türkiye",  # Usa forma del calendario.
            "Curacao": "Curaçao",  # Conserva forma del calendario.
            "United States of America": "United States",  # Une variantes del pais.
            "USA": "United States",  # Une abreviatura.
        }  # Termina diccionario de nombres.

        def clean_team(name):  # Define limpieza reusable.
            if pd.isna(name):  # Mantiene valores faltantes.
                return name  # Devuelve NaN sin modificar.
            text = str(name).strip()  # Quita espacios alrededor.
            return NAME_MAP.get(text, text)  # Aplica equivalencia si existe.
        """
    ),
    code(
        """
        results = results_raw.copy()  # Crea copia del historial.
        results["date"] = pd.to_datetime(results["date"], errors="coerce")  # Convierte fechas.
        results["home_team"] = results["home_team"].map(clean_team)  # Normaliza equipo local.
        results["away_team"] = results["away_team"].map(clean_team)  # Normaliza equipo visitante.
        results = results.dropna(subset=["date", "home_score", "away_score"])  # Quita partidos sin marcador.
        results["home_score"] = results["home_score"].astype(int)  # Asegura goles locales enteros.
        results["away_score"] = results["away_score"].astype(int)  # Asegura goles visitantes enteros.
        results = results[results["date"] >= "1993-01-01"].sort_values("date").reset_index(drop=True)  # Usa etapa con ranking FIFA.
        results["match_id"] = np.arange(len(results))  # Crea identificador unico.

        ranking_history = ranking_history_raw.copy()  # Copia ranking historico.
        ranking_history["team"] = ranking_history["country_full"].map(clean_team)  # Normaliza pais.
        ranking_history["rank_date"] = pd.to_datetime(ranking_history["rank_date"], errors="coerce")  # Convierte fecha de ranking.
        ranking_history = ranking_history.rename(columns={"total_points": "ranking_points"})  # Renombra puntos.
        ranking_history = ranking_history[["team", "rank_date", "rank", "ranking_points"]].dropna(subset=["rank_date"])  # Conserva columnas utiles.
        ranking_history = ranking_history.sort_values(["team", "rank_date"]).drop_duplicates(["team", "rank_date"])  # Evita duplicados.

        ranking_2026 = ranking_2026_raw.copy()  # Copia ranking 2026.
        ranking_2026["team"] = ranking_2026["team"].map(clean_team)  # Normaliza nombres.
        ranking_2026 = ranking_2026.rename(columns={"points": "ranking_points"})  # Usa mismo nombre de puntos.
        """
    ),
    code(
        """
        schedule = schedule_raw.copy()  # Copia calendario.
        schedule["home_team"] = schedule["home_team"].map(clean_team)  # Normaliza local.
        schedule["away_team"] = schedule["away_team"].map(clean_team)  # Normaliza visitante.
        schedule["order"] = np.arange(len(schedule))  # Guarda orden original.

        graph = {}  # Crea grafo equipo-rival para detectar grupos.
        first_seen = {}  # Guarda primera aparicion por equipo.
        for row in schedule.itertuples():  # Recorre partidos de grupo.
            graph.setdefault(row.home_team, set()).add(row.away_team)  # Conecta local con visitante.
            graph.setdefault(row.away_team, set()).add(row.home_team)  # Conecta visitante con local.
            first_seen.setdefault(row.home_team, row.order)  # Registra aparicion local.
            first_seen.setdefault(row.away_team, row.order)  # Registra aparicion visitante.

        groups = {}  # Guarda grupos reconstruidos.
        visited = set()  # Guarda equipos ya procesados.
        for seed in sorted(graph, key=lambda x: first_seen[x]):  # Recorre equipos por aparicion.
            if seed in visited:  # Evita repetir componentes.
                continue  # Salta si ya fue asignado.
            stack = [seed]  # Inicia busqueda.
            component = []  # Guarda componente del grupo.
            visited.add(seed)  # Marca semilla.
            while stack:  # Explora vecinos.
                team = stack.pop()  # Toma equipo pendiente.
                component.append(team)  # Agrega a componente.
                for opponent in graph[team]:  # Recorre rivales.
                    if opponent not in visited:  # Solo toma no visitados.
                        visited.add(opponent)  # Marca rival.
                        stack.append(opponent)  # Programa rival.
            label = chr(ord("A") + len(groups))  # Crea etiqueta A-L.
            groups[label] = sorted(component, key=lambda x: first_seen[x])  # Ordena equipos por calendario.

        (OUT_DATA / "groups_2026.json").write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")  # Guarda grupos.
        print(groups)  # Muestra grupos reconstruidos.
        """
    ),
    code(
        """
        def attach_rank(df, side):  # Une ranking previo a cada partido.
            team_col = f"{side}_team"  # Define columna del equipo.
            pieces = []  # Acumula partes por seleccion.
            for team, part in df.groupby(team_col):  # Procesa cada seleccion.
                left = part[["match_id", "date"]].sort_values("date")  # Toma fechas de sus partidos.
                hist = ranking_history[ranking_history["team"] == team].sort_values("rank_date")  # Toma ranking del equipo.
                if hist.empty:  # Maneja equipos sin ranking.
                    merged = left.assign(rank=np.nan, ranking_points=np.nan)  # Crea faltantes controlados.
                else:  # Si hay ranking disponible.
                    merged = pd.merge_asof(left, hist, left_on="date", right_on="rank_date", direction="backward")  # Une ranking anterior.
                pieces.append(merged[["match_id", "rank", "ranking_points"]])  # Guarda columnas utiles.
            out = pd.concat(pieces, ignore_index=True)  # Une todas las selecciones.
            return out.rename(columns={"rank": f"{side}_rank", "ranking_points": f"{side}_ranking_points"})  # Prefija columnas.

        matches = results.copy()  # Crea tabla modelable.
        matches = matches.merge(attach_rank(matches, "home"), on="match_id", how="left")  # Une ranking local.
        matches = matches.merge(attach_rank(matches, "away"), on="match_id", how="left")  # Une ranking visitante.
        matches["home_rank"] = matches["home_rank"].fillna(211)  # Imputa peor ranking local.
        matches["away_rank"] = matches["away_rank"].fillna(211)  # Imputa peor ranking visitante.
        matches["home_ranking_points"] = matches["home_ranking_points"].fillna(matches["home_ranking_points"].median())  # Imputa puntos local.
        matches["away_ranking_points"] = matches["away_ranking_points"].fillna(matches["away_ranking_points"].median())  # Imputa puntos visitante.
        """
    ),
    code(
        """
        home_long = matches[["match_id", "date", "home_team", "away_team", "home_score", "away_score"]].copy()  # Vista local.
        home_long = home_long.rename(columns={"home_team": "team", "away_team": "opponent", "home_score": "goals_for", "away_score": "goals_against"})  # Nombres desde local.
        home_long["is_home"] = 1  # Marca local.

        away_long = matches[["match_id", "date", "away_team", "home_team", "away_score", "home_score"]].copy()  # Vista visitante.
        away_long = away_long.rename(columns={"away_team": "team", "home_team": "opponent", "away_score": "goals_for", "home_score": "goals_against"})  # Nombres desde visitante.
        away_long["is_home"] = 0  # Marca visitante.

        team_history = pd.concat([home_long, away_long], ignore_index=True)  # Une ambas vistas.
        team_history["points"] = np.select([team_history["goals_for"] > team_history["goals_against"], team_history["goals_for"] == team_history["goals_against"]], [3, 1], default=0)  # Calcula puntos.
        team_history["win"] = (team_history["goals_for"] > team_history["goals_against"]).astype(int)  # Marca victoria.
        team_history["draw"] = (team_history["goals_for"] == team_history["goals_against"]).astype(int)  # Marca empate.
        team_history["loss"] = (team_history["goals_for"] < team_history["goals_against"]).astype(int)  # Marca derrota.
        team_history["goal_diff"] = team_history["goals_for"] - team_history["goals_against"]  # Calcula diferencia.
        team_history["pair"] = team_history.apply(lambda row: " vs ".join(sorted([row["team"], row["opponent"]])), axis=1)  # Crea llave H2H.

        team_history = team_history.sort_values(["team", "date", "match_id"]).reset_index(drop=True)  # Ordena por equipo y fecha.
        for col in ["goals_for", "goals_against", "points", "goal_diff"]:  # Recorre variables recientes.
            team_history[f"{col}_last10"] = team_history.groupby("team")[col].transform(lambda s: s.shift().rolling(10, min_periods=1).mean())  # Calcula promedio previo.

        team_history = team_history.sort_values(["pair", "team", "date", "match_id"]).reset_index(drop=True)  # Ordena para H2H.
        for col in ["win", "draw", "loss", "goal_diff"]:  # Recorre variables H2H.
            team_history[f"h2h_{col}_last10"] = team_history.groupby(["pair", "team"])[col].transform(lambda s: s.shift().rolling(10, min_periods=1).sum())  # Suma duelos previos.

        form_cols = ["goals_for_last10", "goals_against_last10", "points_last10", "goal_diff_last10", "h2h_win_last10", "h2h_draw_last10", "h2h_loss_last10", "h2h_goal_diff_last10"]  # Lista features de forma.
        team_history[form_cols] = team_history[form_cols].fillna(0)  # Rellena equipos sin historial previo.
        """
    ),
    code(
        """
        home_features = team_history[team_history["is_home"] == 1][["match_id"] + form_cols].copy()  # Features del local.
        home_features = home_features.rename(columns={col: f"home_{col}" for col in form_cols})  # Prefijo local.
        away_features = team_history[team_history["is_home"] == 0][["match_id"] + form_cols].copy()  # Features del visitante.
        away_features = away_features.rename(columns={col: f"away_{col}" for col in form_cols})  # Prefijo visitante.
        matches = matches.merge(home_features, on="match_id", how="left")  # Une forma local.
        matches = matches.merge(away_features, on="match_id", how="left")  # Une forma visitante.
        """
    ),
    code(
        """
        players = players_raw.copy()  # Copia datos de jugadores.
        players["team"] = players["nationality"].map(clean_team)  # Normaliza nacionalidad.
        attr_cols = ["overall", "pace", "shooting", "defending", "physic"]  # Atributos pedidos por rubrica.
        players[attr_cols] = players[attr_cols].apply(pd.to_numeric, errors="coerce")  # Convierte atributos a numeros.
        players["pace"] = players["pace"].fillna(players["movement_sprint_speed"])  # Imputa ritmo para porteros.
        players["shooting"] = players["shooting"].fillna(players["attacking_finishing"])  # Imputa tiro.
        players["defending"] = players["defending"].fillna(players["defending_standing_tackle"])  # Imputa defensa.
        players["physic"] = players["physic"].fillna(players["power_strength"])  # Imputa fisico.
        player_agg = players.sort_values(["team", "overall"], ascending=[True, False]).groupby("team").head(23)  # Simula convocados top 23.
        player_agg = player_agg.groupby("team")[attr_cols].mean().reset_index()  # Promedia por seleccion.
        player_agg = player_agg.rename(columns={"physic": "physical"})  # Usa nombre physical.
        for col in ["overall", "pace", "shooting", "defending", "physical"]:  # Recorre atributos finales.
            player_agg[col] = player_agg[col].fillna(player_agg[col].median())  # Imputa faltantes.
        player_agg.to_csv(OUT_DATA / "player_aggregates.csv", index=False)  # Guarda agregados.
        """
    ),
    code(
        """
        def add_players(df, side):  # Une atributos de jugadores a un lado.
            team_col = f"{side}_team"  # Define columna del equipo.
            renamed = player_agg.rename(columns={col: f"{side}_player_{col}" for col in ["overall", "pace", "shooting", "defending", "physical"]})  # Prefija atributos.
            merged = df.merge(renamed, left_on=team_col, right_on="team", how="left")  # Une por equipo.
            return merged.drop(columns=["team"])  # Quita columna auxiliar.

        matches = add_players(matches, "home")  # Une atributos local.
        matches = add_players(matches, "away")  # Une atributos visitante.
        for col in [c for c in matches.columns if "_player_" in c]:  # Recorre columnas de jugadores.
            matches[col] = matches[col].fillna(matches[col].median())  # Imputa equipos sin jugadores.
        """
    ),
    code(
        """
        matches["target_result"] = np.select([matches["home_score"] > matches["away_score"], matches["home_score"] == matches["away_score"]], [0, 1], default=2)  # 0 gana A, 1 empate, 2 gana B.
        matches["target_home_goals"] = matches["home_score"]  # Objetivo auxiliar de goles A.
        matches["target_away_goals"] = matches["away_score"]  # Objetivo auxiliar de goles B.
        matches["is_world_cup_2022_test"] = ((matches["tournament"] == "FIFA World Cup") & (matches["date"] >= "2022-11-20") & (matches["date"] <= "2022-12-18")).astype(int)  # Reserva Mundial 2022.

        feature_cols = [
            "home_rank", "away_rank", "home_ranking_points", "away_ranking_points",
            "home_goals_for_last10", "away_goals_for_last10", "home_goals_against_last10", "away_goals_against_last10",
            "home_points_last10", "away_points_last10", "home_goal_diff_last10", "away_goal_diff_last10",
            "home_h2h_win_last10", "away_h2h_win_last10", "home_h2h_draw_last10", "away_h2h_draw_last10",
            "home_h2h_loss_last10", "away_h2h_loss_last10", "home_h2h_goal_diff_last10", "away_h2h_goal_diff_last10",
            "home_player_overall", "away_player_overall", "home_player_pace", "away_player_pace",
            "home_player_shooting", "away_player_shooting", "home_player_defending", "away_player_defending",
            "home_player_physical", "away_player_physical", "neutral",
        ]  # Features base.

        for name, left, right in [
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
        ]:  # Recorre pares de diferencia.
            matches[f"diff_{name}"] = matches[left] - matches[right]  # Calcula ventaja del equipo A.
            feature_cols.append(f"diff_{name}")  # Agrega diferencia a features.

        matches["neutral"] = matches["neutral"].astype(float)  # Convierte booleano neutral a numero.
        matches[feature_cols] = matches[feature_cols].replace([np.inf, -np.inf], np.nan)  # Limpia infinitos.
        matches[feature_cols] = matches[feature_cols].fillna(matches[feature_cols].median(numeric_only=True))  # Imputa faltantes.
        matches.to_csv(OUT_DATA / "feature_dataset.csv", index=False)  # Guarda dataset final.
        pd.Series(feature_cols).to_csv(OUT_DATA / "feature_columns.csv", index=False, header=["feature"])  # Guarda orden de features.
        team_history.to_csv(OUT_DATA / "team_match_history.csv", index=False)  # Guarda historial para LSTM y dashboard.
        """
    ),
    code(
        """
        teams_2026 = sorted(set(schedule["home_team"]) | set(schedule["away_team"]))  # Lista equipos del Mundial 2026.
        latest_form = team_history.sort_values("date").groupby("team").tail(1).set_index("team")  # Toma forma mas reciente.
        state = pd.DataFrame({"team": teams_2026})  # Crea tabla estado 2026.
        state = state.merge(ranking_2026[["team", "rank", "ranking_points"]], on="team", how="left")  # Une ranking 2026.
        state = state.merge(latest_form[["goals_for_last10", "goals_against_last10", "points_last10", "goal_diff_last10"]].reset_index(), on="team", how="left")  # Une forma reciente.
        state = state.merge(player_agg, on="team", how="left")  # Une atributos de jugadores.
        state["rank"] = state["rank"].fillna(211)  # Imputa ranking faltante.
        state["ranking_points"] = state["ranking_points"].fillna(state["ranking_points"].median())  # Imputa puntos faltantes.
        num_cols = state.select_dtypes(include=[np.number]).columns  # Detecta columnas numericas.
        state[num_cols] = state[num_cols].fillna(state[num_cols].median(numeric_only=True))  # Imputa faltantes numericos.
        state.to_csv(OUT_DATA / "team_state_2026.csv", index=False)  # Guarda estado para dashboard.

        champions = world_cup_raw.copy()  # Copia campeones historicos.
        champions["seleccion"] = champions["Champion"].map(clean_team)  # Normaliza campeon.
        champions = champions.groupby("seleccion").size().reset_index(name="mundiales_ganados")  # Cuenta titulos.
        champions.to_csv(OUT_DATA / "champion_history.csv", index=False)  # Guarda historial de campeones.
        """
    ),
    code(
        """
        print("Dataset final:", matches.shape)  # Muestra tamano final.
        print("Features:", len(feature_cols))  # Muestra numero de features.
        print("Partidos de test Mundial 2022:", int(matches["is_world_cup_2022_test"].sum()))  # Verifica test.
        display(matches[["date", "home_team", "away_team", "home_score", "away_score", "target_result", "is_world_cup_2022_test"]].tail())  # Muestra ejemplos.

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Crea panel de graficos.
        matches["target_result"].value_counts().sort_index().plot(kind="bar", ax=axes[0])  # Grafica clases.
        axes[0].set_title("Distribucion de resultados")  # Titulo grafico 1.
        axes[0].set_xlabel("0 gana A, 1 empate, 2 gana B")  # Etiqueta eje X.
        axes[0].set_ylabel("Partidos")  # Etiqueta eje Y.
        axes[1].hist(matches["diff_ranking_points"], bins=40)  # Grafica diferencia de ranking.
        axes[1].set_title("Diferencia de puntos FIFA")  # Titulo grafico 2.
        axes[1].set_xlabel("Equipo A - Equipo B")  # Etiqueta eje X.
        plt.tight_layout()  # Ajusta espacios.
        plt.show()  # Muestra graficos.
        """
    ),
]


nb2 = [
    md(
        """
        # Notebook 02 - Modelos Deep Learning

        Entrena y compara MLP con Adam, MLP con SGD momentum, LSTM y GRU. El Mundial 2022 queda reservado como test.
        Para que el dashboard siempre funcione con features tabulares, se exporta el mejor MLP como `dashboard_model.keras`.
        """
    ),
    code(
        """
        from pathlib import Path  # Maneja rutas del proyecto.
        import time  # Mide velocidad de modelos recurrentes.
        import warnings  # Controla avisos.

        import joblib  # Guarda scaler y metadatos.
        import numpy as np  # Maneja matrices numericas.
        import pandas as pd  # Carga datos procesados.
        import matplotlib.pyplot as plt  # Grafica curvas y matriz.

        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # Calcula metricas.
        from sklearn.model_selection import train_test_split  # Divide entrenamiento y validacion.
        from sklearn.preprocessing import StandardScaler  # Normaliza variables.

        import tensorflow as tf  # Motor de deep learning.
        from tensorflow import keras  # API Keras.
        from tensorflow.keras import layers, regularizers  # Capas y regularizacion.

        warnings.filterwarnings("ignore")  # Limpia avisos menores.
        np.random.seed(42)  # Fija semilla numpy.
        tf.random.set_seed(42)  # Fija semilla TensorFlow.
        plt.style.use("ggplot")  # Estilo grafico sin paquetes extra.
        """
    ),
    code(
        """
        ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()  # Detecta raiz.
        DATA_DIR = ROOT / "artifacts" / "data"  # Datos procesados.
        MODEL_DIR = ROOT / "artifacts" / "models"  # Modelos exportados.
        MODEL_DIR.mkdir(parents=True, exist_ok=True)  # Crea carpeta de modelos.

        df = pd.read_csv(DATA_DIR / "feature_dataset.csv", parse_dates=["date"])  # Carga dataset final.
        feature_cols = pd.read_csv(DATA_DIR / "feature_columns.csv")["feature"].tolist()  # Carga columnas.
        train_df = df[(df["is_world_cup_2022_test"] == 0) & (df["date"] < "2022-11-20")].copy()  # Entrena antes del test.
        test_df = df[df["is_world_cup_2022_test"] == 1].copy()  # Reserva Mundial 2022.

        X = train_df[feature_cols].astype(float).values  # Matriz de features.
        y = train_df["target_result"].astype(int).values  # Etiquetas de resultado.
        y_goals = train_df[["target_home_goals", "target_away_goals"]].astype(float).values  # Objetivo de marcador.
        X_test = test_df[feature_cols].astype(float).values  # Features de test.
        y_test = test_df["target_result"].astype(int).values  # Etiquetas de test.

        X_train, X_val, y_train, y_val, goals_train, goals_val = train_test_split(X, y, y_goals, test_size=0.2, random_state=42, stratify=y)  # Crea validacion.
        scaler = StandardScaler()  # Inicializa normalizador.
        X_train_s = scaler.fit_transform(X_train)  # Ajusta con train.
        X_val_s = scaler.transform(X_val)  # Transforma validacion.
        X_test_s = scaler.transform(X_test)  # Transforma test.
        print(X_train_s.shape, X_val_s.shape, X_test_s.shape)  # Verifica dimensiones.
        """
    ),
    code(
        """
        def build_mlp(optimizer_name):  # Construye red MLP.
            inputs = keras.Input(shape=(len(feature_cols),), name="features")  # Entrada tabular.
            x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(inputs)  # Capa oculta 1 con L2.
            x = layers.Dropout(0.30)(x)  # Dropout regularizador.
            x = layers.Dense(96, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)  # Capa oculta 2 con L2.
            x = layers.Dropout(0.25)(x)  # Dropout intermedio.
            x = layers.Dense(64, activation="relu")(x)  # Capa oculta 3.
            x = layers.Dropout(0.20)(x)  # Dropout adicional.
            x = layers.Dense(32, activation="relu")(x)  # Capa oculta 4.
            result = layers.Dense(3, activation="softmax", name="resultado")(x)  # Probabilidades A/empate/B.
            goals = layers.Dense(2, activation="relu", name="goles")(x)  # Marcador estimado.
            model = keras.Model(inputs, [result, goals], name=f"mlp_{optimizer_name}")  # Modelo multi-salida.
            optimizer = keras.optimizers.SGD(0.01, momentum=0.9) if optimizer_name == "sgd" else keras.optimizers.Adam(0.001)  # Selecciona optimizador.
            model.compile(optimizer=optimizer, loss={"resultado": "sparse_categorical_crossentropy", "goles": "mse"}, loss_weights={"resultado": 1.0, "goles": 0.20}, metrics={"resultado": ["accuracy"]})  # Compila modelo.
            return model  # Devuelve modelo.

        callbacks = [keras.callbacks.EarlyStopping(monitor="val_resultado_accuracy", mode="max", patience=2, restore_best_weights=True)]  # Evita sobreentrenar.
        """
    ),
    code(
        """
        EPOCHS_MLP = 5  # Epocas moderadas para que ejecute en CPU.
        BATCH = 64  # Tamano de lote.
        mlp_adam = build_mlp("adam")  # Crea MLP Adam.
        hist_adam = mlp_adam.fit(X_train_s, {"resultado": y_train, "goles": goals_train}, validation_data=(X_val_s, {"resultado": y_val, "goles": goals_val}), epochs=EPOCHS_MLP, batch_size=BATCH, callbacks=callbacks, verbose=1)  # Entrena Adam.
        mlp_sgd = build_mlp("sgd")  # Crea MLP SGD.
        hist_sgd = mlp_sgd.fit(X_train_s, {"resultado": y_train, "goles": goals_train}, validation_data=(X_val_s, {"resultado": y_val, "goles": goals_val}), epochs=EPOCHS_MLP, batch_size=BATCH, callbacks=callbacks, verbose=1)  # Entrena SGD.
        """
    ),
    code(
        """
        def plot_history(history, title):  # Grafica entrenamiento.
            hist = pd.DataFrame(history.history)  # Convierte historial a tabla.
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Crea panel.
            axes[0].plot(hist["loss"], label="train")  # Loss train.
            axes[0].plot(hist["val_loss"], label="validation")  # Loss validacion.
            axes[0].set_title(f"{title} - loss")  # Titulo loss.
            axes[0].legend()  # Muestra leyenda.
            axes[1].plot(hist["resultado_accuracy"], label="train")  # Accuracy train.
            axes[1].plot(hist["val_resultado_accuracy"], label="validation")  # Accuracy validacion.
            axes[1].set_title(f"{title} - accuracy")  # Titulo accuracy.
            axes[1].legend()  # Muestra leyenda.
            plt.tight_layout()  # Ajusta espacios.
            plt.show()  # Muestra grafico.

        plot_history(hist_adam, "MLP Adam")  # Grafica Adam.
        plot_history(hist_sgd, "MLP SGD momentum")  # Grafica SGD.
        """
    ),
    md(
        """
        ## LSTM/GRU y vanishing gradient

        La red recurrente recibe una ventana de los ultimos 10 partidos. En BPTT el gradiente contiene productos
        repetidos de Jacobianos:

        $$\\frac{\\partial L}{\\partial h_t}=\\sum_{k=t}^{T}\\frac{\\partial L}{\\partial h_k}\\prod_{j=t+1}^{k}\\frac{\\partial h_j}{\\partial h_{j-1}}$$

        Si la norma de muchos factores es menor que 1, el gradiente se reduce al retroceder en el tiempo. LSTM y GRU
        usan compuertas para preservar informacion y reducir ese problema.
        """
    ),
    code(
        """
        team_history = pd.read_csv(DATA_DIR / "team_match_history.csv", parse_dates=["date"])  # Carga historial largo.
        sequence_cols = ["goals_for", "goals_against", "points", "goal_diff"]  # Variables temporales.
        LOOKBACK = 10  # Ventana pedida por rubrica.
        history_by_team = {team: part.sort_values("date") for team, part in team_history.groupby("team")}  # Indexa por equipo.

        def recent_sequence(team, date):  # Obtiene ultimos partidos antes de la fecha.
            part = history_by_team.get(team)  # Busca historial del equipo.
            if part is None:  # Maneja equipo sin historial.
                return np.zeros((LOOKBACK, len(sequence_cols)))  # Devuelve ceros.
            values = part[part["date"] < date].tail(LOOKBACK)[sequence_cols].to_numpy(dtype=float)  # Extrae previos.
            padded = np.zeros((LOOKBACK, len(sequence_cols)))  # Crea matriz con padding.
            if len(values) > 0:  # Si existen previos.
                padded[-len(values):] = values  # Alinea al final.
            return padded  # Devuelve secuencia.

        def sequence_for_row(row):  # Crea secuencia comparativa.
            home = recent_sequence(row["home_team"], row["date"])  # Secuencia equipo A.
            away = recent_sequence(row["away_team"], row["date"])  # Secuencia equipo B.
            return np.concatenate([home, away, home - away], axis=1)  # Une A, B y diferencia.

        seq_train_source = train_df.tail(9000).reset_index(drop=True)  # Usa muestra reciente para ejecutar rapido.
        seq_test_source = test_df.reset_index(drop=True)  # Usa todo el test 2022.
        X_seq = np.stack([sequence_for_row(row) for _, row in seq_train_source.iterrows()])  # Construye tensor train.
        y_seq = seq_train_source["target_result"].astype(int).values  # Etiquetas train.
        X_seq_test = np.stack([sequence_for_row(row) for _, row in seq_test_source.iterrows()])  # Construye tensor test.
        X_seq_train, X_seq_val, y_seq_train, y_seq_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)  # Divide secuencias.
        print(X_seq_train.shape, X_seq_val.shape, X_seq_test.shape)  # Verifica tensores.
        """
    ),
    code(
        """
        def build_recurrent(kind):  # Construye LSTM o GRU.
            recurrent = layers.LSTM if kind == "lstm" else layers.GRU  # Selecciona tipo de capa.
            inputs = keras.Input(shape=(LOOKBACK, X_seq_train.shape[-1]), name="trayectoria")  # Entrada secuencial.
            x = recurrent(48, return_sequences=True, dropout=0.20)(inputs)  # Primera capa recurrente.
            x = recurrent(24, dropout=0.20)(x)  # Segunda capa recurrente.
            x = layers.Dense(24, activation="relu")(x)  # Capa densa final.
            outputs = layers.Dense(3, activation="softmax")(x)  # Probabilidades de clase.
            model = keras.Model(inputs, outputs, name=kind)  # Crea modelo.
            model.compile(optimizer=keras.optimizers.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])  # Compila.
            return model  # Devuelve modelo.

        seq_callbacks = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=1, restore_best_weights=True, mode="max")]  # Detiene rapido.
        start = time.time()  # Inicia tiempo LSTM.
        lstm = build_recurrent("lstm")  # Crea LSTM.
        hist_lstm = lstm.fit(X_seq_train, y_seq_train, validation_data=(X_seq_val, y_seq_val), epochs=3, batch_size=BATCH, callbacks=seq_callbacks, verbose=1)  # Entrena LSTM.
        lstm_seconds = time.time() - start  # Guarda tiempo LSTM.
        start = time.time()  # Inicia tiempo GRU.
        gru = build_recurrent("gru")  # Crea GRU.
        hist_gru = gru.fit(X_seq_train, y_seq_train, validation_data=(X_seq_val, y_seq_val), epochs=3, batch_size=BATCH, callbacks=seq_callbacks, verbose=1)  # Entrena GRU.
        gru_seconds = time.time() - start  # Guarda tiempo GRU.
        """
    ),
    code(
        """
        def eval_mlp(model, name):  # Evalua MLP.
            probs, goals = model.predict(X_test_s, verbose=0)  # Predice clases y goles.
            pred = probs.argmax(axis=1)  # Convierte a clase.
            return {"modelo": name, "accuracy": accuracy_score(y_test, pred), "f1_macro": f1_score(y_test, pred, average="macro"), "segundos": np.nan}  # Devuelve metricas.

        def eval_seq(model, name, seconds):  # Evalua recurrente.
            probs = model.predict(X_seq_test, verbose=0)  # Predice clases.
            pred = probs.argmax(axis=1)  # Convierte a clase.
            return {"modelo": name, "accuracy": accuracy_score(y_test, pred), "f1_macro": f1_score(y_test, pred, average="macro"), "segundos": seconds}  # Devuelve metricas.

        metrics = pd.DataFrame([
            eval_mlp(mlp_adam, "MLP Adam"),
            eval_mlp(mlp_sgd, "MLP SGD momentum"),
            eval_seq(lstm, "LSTM", lstm_seconds),
            eval_seq(gru, "GRU", gru_seconds),
        ]).sort_values("f1_macro", ascending=False)  # Compara modelos.
        display(metrics)  # Muestra tabla.
        metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)  # Guarda metricas.
        """
    ),
    code(
        """
        mlp_metrics = metrics[metrics["modelo"].str.contains("MLP")].sort_values("f1_macro", ascending=False)  # Filtra MLP.
        dashboard_name = mlp_metrics.iloc[0]["modelo"]  # Elige mejor MLP para dashboard.
        dashboard_model = mlp_adam if dashboard_name == "MLP Adam" else mlp_sgd  # Recupera modelo elegido.

        mlp_adam.save(MODEL_DIR / "mlp_adam.keras")  # Guarda MLP Adam.
        mlp_sgd.save(MODEL_DIR / "mlp_sgd.keras")  # Guarda MLP SGD.
        lstm.save(MODEL_DIR / "lstm.keras")  # Guarda LSTM.
        gru.save(MODEL_DIR / "gru.keras")  # Guarda GRU.
        dashboard_model.save(MODEL_DIR / "dashboard_model.keras")  # Guarda modelo seguro para dashboard.
        dashboard_model.save(MODEL_DIR / "best_model.keras")  # Guarda alias de modelo elegido.
        joblib.dump({"scaler": scaler, "feature_cols": feature_cols, "dashboard_model": dashboard_name, "metrics": metrics.to_dict(orient="records")}, MODEL_DIR / "model_bundle.pkl")  # Guarda bundle.
        print("Modelo usado en dashboard:", dashboard_name)  # Reporta decision.
        """
    ),
    code(
        """
        probs, goals = dashboard_model.predict(X_test_s, verbose=0)  # Predice con modelo del dashboard.
        pred = probs.argmax(axis=1)  # Obtiene clase estimada.
        print(classification_report(y_test, pred, target_names=["Gana A", "Empate", "Gana B"]))  # Reporte de test.
        cm = confusion_matrix(y_test, pred)  # Calcula matriz.
        fig, ax = plt.subplots(figsize=(5, 4))  # Crea figura.
        image = ax.imshow(cm, cmap="Blues")  # Dibuja mapa de calor.
        ax.set_xticks(range(3), ["Gana A", "Empate", "Gana B"])  # Etiquetas prediccion.
        ax.set_yticks(range(3), ["Gana A", "Empate", "Gana B"])  # Etiquetas reales.
        for i in range(3):  # Recorre filas.
            for j in range(3):  # Recorre columnas.
                ax.text(j, i, cm[i, j], ha="center", va="center")  # Escribe conteo.
        ax.set_title("Matriz de confusion - Mundial 2022")  # Titulo.
        ax.set_xlabel("Prediccion")  # Eje X.
        ax.set_ylabel("Real")  # Eje Y.
        fig.colorbar(image, ax=ax)  # Barra de color.
        plt.tight_layout()  # Ajusta grafico.
        plt.show()  # Muestra grafico.
        """
    ),
]


nb3 = [
    md(
        """
        # Notebook 03 - Dashboard Streamlit

        Este notebook genera el dashboard funcional en espanol. No usa Plotly ni Seaborn; solo Streamlit,
        Pandas, NumPy, Joblib y TensorFlow.
        """
    ),
    code(
        """
        from pathlib import Path  # Maneja rutas.
        import textwrap  # Limpia indentacion del codigo exportado.

        ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()  # Detecta raiz.
        APP_PATH = ROOT / "app_streamlit.py"  # Define ruta del dashboard.
        """
    ),
    code(
        "STREAMLIT_APP = "
        + repr(app_streamlit)
        + '  # Codigo completo del dashboard.\n'
        + 'APP_PATH.write_text(textwrap.dedent(STREAMLIT_APP).strip() + "\\n", encoding="utf-8")  # Escribe app Streamlit.\n'
        + 'print(f"Dashboard generado en: {APP_PATH}")  # Confirma ubicacion.\n'
    ),
    code(
        """
        requirements = [  # Dependencias reales usadas por el proyecto.
            "pandas",  # Tablas y datos.
            "numpy",  # Calculo numerico.
            "scikit-learn",  # Metricas, scaler y modelo del dashboard.
            "joblib",  # Persistencia de modelos.
            "matplotlib",  # Graficos de notebooks.
            "streamlit",  # Dashboard.
            "nbformat",  # Generacion de notebooks.
        ]  # Termina lista.
        (ROOT / "requirements.txt").write_text("\\n".join(requirements) + "\\n", encoding="utf-8")  # Guarda requirements.
        print("requirements.txt generado")  # Confirma archivo.
        """
    ),
    md(
        """
        Ejecutar el dashboard desde la raiz:

        ```powershell
        streamlit run app_streamlit.py --server.headless true --server.port 8501
        ```
        """
    ),
]


write_nb(NOTEBOOKS / "01_preparacion_datos_features.ipynb", nb1)
write_nb(NOTEBOOKS / "02_modelos_deep_learning.ipynb", nb2)
write_nb(NOTEBOOKS / "03_simulador_dashboard.ipynb", nb3)
APP_PATH.write_text(textwrap.dedent(app_streamlit).strip() + "\n", encoding="utf-8")
(ROOT / "requirements.txt").write_text("pandas\nnumpy\nscikit-learn\njoblib\nmatplotlib\nstreamlit\nnbformat\n", encoding="utf-8")
print("Proyecto generado en", ROOT)
