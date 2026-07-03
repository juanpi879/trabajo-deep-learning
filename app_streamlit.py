from pathlib import Path
from html import escape
import itertools
import json
import time

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


def seed_bracket_by_top_champions(qualified, top10_teams):
    top_seeded = [team for team in top10_teams if team in qualified]
    remaining = [team for team in qualified if team not in top_seeded]
    size = len(qualified)
    slots = [None] * size
    preferred_slots = [
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
    for team, slot in zip(top_seeded, preferred_slots):
        slots[slot] = team
    fill_teams = iter(remaining)
    return [team if team is not None else next(fill_teams) for team in slots]


def champion_simulation_table(groups, simulations, progress_bar=None, status_box=None):
    counts = {}
    qualified, _ = qualified_teams(groups)
    rng = np.random.default_rng(42)
    started_at = time.perf_counter()
    update_every = max(1, simulations // 200)
    for sim_idx in range(simulations):
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
        if progress_bar is not None and ((sim_idx + 1) % update_every == 0 or sim_idx + 1 == simulations):
            progress = (sim_idx + 1) / simulations
            elapsed = time.perf_counter() - started_at
            eta_seconds = (elapsed / progress) - elapsed if progress > 0 else 0
            progress_bar.progress(progress)
            if status_box is not None:
                status_box.caption(
                    f"Simulaciones completadas: {sim_idx + 1:,} / {simulations:,} | "
                    f"Transcurrido: {format_seconds(elapsed)} | ETA: {format_seconds(eta_seconds)}"
                )
    table = pd.DataFrame({"seleccion": list(counts), "veces_campeon": list(counts.values())})
    table["simulaciones"] = simulations
    table["winrate_%"] = 100 * table["veces_campeon"] / simulations
    table = table.merge(champion_history, on="seleccion", how="left")
    table["mundiales_ganados"] = table["mundiales_ganados"].fillna(0).astype(int)
    return table.sort_values(["veces_campeon", "winrate_%"], ascending=False).reset_index(drop=True)


def format_seconds(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


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


def render_bracket(bracket, top10):
    round_order = ["Ronda de 32", "Octavos", "Cuartos", "Semifinal", "Final"]
    top10_lookup = {
        row["seleccion"]: {
            "rank": idx + 1,
            "winrate": row["winrate_%"],
            "wins": row["veces_campeon"],
        }
        for idx, row in top10.reset_index(drop=True).iterrows()
    }

    def team_slot(team, probability, winner):
        safe_team = escape(str(team))
        is_winner = team == winner
        top_info = top10_lookup.get(team)
        classes = "team-slot winner-slot" if is_winner else "team-slot"
        if top_info:
            classes += " top-team"
            rank_html = f"<span class='seed-badge'>Top #{top_info['rank']}</span>"
            champ_html = f"<span class='metric-chip'>Campeon: {top_info['winrate']:.1f}%</span>"
        else:
            rank_html = "<span class='seed-badge muted'>Sin top</span>"
            champ_html = "<span class='metric-chip muted'>Campeon: fuera top 10</span>"
        bar_width = max(4, probability * 100)
        return (
            f'<div class="{classes}">'
            f'<div class="team-main">{rank_html}<span class="team-name">{safe_team}</span></div>'
            f'<div class="team-side">{champ_html}<span class="metric-chip advance">Avanza: {probability:.0%}</span></div>'
            f'<div class="prob-track"><span style="width: {bar_width:.1f}%"></span></div>'
            "</div>"
        )

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
        .bracket-round {
            min-width: 260px;
        }
        .bracket-title {
            font-weight: 700;
            margin-bottom: 0.65rem;
            text-align: center;
            color: #0f172a;
            font-size: 0.95rem;
            letter-spacing: 0;
        }
        .match-card {
            border: 1px solid #cbd5e1;
            border-radius: 8px;
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
            min-height: 68px;
            padding: 0.48rem 0.55rem 0.7rem;
            border: 1px solid #e2e8f0;
            border-radius: 7px;
            background: #ffffff;
            color: #334155;
            font-size: 0.82rem;
        }
        .team-slot + .team-slot {
            margin-top: 0.35rem;
        }
        .winner-slot {
            border-color: #14b8a6;
            background: #ecfdf5;
            color: #0f172a;
        }
        .top-team {
            box-shadow: inset 4px 0 0 #2563eb;
        }
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
            background: #dbeafe;
            color: #1d4ed8;
            text-align: center;
            font-size: 0.68rem;
            font-weight: 800;
        }
        .seed-badge.muted {
            background: #e2e8f0;
            color: #64748b;
            font-weight: 700;
        }
        .metric-chip {
            border-radius: 999px;
            padding: 0.1rem 0.42rem;
            background: #ccfbf1;
            color: #0f766e;
            font-size: 0.68rem;
            font-weight: 800;
        }
        .metric-chip.advance {
            background: #e0f2fe;
            color: #0369a1;
        }
        .metric-chip.muted {
            background: #f1f5f9;
            color: #64748b;
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
            background: #2563eb;
        }
        .match-footer {
            display: flex;
            justify-content: space-between;
            gap: 0.4rem;
            color: #0f766e;
            margin-top: 0.42rem;
            font-size: 0.74rem;
            font-weight: 750;
            line-height: 1.1rem;
        }
        .round-2 .match-card {
            margin-top: 2.15rem;
            margin-bottom: 2.1rem;
        }
        .round-3 .match-card {
            margin-top: 5.3rem;
            margin-bottom: 5.2rem;
        }
        .round-4 .match-card {
            margin-top: 11.7rem;
            margin-bottom: 11.6rem;
        }
        .round-5 .match-card {
            margin-top: 24.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    board_parts = ["<div class='bracket-board'>"]
    for round_idx, round_name in enumerate(round_order, start=1):
        round_matches = bracket[bracket["ronda"] == round_name].reset_index(drop=True)
        board_parts.append(f"<div class='bracket-round round-{round_idx}'>")
        board_parts.append(f"<div class='bracket-title'>{escape(round_name)}</div>")
        for _, row in round_matches.iterrows():
            p_a_adv = row["p_a"] + 0.5 * row["p_draw"]
            p_b_adv = row["p_b"] + 0.5 * row["p_draw"]
            winner = escape(str(row["ganador"]))
            board_parts.append(
                "<div class='match-card'>"
                f"{team_slot(row['team_a'], p_a_adv, row['ganador'])}"
                f"{team_slot(row['team_b'], p_b_adv, row['ganador'])}"
                "<div class='match-footer'>"
                f"<span>Avanza: {winner}</span>"
                f"<span>Empate base: {row['p_draw']:.0%}</span>"
                "</div>"
                "</div>"
            )
        board_parts.append("</div>")
    board_parts.append("</div>")
    st.markdown("".join(board_parts), unsafe_allow_html=True)


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
    with st.spinner("Calculando probabilidades de grupos..."):
        qualified, tables = qualified_teams(groups_selected)
    tabs = st.tabs(list(groups_selected.keys()))
    for tab, group in zip(tabs, groups_selected.keys()):
        with tab:
            st.dataframe(tables[group], use_container_width=True, hide_index=True)

st.divider()
forced = st.selectbox("Forzar ganador si aparece en un cruce", ["Sin forzar"] + all_teams)
forced_team = None if forced == "Sin forzar" else forced

col1, col2 = st.columns([0.58, 0.42])
with col2:
    st.subheader("Top 10 campeones probables")
    sims = st.number_input("Numero de simulaciones", min_value=1, max_value=1_000_000, value=1_000, step=1_000)
    sim_progress = st.progress(0.0)
    sim_status = st.empty()
    with st.spinner("Ejecutando simulaciones de campeon..."):
        champion_results = champion_simulation_table(groups_selected, int(sims), sim_progress, sim_status)
    sim_progress.empty()
    sim_status.empty()
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

with col1:
    st.subheader("Cuadro de eliminacion")
    seeded_qualified = seed_bracket_by_top_champions(qualified, top10["seleccion"].tolist())
    with st.spinner("Armando bracket de eliminacion con el top 10..."):
        bracket, champion = build_bracket(seeded_qualified, forced_team)
    st.caption("El cuadro se siembra y resalta usando el top 10 calculado por simulacion.")
    render_bracket(bracket, top10)
    st.metric("Campeon proyectado del bracket", champion)
