"""
Network visualization utility for the ABM organization structure.

What this file does:
1. Extracts team-level and agent-level connectivity from a model.
2. Lays out teams as non-overlapping circles and places agents inside each team.
3. Draws team links, intra-team agent links, and optional cross-team links.
4. Produces publication-ready network figures saved to `figures/`.

How it is used in the pipeline:
- After ABM runs are created, this script is used to generate representative
  structural snapshots of the organization network (Step 3 in README).
- These figures support interpretation of the simulation setup in the IDETC paper.
"""

import math
import random
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def _unique_undirected_edges(pairs):
    edges = set()
    for a, b in pairs:
        if a == b:
            continue
        x, y = (a, b) if hash(a) <= hash(b) else (b, a)
        edges.add((x, y))
    return list(edges)

def _team_edges_from_model(model, teams):
    # If your org already has explicit team connectivity
    pairs = []
    for t in teams:
        for u in model.organization.teams_connected_to(t):
            if u in teams:
                pairs.append((t, u))
    return _unique_undirected_edges(pairs)

def _agent_edges_all(model):
    # Full agent graph from your API
    pairs = []
    all_agents = list(model.organization.all_agent_ids)  # if you have this
    aset = set(all_agents)
    for a in all_agents:
        for b in model.organization.agents_connected_to(a):
            if b in aset:
                pairs.append((a, b))
    return _unique_undirected_edges(pairs), all_agents

def _agent_edges_within_team(model, agents):
    aset = set(agents)
    pairs = []
    for a in agents:
        for b in model.organization.agents_connected_to(a):
            if b in aset:
                pairs.append((a, b))
    return _unique_undirected_edges(pairs)

def _points_in_circle(n, radius, rng):
    pts = []
    for _ in range(n):
        r = radius * math.sqrt(rng.random())
        theta = 2 * math.pi * rng.random()
        pts.append((r * math.cos(theta), r * math.sin(theta)))
    return pts

def _separate_circles(pos, radius, padding=0.25, iters=400, step=0.35):
    """
    Push circles apart until no overlap.
    pos: dict[node] -> (x,y)
    radius: dict[node] -> float
    """
    nodes = list(pos.keys())
    for _ in range(iters):
        moved = False
        for i, j in itertools.combinations(nodes, 2):
            xi, yi = pos[i]
            xj, yj = pos[j]
            dx, dy = xj - xi, yj - yi
            dist = math.hypot(dx, dy)

            min_dist = radius[i] + radius[j] + padding
            if dist == 0:
                # jitter to break symmetry
                dx, dy = 1e-6, 0.0
                dist = 1e-6

            if dist < min_dist:
                moved = True
                # push away along the line connecting centers
                ux, uy = dx / dist, dy / dist
                overlap = (min_dist - dist)

                # split motion between both nodes
                shift = 0.5 * overlap * step
                pos[i] = (xi - ux * shift, yi - uy * shift)
                pos[j] = (xj + ux * shift, yj + uy * shift)
        if not moved:
            break
    return pos

def _clip_segment_to_circles(p1, p2, r1, r2, gap=0.03):
    """
    Move endpoints of segment p1->p2 onto the circle boundaries.
    gap is an extra offset beyond the boundary (in same units as coords).
    """
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    d = math.hypot(dx, dy)
    if d == 0:
        return p1, p2

    ux, uy = dx / d, dy / d

    # start at boundary of circle1, end at boundary of circle2
    s = (x1 + ux * (r1 + gap), y1 + uy * (r1 + gap))
    e = (x2 - ux * (r2 + gap), y2 - uy * (r2 + gap))
    return s, e

def plot_org_structure(
    model,
    team_radius=1.0,
    team_padding=0.35,
    team_layout_k=4.0,
    team_scale=4.0,
    seed=7,
    draw_cross_team_agent_edges=False,
    figsize=(11, 9),
):
    rng = random.Random(seed)

    teams = list(model.organization.all_team_ids)

    # --- TEAM GRAPH ---
    team_edges = _team_edges_from_model(model, teams)

    # If team_edges is empty, your teams_connected_to() might not be populated.
    # In that case, we can derive team edges from agent edges across teams.
    if len(team_edges) == 0:
        # derive team edges from agent edges
        try:
            agent_edges, all_agents = _agent_edges_all(model)
            team_of = {}
            for t in teams:
                for a in model.organization.agents_from_team(t):
                    team_of[a] = t
            derived_pairs = []
            for a, b in agent_edges:
                ta, tb = team_of.get(a), team_of.get(b)
                if ta is not None and tb is not None and ta != tb:
                    derived_pairs.append((ta, tb))
            team_edges = _unique_undirected_edges(derived_pairs)
        except Exception:
            pass

    Gt = nx.Graph()
    Gt.add_nodes_from(teams)
    Gt.add_edges_from(team_edges)

    # Layout with larger k -> more spacing
    pos_team = nx.spring_layout(Gt, seed=seed, k=team_layout_k)

    # Layout with larger k -> more spacing
    pos_team = nx.spring_layout(Gt, seed=seed, k=team_layout_k)

    # scale up/down (THIS is your main distance knob)
    scale = team_scale * team_radius
    for t in pos_team:
        pos_team[t] = (pos_team[t][0] * scale, pos_team[t][1] * scale)

    # --- FORCE NON-OVERLAP OF TEAM CIRCLES ---
    team_r = {t: team_radius for t in teams}
    pos_team = _separate_circles(pos_team, team_r, padding=team_padding, iters=600)

    # --- AGENTS INSIDE TEAMS ---
    pos_agent = {}
    agents_by_team = {}
    intra_edges_by_team = {}

    for team in teams:
        agents = list(model.organization.agents_from_team(team))
        agents_by_team[team] = agents
        intra_edges_by_team[team] = _agent_edges_within_team(model, agents)

        inner_r = team_radius * 0.78
        offsets = _points_in_circle(len(agents), inner_r, rng)
        cx, cy = pos_team[team]
        for a, (dx, dy) in zip(agents, offsets):
            pos_agent[a] = (cx + dx, cy + dy)

    # Optional cross-team agent edges (if you want)
    cross_edges = []
    if draw_cross_team_agent_edges:
        agent_to_team = {}
        for t in teams:
            for a in agents_by_team[t]:
                agent_to_team[a] = t
        for t in teams:
            for a in agents_by_team[t]:
                for b in model.organization.agents_connected_to(a):
                    if b in agent_to_team and agent_to_team[b] != t:
                        cross_edges.append((a, b))
        cross_edges = _unique_undirected_edges(cross_edges)

    # --- DRAW ---
    fig, ax = plt.subplots(figsize=figsize)

    # Team-team edges (thick dashed black, on top)
    # --- Team-team edges (subtle, background layer) ---
    TEAM_EDGE_LW = 1        # thin
    TEAM_EDGE_ALPHA = 0.75    # very transparent
    TEAM_EDGE_COLOR = "#494949"
    TEAM_EDGE_Z = 0           # behind everything

    edge_gap = 0.05 * team_radius  # keep your boundary clipping

    for (a, b) in Gt.edges():
        p1 = pos_team[a]
        p2 = pos_team[b]
        s, e = _clip_segment_to_circles(
            p1, p2,
            team_radius,
            team_radius,
            gap=edge_gap
        )
        ax.plot(
            [s[0], e[0]],
            [s[1], e[1]],
            linestyle="--",
            linewidth=TEAM_EDGE_LW,
            alpha=TEAM_EDGE_ALPHA,
            color=TEAM_EDGE_COLOR,
            zorder=TEAM_EDGE_Z
        )

    # Team circles (dashed outline)
    for team in teams:
        cx, cy = pos_team[team]
        ax.add_patch(Circle((cx, cy), team_radius,
                            fill=False, linestyle="--",
                            linewidth=2.5, color="black",
                            alpha=0.9, zorder=6))

    # Intra-team agent edges
    AGENT_EDGE_COLOR = "#14368D"
    AGENT_EDGE_ALPHA = 0.65
    AGENT_EDGE_LW = 1
    for team in teams:
        for (u, v) in intra_edges_by_team[team]:
            x1, y1 = pos_agent[u]
            x2, y2 = pos_agent[v]
            ax.plot(
                [x1, x2], [y1, y2],
                linewidth=AGENT_EDGE_LW,
                alpha=AGENT_EDGE_ALPHA,
                color=AGENT_EDGE_COLOR,
                zorder=2
            )

    # Cross-team agent edges (optional)
    for (u, v) in cross_edges:
        x1, y1 = pos_agent[u]
        x2, y2 = pos_agent[v]
        ax.plot([x1, x2], [y1, y2],
                linewidth=0.8, alpha=0.25,
                linestyle="--", color="gray", zorder=1)

    # Agent nodes
    NODE_COLOR = "#1B1E72"
    xs = [pos_agent[a][0] for t in teams for a in agents_by_team[t]]
    ys = [pos_agent[a][1] for t in teams for a in agents_by_team[t]]
    ax.scatter(xs, ys,
        s=35,
        zorder=7,
        color=NODE_COLOR,
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    from utils import get_model, BASE_DIR

    configs = [
        ("2_12_6_3_2", "network_example_1"),
    ]

    for model_name, filename in configs:
        model = get_model(model_name)

        fig, ax = plot_org_structure(
            model,
            team_padding=1,
            team_scale=1
        )

        save_path = BASE_DIR / "figures" / f"{filename}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {save_path}")
