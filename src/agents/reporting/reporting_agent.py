import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


class ReportingAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )

        self.system_prompt = """You are an expert football match analyst writing a post-match report.

FORMAT YOUR REPORT EXACTLY LIKE THIS:

## Match Overview
One short paragraph (2-3 sentences) summarizing the overall story of the match.

## Key Statistics
| Stat | Team A | Team B |
|------|--------|--------|
| Possession | X% | Y% |
| Passes | X | Y |
| Tackles Won | X | Y |
| Interceptions | X | Y |
| Turnovers Lost | X | Y |
| Total Distance (km) | X | Y |
| Max Speed (km/h) | X | Y |
| Total Sprints | X | Y |

## Tactical Analysis
One paragraph (3-4 sentences) analyzing what the numbers reveal about each team's approach — pressing, dominance, defensive solidity.

## Physical Analysis
One paragraph (2-3 sentences) analyzing the physical output — which team worked harder, who covered the most ground, sprint patterns.

## Standout Performers
Bullet list of 3-5 players with a one-line note on what they did well. Include physical stats where relevant.

## Data Notes
One sentence acknowledging any data limitations.

RULES:
- Be concise. Total report should be under 300 words.
- Write like a Sky Sports or BBC Sport pundit — sharp, insightful, no filler.
- Focus on PATTERNS, not just numbers.
- Do NOT repeat raw numbers the reader can see in the table."""

    def generate_report(self, events_file_path):
        """
        Generate a match report from the events.json file.
        """
        stats_text = self._build_stats_text(events_file_path)

        print("[ReportingAgent] Sending stats to Groq LLM...")

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Here are the match statistics:\n\n{stats_text}\n\nWrite a post-match analysis report."),
        ]

        response = self.llm.invoke(messages)
        report = response.content

        print("\n" + "=" * 50)
        print("MATCH REPORT")
        print("=" * 50)
        print(report)

        return report

    def _build_stats_text(self, events_file_path):
        """Convert events.json data into a text summary for the LLM."""
        with open(events_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        events = data.get("events", [])
        physical_stats = data.get("physical_stats", {})
        
        team_stats = {
            "Equipo A": {"possession": 0, "passes": 0, "turnovers_lost": 0, "turnovers_won": 0, "tackles": 0, "interceptions": 0, "dist": 0.0, "sprints": 0, "top_speed": 0.0},
            "Equipo B": {"possession": 0, "passes": 0, "turnovers_lost": 0, "turnovers_won": 0, "tackles": 0, "interceptions": 0, "dist": 0.0, "sprints": 0, "top_speed": 0.0}
        }
        
        player_stats = {}
        track_to_team = {}

        total_poss = 0
        
        for ev in events:
            team = ev.get("nombre_equipo")
            if team not in team_stats:
                continue
                
            action = str(ev.get("action", "")).lower()
            player = ev.get("nombre_jugador", "Desconocido")
            tid = str(ev.get("track_id"))
            
            if tid != "None":
                track_to_team[tid] = (team, player)
            
            if player not in player_stats:
                player_stats[player] = {"team": team, "passes": 0, "possession_events": 0, "distance": 0.0, "speed": 0.0, "sprints": 0}
                
            team_stats[team]["possession"] += 1
            total_poss += 1
            player_stats[player]["possession_events"] += 1
            
            if "pase" in action:
                team_stats[team]["passes"] += 1
                player_stats[player]["passes"] += 1
            elif "perdida" in action or "turnover" in action:
                team_stats[team]["turnovers_lost"] += 1
            elif "recuperacion" in action or "robo" in action or "tackle" in action:
                team_stats[team]["tackles"] += 1
            elif "intercepcion" in action:
                team_stats[team]["interceptions"] += 1

        for tid_str, p_stats in physical_stats.items():
            if tid_str in track_to_team:
                team, player = track_to_team[tid_str]
                dist = p_stats.get("distance_km", 0.0)
                sprints = p_stats.get("sprint_count", 0)
                speed = p_stats.get("top_speed_kmh", 0.0)
                
                team_stats[team]["dist"] += dist
                team_stats[team]["sprints"] += sprints
                team_stats[team]["top_speed"] = max(team_stats[team]["top_speed"], speed)
                
                if player in player_stats:
                    player_stats[player]["distance"] += dist
                    player_stats[player]["sprints"] += sprints
                    player_stats[player]["speed"] = max(player_stats[player]["speed"], speed)

        lines = []
        for team_name in ["Equipo A", "Equipo B"]:
            ts = team_stats[team_name]
            poss_pct = (ts["possession"] / total_poss * 100) if total_poss > 0 else 0

            lines.append(f"{team_name}:")
            lines.append(f"  Possession: {poss_pct:.1f}%")
            lines.append(f"  Passes completed: {ts['passes']}")
            lines.append(f"  Turnovers lost: {ts['turnovers_lost']}")
            lines.append(f"  Turnovers won: {ts['turnovers_won']}")
            lines.append(f"  Tackles won: {ts['tackles']}")
            lines.append(f"  Interceptions: {ts['interceptions']}")
            lines.append(f"  Total distance covered (km): {ts['dist']:.2f}")
            lines.append(f"  Max speed (km/h): {ts['top_speed']:.1f}")
            lines.append(f"  Total sprints: {ts['sprints']}")
            lines.append("")

        lines.append("Top Players (by involvement):")
        sorted_players = sorted(
            player_stats.items(),
            key=lambda x: x[1]["possession_events"],
            reverse=True,
        )

        for p_name, p_data in sorted_players[:10]:
            if p_name != "Desconocido TNone" and "Desconocido" not in p_name:
                lines.append(
                    f"  {p_name} ({p_data['team']}): "
                    f"{p_data['passes']} passes, "
                    f"{p_data['possession_events']} events on ball, "
                    f"{p_data['distance']:.2f} km, "
                    f"{p_data['speed']:.1f} km/h max, "
                    f"{p_data['sprints']} sprints"
                )

        lines.append("")
        lines.append("NOTE: This data comes from EDApp's event detection system (events.json).")

        return "\n".join(lines)
