import sys
from pathlib import Path

# Configurar path para importar modulos de la raiz
root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.agents.reporting.reporting_agent import ReportingAgent

def main():
    events_file = root_path / "output" / "events.json"
    output_file = root_path / "output" / "match_report.txt"
    
    if not events_file.exists():
        print(f"Error: No se encontro el archivo de eventos en {events_file}")
        sys.exit(1)
        
    print(f"Iniciando ReportingAgent con {events_file}...")
    
    agent = ReportingAgent()
    report = agent.generate_report(str(events_file))
    
    print("\n[RESULTADO]")
    print(report)
    
    # Guardar en output/match_report.txt
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"\nInforme guardado exitosamente en: {output_file}")

if __name__ == "__main__":
    main()
