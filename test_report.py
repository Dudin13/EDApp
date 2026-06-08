import os
from dotenv import load_dotenv

# Asegurar que se cargan las variables del .env
load_dotenv()

from src.graph.match_graph import build_match_graph

if __name__ == "__main__":
    print("Inicializando grafo...")
    graph = build_match_graph()
    
    # Estado inicial: asume que nuestro events_file ya está generado
    initial_state = {
        "events_file": "output/events.json"
    }
    
    print("\nEjecutando grafo sobre output/events.json...")
    result = graph.invoke(initial_state)
    
    print("\n--- EJECUCIÓN TERMINADA ---")
