import os
import time
import pandas as pd
import yaml
try:
    from tqdm import tqdm
except ImportError:
    print("Por favor instala tqdm: pip install tqdm")
    exit(1)

from pathlib import Path

def get_latest_run(base_dir="ml/training/runs"):
    if not os.path.exists(base_dir): 
        return None
    
    all_runs = []
    # Buscar en detect y segment
    for sub in ["detect", "segment"]:
        d = os.path.join(base_dir, sub)
        if os.path.exists(d):
            runs = [os.path.join(d, o) for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d, o))]
            all_runs.extend(runs)
            
    if not all_runs: 
        return None
        
    return max(all_runs, key=os.path.getmtime)

def monitor_training():
    run_dir = get_latest_run()
    if not run_dir:
        print("No se encontraron entrenamientos.")
        return
        
    args_file = os.path.join(run_dir, "args.yaml")
    results_file = os.path.join(run_dir, "results.csv")
    
    if not os.path.exists(args_file): return
    
    with open(args_file, 'r') as f:
        args = yaml.safe_load(f)
        total_epochs = args.get('epochs', 50)
        
    print(f"\n[+] Monitoreando entrenamiento en: {run_dir}")
    print("[+] Presiona Ctrl+C para salir.\n")
    
    # Check initial progress
    last_epoch = 0
    if os.path.exists(results_file):
        try:
            df = pd.read_csv(results_file)
            if not df.empty:
                df.columns = df.columns.str.strip()
                last_epoch = int(df['epoch'].max())
        except Exception:
            pass

    with tqdm(total=total_epochs, initial=last_epoch, desc="Progreso del Modelo", unit="epoch") as pbar:
        while last_epoch < total_epochs:
            if os.path.exists(results_file):
                try:
                    df = pd.read_csv(results_file)
                    if not df.empty:
                        df.columns = df.columns.str.strip()
                        current_epoch = int(df['epoch'].max())
                        if current_epoch > last_epoch:
                            pbar.update(current_epoch - last_epoch)
                            last_epoch = current_epoch
                except Exception:
                    pass
            
            if last_epoch >= total_epochs:
                break
                
            time.sleep(10)
            
    print("\n¡Entrenamiento completado!")

if __name__ == "__main__":
    monitor_training()
