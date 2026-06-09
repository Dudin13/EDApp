import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_script(script_name, args_list=None):
    if args_list is None:
        args_list = []
    
    python_exe = sys.executable
    
    # Buscamos el script en varias ubicaciones probables
    candidates = [
        ROOT / "scripts" / script_name,
        ROOT / "app" / "scripts" / script_name,
        ROOT / script_name
    ]
    
    target_path = None
    for cand in candidates:
        if cand.exists():
            target_path = cand
            break
            
    if not target_path:
        print(f"Error: Script '{script_name}' no encontrado.")
        sys.exit(1)
        
    cmd = [python_exe, str(target_path)] + args_list
    print(f"Ejecutando: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"El comando falló con código de salida {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="EDApp Unified CLI - Wrapper para scripts del pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analizar un video")
    analyze_parser.add_argument("video", help="Ruta al video")
    analyze_parser.add_argument("--deep", action="store_true", help="Ejecutar el pipeline completo")

    # report
    report_parser = subparsers.add_parser("report", help="Generar reporte de analisis")

    # clips
    clips_parser = subparsers.add_parser("clips", help="Generar clips de las acciones")

    # train
    train_parser = subparsers.add_parser("train", help="Entrenar el modelo de deteccion")
    train_parser.add_argument("--version", required=True, help="Versión de entrenamiento, ej: v7")

    args = parser.parse_args()

    if args.command == "analyze":
        if args.deep:
            run_script("run_full_pipeline.py", [args.video])
        else:
            run_script("run_demo_analysis.py", [args.video])

    elif args.command == "report":
        run_script("generate_report.py")

    elif args.command == "clips":
        run_script("clip_maker.py")

    elif args.command == "train":
        bat_file = ROOT / f"Train_{args.version}.bat"
        if not bat_file.exists():
            print(f"Error: '{bat_file.name}' no encontrado en la raíz.")
            sys.exit(1)
        
        cmd = [str(bat_file)]
        print(f"Ejecutando script de entrenamiento: {bat_file.name}")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"El comando falló con código de salida {e.returncode}")
            sys.exit(e.returncode)
        except KeyboardInterrupt:
            print("\nProceso interrumpido por el usuario.")
            sys.exit(1)

if __name__ == "__main__":
    main()
