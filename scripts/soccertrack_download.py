"""
soccertrack_download.py
========================
Descarga SOLO las anotaciones MOT de 1 partido de SoccerTrack v2
desde HuggingFace — sin descargar los vídeos.

Uso:
    python scripts/soccertrack_download.py --token TU_HF_TOKEN

Requisitos:
    pip install huggingface_hub
"""

import argparse
import sys
from pathlib import Path


def download_mot_annotations(token: str, output_dir: str = "c:/apped/data/datasets/soccertrack_v2"):
    """
    Descarga solo las anotaciones MOT (gt.txt + seqinfo.ini) de un partido.
    Excluye explícitamente todos los vídeos y archivos pesados.
    """
    try:
        from huggingface_hub import snapshot_download, list_repo_files
    except ImportError:
        print("[ERROR] huggingface_hub no instalado.")
        print("  Instala con: pip install huggingface_hub")
        sys.exit(1)

    repo_id = "atomscott/soccertrack-v2"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Conectando a HuggingFace con token...")
    print(f"[INFO] Dataset: {repo_id}")
    print(f"[INFO] Destino: {output_path}")
    print()

    # --- Listar archivos disponibles primero ---
    print("[INFO] Listando archivos del dataset (sin descargar)...")
    try:
        all_files = list(list_repo_files(repo_id, repo_type="dataset", token=token))
    except Exception as e:
        print(f"[ERROR] No se pudo acceder al dataset: {e}")
        print()
        print("Posibles causas:")
        print("  1. Token inválido o expirado")
        print("  2. No has aceptado las condiciones del dataset en:")
        print("     https://huggingface.co/datasets/atomscott/soccertrack-v2")
        print("  3. El dataset es privado/gated y necesitas solicitar acceso")
        sys.exit(1)

    print(f"[INFO] Total archivos en el dataset: {len(all_files)}")
    print()

    # Mostrar estructura
    mot_files = [f for f in all_files if "mot" in f.lower()]
    video_files = [f for f in all_files if f.endswith((".mp4", ".avi", ".mkv"))]
    gsr_files = [f for f in all_files if "gsr" in f.lower()]

    print(f"  Archivos MOT:   {len(mot_files)}")
    print(f"  Archivos GSR:   {len(gsr_files)}")
    print(f"  Vídeos (MP4):   {len(video_files)}")
    print()

    # Mostrar primeros partidos disponibles
    match_ids = set()
    for f in all_files:
        parts = Path(f).parts
        if parts:
            match_ids.add(parts[0])
    
    match_ids = sorted(match_ids)
    print(f"[INFO] Partidos disponibles ({len(match_ids)} total):")
    for m in match_ids[:5]:
        print(f"  - {m}")
    if len(match_ids) > 5:
        print(f"  ... y {len(match_ids) - 5} más")
    print()

    if not match_ids:
        print("[ERROR] No se encontraron partidos en el dataset.")
        sys.exit(1)

    # Seleccionar el primer partido
    first_match = match_ids[0]
    print(f"[INFO] Descargando anotaciones del partido: {first_match}")

    # Filtrar solo archivos MOT del primer partido (sin vídeos)
    files_to_download = [
        f for f in all_files
        if f.startswith(first_match) and not f.endswith((".mp4", ".avi", ".mkv"))
    ]

    print(f"[INFO] Archivos a descargar: {len(files_to_download)}")
    for f in files_to_download:
        print(f"  + {f}")
    print()

    # Descargar con allow_patterns
    allow_patterns = [f"{first_match}/mot/**", f"{first_match}/seqinfo.ini", f"{first_match}/*.ini"]

    print("[INFO] Iniciando descarga (solo MOT, sin vídeos)...")
    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            allow_patterns=[
                f"{first_match}/mot/**",
                f"{first_match}/**/*.ini",
                f"{first_match}/**/*.txt",
                f"{first_match}/**/*.json",
            ],
            ignore_patterns=["*.mp4", "*.avi", "*.mkv", "*.mov"],
            local_dir=str(output_path),
        )
        print(f"\n[OK] Descarga completada en: {local_path}")
        
        # Verificar que tenemos el gt.txt
        gt_files = list(Path(local_path).rglob("gt.txt"))
        if gt_files:
            print(f"\n[OK] gt.txt encontrado:")
            for gt in gt_files:
                size_kb = gt.stat().st_size / 1024
                print(f"   {gt} ({size_kb:.1f} KB)")
        else:
            print("\n[WARNING] No se encontró gt.txt — revisa los allow_patterns")

        return local_path, first_match

    except Exception as e:
        print(f"[ERROR] Error durante la descarga: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga anotaciones MOT de SoccerTrack v2")
    parser.add_argument("--token", required=True, help="Token de HuggingFace (hf_...)")
    parser.add_argument("--output", default="c:/apped/data/datasets/soccertrack_v2",
                        help="Directorio de salida")
    args = parser.parse_args()

    download_mot_annotations(args.token, args.output)
