import subprocess
import sys

def run(cmd):
    print("\n>>>", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

def main():
    # Usa lo stesso interprete del venv attivo
    py = sys.executable

    run([py, "-m", "src.train_ae"])
    run([py, "-m", "src.train_tcn"])
    run([py, "-m", "src.infer_submit"])   # default split="test" nel tuo script

    print("\nDONE. Controlla artifacts/ per i modelli e la submission.")

if __name__ == "__main__":
    main()
