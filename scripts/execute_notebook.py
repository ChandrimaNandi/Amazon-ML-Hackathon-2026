"""
Executes amazon-ml-hackathon.ipynb cell by cell in a persistent namespace,
capturing all genuine stdout, stderr, and displays, and saving the notebook
with full execution outputs.
"""

import sys
import io
import time
import json
import traceback
import contextlib
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_notebook(nb_path: Path):
    print(f"[RUNNER] Loading notebook from {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Shared execution namespace
    globs = {
        "__name__": "__main__",
        "__file__": str(nb_path),
    }

    # Custom display handler for pandas / objects
    def custom_display(*args):
        for a in args:
            if isinstance(a, pd.DataFrame):
                print(a.to_string(index=False))
            else:
                print(str(a))

    globs["display"] = custom_display

    exec_count = 1
    total_cells = len(nb["cells"])
    
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        code_text = "".join(cell["source"])
        print(f"\n[RUNNER] Executing Cell {idx+1}/{total_cells} (Exec count: {exec_count})...")
        first_line = code_text.strip().split("\n")[0][:70]
        print(f"  Code: {first_line}")

        cell["outputs"] = []
        cell["execution_count"] = exec_count

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        start_t = time.time()
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            try:
                exec(code_text, globs)
            except Exception as e:
                traceback.print_exc(file=stderr_buf)
                err_text = stderr_buf.getvalue()
                print(f"[ERROR in Cell {idx+1}]: {e}\n{err_text}")
                cell["outputs"].append({
                    "output_type": "error",
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": err_text.split("\n")
                })
                # Save partial notebook
                with open(nb_path, "w", encoding="utf-8") as f:
                    json.dump(nb, f, indent=1)
                sys.exit(1)

        elapsed = time.time() - start_t
        out_str = stdout_buf.getvalue()
        err_str = stderr_buf.getvalue()

        if out_str:
            lines = [l + "\n" for l in out_str.rstrip("\n").split("\n")]
            cell["outputs"].append({
                "output_type": "stream",
                "name": "stdout",
                "text": lines
            })
            print(f"  Output ({elapsed:.2f}s):\n" + "".join(lines[:6]) + ("  ..." if len(lines) > 6 else ""))
            
        if err_str:
            lines = [l + "\n" for l in err_str.rstrip("\n").split("\n")]
            cell["outputs"].append({
                "output_type": "stream",
                "name": "stderr",
                "text": lines
            })

        exec_count += 1

    # Save executed notebook
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"\n[RUNNER] Successfully executed all code cells. Saved to {nb_path}.")

    # Sync to runs/ and notebooks/
    runs_path = PROJECT_ROOT / "runs" / "amazon-ml-hackathon.ipynb"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(runs_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    notebk_path = PROJECT_ROOT / "notebooks" / "entity_resolution_experiments.ipynb"
    notebk_path.parent.mkdir(parents=True, exist_ok=True)
    with open(notebk_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"[RUNNER] Synced executed notebook to {runs_path} and {notebk_path}.")

if __name__ == "__main__":
    target = PROJECT_ROOT / "amazon-ml-hackathon.ipynb"
    run_notebook(target)
