import argparse
import os
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(variable, "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mbc_pfs_tables_plots")

from paper_tables import TABLES, Context, load, render


def main():
    parser = argparse.ArgumentParser(
        description="Print the manuscript tables (t1, t2) and Extended Data Tables (ed1-ed28) from saved outputs. "
                    "Run from the repository root after the commands in commands.txt.")
    parser.add_argument("tables", nargs="*", metavar="TABLE", help="Table keys to print, for example t2 ed15. Default: all.")
    parser.add_argument("--list", action="store_true", help="List the table keys and exit.")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    keys = args.tables or list(TABLES)
    unknown = [key for key in keys if key not in TABLES]
    if unknown:
        parser.error(f"unknown table keys: {', '.join(unknown)}; choose from {', '.join(TABLES)}")
    if args.list:
        for key in keys:
            with redirect_stdout(sys.stderr):
                module = load(key)
            print(f"{key:5} {module.NUMBER:24} {module.LABEL}")
        return 0
    ctx, failed = Context(), []
    for key in keys:
        print(f"Build {key}", file=sys.stderr, flush=True)
        try:
            with redirect_stdout(sys.stderr):
                module = load(key)
                panels = module.build(ctx)
        except Exception:
            traceback.print_exc()
            failed.append(key)
            continue
        print(render(module, panels), flush=True)
    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
