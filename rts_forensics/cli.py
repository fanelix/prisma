import argparse

from .config import load_config
from .pipeline import run
from .report import write_results


def main():
    parser = argparse.ArgumentParser(description="Exploratory raw RTS observation forensics")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("run")
    command.add_argument("data", nargs="+")
    command.add_argument("--config")
    command.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        result = run(args.data, load_config(args.config))
        write_results(result, args.out)
    except (ValueError, OSError) as error:
        parser.exit(2, str(error) + "\n")
    print(f"{len(result['prism_summary'])} prism/station series exported to {args.out}")
    print("Exploratory; frame corrections experimental; inspect report.html and audit tables.")


if __name__ == "__main__":
    main()
