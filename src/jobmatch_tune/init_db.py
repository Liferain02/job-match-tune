from __future__ import annotations

import argparse

from jobmatch_tune.database import init_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/jobmatch_tune.sqlite3")
    args = parser.parse_args()
    init_db(args.db)
    print(f"initialized database: {args.db}")


if __name__ == "__main__":
    main()
