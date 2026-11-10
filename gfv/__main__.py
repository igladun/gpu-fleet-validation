"""Command line: `python3 -m gfv <command>` or `bash gfv-accept.sh <command>`."""

import argparse
import sys

from . import build_info


def main(argv=None):
    p = argparse.ArgumentParser(prog="gfv", description="GPU fleet validation scripts")
    sub = p.add_subparsers(dest="command")
    from .acceptance import run as acceptance
    acc = sub.add_parser("accept", help="L2 node acceptance gate: run on one node before it takes work")
    acceptance.add_arguments(acc)
    sub.add_parser("classes", help="list bundled node classes")
    sub.add_parser("version", help="print version")
    args = p.parse_args(argv)
    if args.command == "accept":
        return acceptance.accept(args)
    if args.command == "classes":
        from .common.manifest import bundled_classes
        for cls, (source, _) in sorted(bundled_classes().items()):
            print(f"{cls}\t{source}")
        return 0
    if args.command == "version":
        print(f"gfv {build_info()['label']}")
        return 0
    p.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
