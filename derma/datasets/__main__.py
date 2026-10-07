"""Command line: ``python -m derma.datasets {status,prepare} ...``."""

import argparse

from . import DATASETS


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m derma.datasets',
                                     description='Check and prepare the public EDA datasets used by DERMA.')
    parser.add_argument('command', choices=['status', 'prepare'])
    parser.add_argument('datasets', nargs='*', help=f'dataset names (default: all): {", ".join(DATASETS)}')
    parser.add_argument('--data-dir', help='data folder (default: DERMA_DATA_DIR)')
    parser.add_argument('--ignore-checksum', action='store_true',
                        help='prepare even if the archive SHA-256 does not match')
    args = parser.parse_args(argv)

    names = [n.upper() for n in args.datasets] or list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        parser.error(f'unknown dataset(s): {", ".join(unknown)}')

    for name in names:
        if args.command == 'status':
            DATASETS[name].status(args.data_dir)
        else:
            print(f'Preparing {name}...')
            paths = DATASETS[name].prepare(args.data_dir, ignore_checksum=args.ignore_checksum)
            print(f'{name}: {len(paths)} recordings prepared')


if __name__ == '__main__':
    main()
