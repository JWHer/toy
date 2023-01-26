from archive.common import Archive

import argparse
import sys
import yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-file', type=str)
    parser.add_argument('--archive-config-file', type=str)
    parsed, _ = parser.parse_known_args(args=sys.argv[1:])

    with open(parsed.archive_config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    archive = Archive(config, filename=parsed.archive_file)
    archive.compose()
    archive.package()
