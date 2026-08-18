import argparse
import os

from omegaconf import OmegaConf


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_project_path(project_root, path_value):
    path_str = os.path.expanduser(str(path_value))
    if os.path.isabs(path_str):
        return os.path.normpath(path_str)
    return os.path.normpath(os.path.join(project_root, path_str))


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/wikimel.yaml')
    parser.add_argument('--test-only-ckpt', type=str, default='')
    args_ns, raw_overrides = parser.parse_known_args()

    overrides = []
    for raw in raw_overrides:
        item = raw[2:] if raw.startswith('--') else raw
        if '=' not in item or item.startswith('='):
            parser.error(f'invalid config override {raw!r}; expected key=value')
        overrides.append(item)

    args = OmegaConf.load(args_ns.config)
    if overrides:
        args = OmegaConf.merge(args, OmegaConf.from_dotlist(overrides))
    args.config_path = os.path.abspath(args_ns.config)
    args.project_root = get_project_root()
    if args_ns.test_only_ckpt:
        args.test_only_ckpt = resolve_project_path(args.project_root, args_ns.test_only_ckpt)
    else:
        args.test_only_ckpt = ''
    return args
