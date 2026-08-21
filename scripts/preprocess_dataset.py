#!/usr/bin/env python3
"""
Preprocess the raw dataset (e.g. BIRD dev.json + dev_databases/) into the initial
snapshot that value_retrieval expects as its input. Run this once before the main
pipeline; every later stage builds on the snapshot this produces.

    python scripts/preprocess_dataset.py --config config/bird.toml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    a = ap.parse_args()
    os.environ['CONFIG_PATH'] = os.path.abspath(a.config)

    from dextersql.core.dataset import DatasetFactory, save_dataset
    from dextersql.core.logger import configure_logger, logger
    from dextersql.core.config import get_config

    app_config = get_config()
    configure_logger(app_config.logger_config.print_level)

    logger.info(f'Preprocessing dataset: {app_config.dataset_config.type} {app_config.dataset_config.split}')
    dataset = DatasetFactory.get_dataset(app_config.dataset_config)
    logger.info(f'Dataset loaded: {len(dataset)} items')
    save_dataset(dataset, app_config.dataset_config.save_path)
    logger.info(f'Dataset saved: {app_config.dataset_config.save_path}')


if __name__ == '__main__':
    main()
