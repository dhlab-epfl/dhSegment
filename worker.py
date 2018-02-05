from train import ex
import argparse
import glob
import os
import sys
import time
import json
import better_exceptions


if __name__ == '__main__':
    print('Starting worker')

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--configs-dir", required=True, help="Folder with configs file")
    ap.add_argument("-f", "--failed-configs-dir", required=True, help="Folder with failed experiments")
    args = ap.parse_args()

    while True:
        config_files = glob.glob(os.path.join(args.configs_dir, '**/*.json'), recursive=True)
        if len(config_files) == 0:
            time.sleep(3)
            continue

        # Found a config file
        config_file = config_files[0]
        print('Found config file : {}'.format(config_file))
        with open(config_file, 'r') as f:
            config = json.load(f)
        try:
            os.remove(config_file)
        except Exception:
            print('Some worker took MY JEB, #MAGA')
            continue

        print("Running config")
        #try:
        res = ex.run(config_updates=config)
        #except Exception as e:
        #    filename = os.path.relpath(config_file, args.configs_dir)
        #    print('Experiment {} failed : {}'.format(filename, e))
        #    output_file = os.path.join(args.failed_configs_dir, filename)
        #    os.makedirs(os.path.dirname(output_file), exist_ok=True)
        #    with open(output_file, 'w') as f:
        #        json.dump(config, f)
        print("Running Done")