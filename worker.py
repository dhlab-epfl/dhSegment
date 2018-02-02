from train import ex
import argparse
import glob
import os
import sys
import time
import json


if __name__ == '__main__':
    print('Starting worker')

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--configs-dir", required=True, help="Folder with configs file")
    #ap.add_argument("-f", "--failed-configs-dir", required=True, help="Folder with failed experiments")
    args = ap.parse_args()

    while True:
        config_files = glob.glob(os.path.join(args.configs_dir, '*.json'))
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
        try:
            res = ex.run(config_updates=config)
        except Exception:
            print('Experiment failed')
        #TODO check experiment result and if failed save config file to failed_configs_dir
        print("Running Done")