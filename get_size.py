import pandas as pd
import json, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file')

    args = parser.parse_args()

    with open(f"{args.input_file}", 'r') as f:
        json_file = json.load(f)
        print(len(json_file))

if __name__ == '__main__':
    main()