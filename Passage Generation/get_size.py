import pandas as pd
import json, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file')

    args = parser.parse_args()

    with open(f"{args.input_file}", 'r') as f:
        json_file = json.load(f)
        count = 0
        for line in json_file:
            line = json_file[line]
            if(line['positive_ctxs'] == []):
                continue

            count += 1

        print(count)

if __name__ == '__main__':
    main()
