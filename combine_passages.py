import pandas as pd
import argparse

def open_file(file_input):
    return pd.read_csv(file_input, sep='\t')

def combine_passages(df1, df2):
    combined = {}

    for index, row in df1.iterrows():
        combined[row['id']] = row

    for index, row in df2.iterrows():
        combined[row['id']] = row

    return combined

def dump_passages(passages, output_file):
    passages_array = []

    for index in passages:
        passages_array.append(passages[index])

    df = pd.DataFrame(passages_array, columns=['id', 'text', 'title'])
    df.to_csv(f"{output_file}", sep='\t', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_file')
    parser.add_argument('-d', '--dev_file')
    parser.add_argument('-o', '--output_file')

    args = parser.parse_args()

    train_passages = open_file(args.train_file)
    print(len(train_passages))
    dev_passages = open_file(args.dev_file)
    print(len(dev_passages))

    combined_passages = combine_passages(train_passages, dev_passages)
    print(len(combined_passages))

    dump_passages(combined_passages, args.output_file)

if __name__ == '__main__':
    main()