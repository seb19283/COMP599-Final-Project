import pandas as pd
import json, argparse, random
from tqdm import tqdm

def get_data(input_file):
    with open(f"{input_file}", 'r') as f:
        return json.load(f)

def get_passages(passages_file):
    df = pd.read_csv(passages_file, sep='\t')
    passages = {}

    for index, row in df.iterrows():
        passages[row['id']] = row

    return passages


def get_valid_questions(passages, training):
    updated_training = {}
    i = 0
    for line in tqdm(training):
        question_data = line
        filtered_data = {}

        filtered_data['dataset'] = question_data['dataset']
        filtered_data['question'] = question_data['question']
        filtered_data['answers'] = question_data['answers']

        positive = []
        for ctx in question_data['positive_ctxs']:
            if ctx['psg_id'] in passages:
                positive.append(ctx)

        if len(positive) < 5 and len(question_data['positive_ctxs']) != len(positive):
            continue

        negative = []
        for ctx in question_data['negative_ctxs']:
            if ctx['psg_id'] in passages:
                negative.append(ctx)

        if len(negative) < 5 and len(question_data['negative_ctxs']) != len(negative):
            continue

        hard_negative = []
        for ctx in question_data['hard_negative_ctxs']:
            if ctx['psg_id'] in passages:
                hard_negative.append(ctx)

        if len(hard_negative) < 5 and len(question_data['hard_negative_ctxs']) != len(hard_negative):
            continue

        filtered_data['positive_ctxs'] = positive
        filtered_data['negative_ctxs'] = negative
        filtered_data['hard_negative_ctxs'] = hard_negative
        updated_training[i] = filtered_data
        i += 1

    return updated_training

def dump_training_to_csv(training, output_file):
    with open(f"{output_file}", 'w') as o:
        json.dump(training, o, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file')
    parser.add_argument('-p', '--passages_file')
    parser.add_argument('-o', '--output_training_file')

    args = parser.parse_args()

    print("Importing training data.")
    json_input = get_data(args.input_file)

    print("Importing passages.")
    passages = get_passages(args.passages_file)
    print(len(passages))

    print("Getting valid questions.")
    valid_training = get_valid_questions(passages, json_input)
    print(len(valid_training))
    dump_training_to_csv(valid_training, args.output_training_file)

if __name__ == '__main__':
    main()