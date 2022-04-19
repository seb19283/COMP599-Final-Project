import pandas as pd
import json, argparse, random
from tqdm import tqdm

def get_train_data(input_file):
    with open(f"{input_file}", 'r') as f:
        return json.load(f)

def get_random_sample(json_file, num_samples):
    num_lines = len(json_file)
    samples = num_samples
    lines_to_choose = random.sample(range(0, num_lines), min(num_lines, samples))
    return lines_to_choose

def get_passages(json_file, lines_to_choose):
    passages = {}
    for (i, line) in enumerate(tqdm(lines_to_choose)):
        question_data = json_file[line]

        positive = []
        negative = []
        for ctx in question_data['positive_ctxs']:
            # Change to passage_id for NQ dataset
            information = [ctx['psg_id'], ctx['title'], ctx['text']]
            positive.append(information)
            if len(positive) == 10:
                break
        for p in positive:
            passages[p[0]] = p

        for ctx in question_data['negative_ctxs']:
            information = [ctx['psg_id'], ctx['title'], ctx['text']]
            if not ctx['psg_id'] in passages:
                negative.append(information)
            if len(negative) == 5:
                break

        for n in negative:
            passages[n[0]] = n

        negative = []
        for ctx in question_data['hard_negative_ctxs']:
            information = [ctx['psg_id'], ctx['title'], ctx['text']]
            if not ctx['psg_id'] in passages:
                negative.append(information)
            if len(negative) == 5:
                break

        for n in negative:
            passages[n[0]] = n

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
            # Change to passage_id for NQ dataset
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

def dump_passages_to_csv(passages, output_file):
    passages_array = []
    for indx in passages:
        passages_array.append(passages[indx])

    df = pd.DataFrame(passages_array, columns=['id', 'text', 'title'])
    df.to_csv(f"{output_file}", sep='\t', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file')
    parser.add_argument('-c', '--num_lines_to_choose', type=int)
    parser.add_argument('-op', '--output_passages_file')
    parser.add_argument('-ot', '--output_training_file')

    args = parser.parse_args()

    print("Importing training data.")
    json_input = get_train_data(args.input_file)

    print("Getting lines to choose.")
    lines_to_choose = get_random_sample(json_input, args.num_lines_to_choose)

    print("Getting training data.")
    passages = get_passages(json_input, lines_to_choose)
    print(len(passages))

    print("Getting valid questions.")
    valid_training = get_valid_questions(passages, json_input)
    print(len(valid_training))
    dump_training_to_csv(valid_training, args.output_training_file)
    dump_passages_to_csv(passages, args.output_passages_file)

if __name__ == '__main__':
    main()