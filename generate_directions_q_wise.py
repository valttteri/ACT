import argparse
import sys
import numpy as np
import pickle as pkl
from einops import rearrange
from utils import get_separated_activations
from checks import hf_home_is_correct

def main():
    if not hf_home_is_correct(): # HF_HOME must be set correctly
        return

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B')
    args = parser.parse_args()
    print('Running:\n{}\n'.format(' '.join(sys.argv)))
    print(args)

    # Load activations and labels
    head_wise_activations = pkl.load(open(f'ACT/activations/{args.model_name}_head_wise.pkl', 'rb'))
    labels = pkl.load(open(f'ACT/activations/{args.model_name}_labels.pkl', 'rb'))

    num_heads = 32
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)
    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations)
    
    # Generate directions for each question
    pairs = list(zip(separated_head_wise_activations, separated_labels))
    first_pair = pairs[0]
    print("Activation shape:", first_pair[0].shape)
    print("Label:", first_pair[1])

    head_wise_activation_directions = []
    total_acts = sum([len(p[0]) for p in pairs])
    total_labels = sum([len(p[1]) for p in pairs])
    print("total acts:", total_acts)
    print("total labels:", total_labels)
    #for i, pair in enumerate(pairs):
    #    act, label = pair[0], pair[1]
    #    if len(act) != len(label):
    #        print(f"Pair {i}/{len(pairs)} is wrong: {len(act)}, {len(label)}")
    #    print("num acts, labels:", len(act), len(label))
        #truthful = act[np.array(label) == 1].mean(axis=0)
        #untruthful = act[np.array(label) == 0].mean(axis=0) 

    #head_wise_activation_directions = np.array([a[np.array(l) == 1].mean(axis=0) - a[np.array(l) == 0].mean(axis=0) for a, l in zip(separated_head_wise_activations, separated_labels)])
    pkl.dump(head_wise_activation_directions, open(f'ACT/directions/{args.model_name}_directions.pkl', 'wb'))


if __name__ == '__main__':
    main()