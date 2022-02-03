import pandas as pd
import random

pair2gen_1 = {}
pair2gen_2 = {}
pair2gen_3 = {}
pair2gen_4 = {}
pair2gen_4_b1 = {}
pair2gen_5 = {}
pair2gen_5_b1 = {}
pair2tar = {}
with open('checkpoint/baseline1_test/final_output.tsv') as f_in:
    for line in f_in:
        pair, gen, tar = line.strip().split('\t')
        pair2tar[pair] = tar
        pair2gen_1[pair] = gen
        
with open('checkpoint/baseline2_test/final_output.tsv') as f_in:
    for line in f_in:
        pair, gen, tar = line.strip().split('\t')
        pair2gen_2[pair] = gen
        
with open('checkpoint/baseline3_test/final_output.tsv') as f_in:
    for line in f_in:
        pair, gen, tar = line.strip().split('\t')
        pair2gen_3[pair] = gen
        
with open('checkpoint/baseline4_test/final_output.tsv') as f_in:
    for line in f_in:
        pair, gen, tar = line.strip().split('\t')
        pair2gen_4[pair] = gen
        
with open('checkpoint/baseline4_b1_test/final_output.tsv') as f_in:
    for line in f_in:
        pair, gen, tar = line.strip().split('\t')
        pair2gen_4_b1[pair] = gen
        
with open('checkpoint/baseline5_test/final_output.tsv') as f_in:
    for line in f_in:
        pair, gen, tar = line.strip().split('\t')
        pair2gen_5[pair] = gen
        
with open('checkpoint/baseline5_b1_test/final_output.tsv') as f_in:
    for line in f_in:
        pair, gen, tar = line.strip().split('\t')
        pair2gen_5_b1[pair] = gen
        

pairs = list(pair2tar.keys())
random.shuffle(pairs)
data = [{'pair' : pair,
         'target' : pair2tar[pair],
         'target_score' : 0,
         'baseline1' : pair2gen_1[pair],
         'baseline1_score' : 0,
         'baseline2' : pair2gen_2[pair],
         'baseline2_score' : 0,
         'baseline3' : pair2gen_3[pair],
         'baseline3_score' : 0,
         'baseline4' : pair2gen_4[pair],
         'baseline4_score' : 0,
         'baseline4_b1' : pair2gen_4_b1[pair],
         'baseline4_b1_score' : 0,
         'baseline5' : pair2gen_5[pair],
         'baseline5_score' : 0,
         'baseline5_b1' : pair2gen_5_b1[pair],
         'baseline5_b1_score' : 0} for pair in pairs[:100]]
col = [
    'pair',
    'target',
    'target_score',
    'baseline1',
    'baseline1_score',
    'baseline2',
    'baseline2_score',
    'baseline3',
    'baseline3_score',
    'baseline4',
    'baseline4_score',
    'baseline4_b1',
    'baseline4_b1_score',
    'baseline5',
    'baseline5_score',
    'baseline5_b1',
    'baseline5_b1_score'
]
data = pd.DataFrame(data)
data.to_csv('human_evaluation.tsv', sep='\t', index=False, columns=col)
