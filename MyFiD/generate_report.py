import pandas as pd
posfix_list = ['1', '2', '3', '3_large', '4', '4_b1', '4_large', '5', '5_b1', '5_large']
files = ['baseline%s.log' % posfix for posfix in posfix_list]

data = []
for file in files:
    with open(file) as f_out:
        metric = ''
        rouge_l = False
        bleu_score, rouge_score, meteor_score, bert_score = 0, 0, 0, 0
        for line in f_out:
            line_split = line.split()
            if not line_split:
                continue
            if line_split[0] == 'BLEU':
                metric = 'BLEU'
            elif line_split[0] == 'ROUGE':
                metric = 'ROUGE'
            elif line_split[0] == 'METEOR':
                metric = 'METEOR'
            elif line_split[0] == 'BERTSCORE':
                metric = 'BERTSCORE'
            else:
                if metric == 'BLEU' and line_split[0] == '"score":':
                    bleu_score = float(line_split[1].strip(','))
                elif metric == 'ROUGE' and line_split[0] == '"rouge-l":':
                    rouge_l = True
                elif metric == 'ROUGE' and line_split[0] == '"f":' and rouge_l:
                    rouge_score = float(line_split[1])
                elif metric == 'METEOR' and line_split[0] == 'Final':
                    meteor_score = float(line_split[2])
                elif metric == 'BERTSCORE' and line_split[0] == 'roberta-large-mnli_L19_no-idf_version=0.3.11(hug_trans=4.15.0)_fast-tokenizer':
                    bert_score = float(line_split[-1])
        data.append({'BASELINE' : file, 'BLEU' : bleu_score, 'ROUGE' : rouge_score, 'METEOR' : meteor_score, 'BERTSCORE' : bert_score})

pd.DataFrame(data).to_csv('score_report.tsv', sep='\t', index=False)