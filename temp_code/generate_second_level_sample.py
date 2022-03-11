# def generate_second_level_sample(sample:dict):
#     second_level_sample = {}
#     second_level_sample['key pair'] = sample['pair']
#     second_level_sample['target'] = sample['target']
#     second_level_sample['sources'] = []
#     sources = [nlp(sent) for sent in sample['source']]
#     for t in sample['triple']:
#         ent1_idx, ent2_idx, sent_idx, kw1_span, kw2_span = t
#         kw1_span, kw2_span = eval(kw1_span), eval(kw2_span)
#         if kw1_span[0] > kw2_span[0]:
#             kw1_span, kw2_span = kw2_span, kw1_span
#         doc = sources[sent_idx]
#         i = 0
#         m = {}
#         i2s = {}
#         kw1_i, kw2_i = 0, 0
#         for j in range(len(doc)):
#             m[j] = i
#             if j == kw1_span[0]:
#                 kw1_i = i
#                 i2s[i] = doc[kw1_span[0]:kw1_span[1]+1].text
#             elif j == kw2_span[0]:
#                 kw2_i = i
#                 i2s[i] = doc[kw2_span[0]:kw2_span[1]+1].text
#             elif i not in i2s:
#                 i2s[i] = doc[j].text
#             if (j < kw1_span[0] or j >= kw1_span[1]) and (j < kw2_span[0] or j >= kw2_span[1]):
#                 i += 1
#         g = []
#         tokenized_sent = [[] for _ in range(i)]
#         for tok in doc:
#             head_idx = m[tok.i]
#             tokenized_sent[head_idx].append(tok.text)
#             for child in tok.children:
#                 tail_idx = m[child.i]
#                 if head_idx != tail_idx:
#                     g.extend([(head_idx, tail_idx, child.dep_), (tail_idx, head_idx, 'i_'+child.dep_)])
#         tokenized_sent = [' '.join(p) for p in tokenized_sent]
#         one_sentence_graph = {'pair' : (kw1_i, kw2_i), 'sent' : tokenized_sent, 'graph' : g}
#         second_level_sample['sources'].append(one_sentence_graph)
#     return second_level_sample