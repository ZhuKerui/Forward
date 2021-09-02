data/keyword_f.txt ---- CS keywords

data/wordtree.json ---- word tree for cs keywords

data/entity.txt ---- Reformed cs keywords with '_' replacing ' '

data/co_occur.txt ---- Each line shows the keywords that appear in that line of sentence

data/occur.json ---- Tell which lines do each keyword occur

data/eid2ent.json ---- Mapping from entity id to entity name in wikidata

data/rid2rel.json ---- Mapping from relation id to relation name in wikidat
adata/kg_cs_triples.csv ---- eid-rid-eid triples with eid be referring to possible cs keywords

data/kg_dataset.csv ---- ent-rel-ent triples constructed on knowledge graph with each entity pair co-occurs no less than 10 times in small_sent.txt

data/ollie_pos_dataset.csv ---- data containing triples and sentences with confidence greater than 0.9 in csv form

data/ollie_pos_dataset.json ---- data containing triples and sentences with confidence greater than 0.9

data/ollie_neg_dataset_1.json ---- data containing triples and sentences with confidence less than 0.3

data/ollie_neg_dataset_2.json ---- data containing triples and sentences where no extraction is made

data/my_dataset.json ---- data containing pos, neg_1 and neg_2, splited to train and valid part

data/my_dataset_temp.json ---- smaller set of my_dataset.json

data/single-ollie ---- transformers.dataset style file
