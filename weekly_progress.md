## Jun 24th

### This week's goal
> The output of the present score function gives low score for some sentences that looks good, find the reason.

### This week's progress

+ Thought about score function performance

    1. I think the reason of low score for some good sentences is that I collected negative samples in the wrong way.

+ I did some observation to check the training data

    1. The number of items in training data is 861614
    2. The most frequently observed dependency paths are:

    ```
    ('i_compound', 57504),
    ('compound', 57504),
    ('i_conj', 49534),
    ('conj', 49534),
    ('prep pobj', 42483),
    ('i_pobj i_prep', 42483),
    ('i_amod', 38900),
    ('amod', 38900),
    ('i_pobj i_prep prep pobj', 11338),
    ('dobj', 7644),
    ('i_dobj', 7644),
    ('i_dobj prep pobj', 7625),
    ('i_pobj i_prep dobj', 7625),
    ('i_pobj i_prep compound', 6054),
    ('i_compound prep pobj', 6054),
    ('i_compound compound', 5944),
    ('prep pobj compound', 4646),
    ('i_compound i_pobj i_prep', 4646),
    ('i_amod amod', 4534),
    ('i_amod compound', 4494),
    ('i_compound amod', 4494),
    ('conj conj', 4178),
    ('i_conj i_conj', 4178),
    ('i_compound i_conj', 3537),
    ('conj compound', 3537),
    ('i_dobj nsubj', 3461),
    ('i_nsubj dobj', 3461),
    ('nmod', 2745),
    ('i_nmod', 2745),
    ('appos', 2693)
    ```
    3. The ratio of the top 30 frequent path already take more than 54% of the total training data.
    4. The ratio of the paths that only contain single token (like "compound", "conj") take 39% of the total training data.
    5. Collect dependency path from keyword pairs that has the npmi between -0.1 and 0.1. 
        + The top 30 frequent paths are as below:
        ```
        ('prep pobj', 637197),
        ('i_pobj i_prep', 637197),
        ('i_amod', 299884),
        ('amod', 299884),
        ('i_compound', 299609),
        ('compound', 299609),
        ('i_pobj i_prep dobj', 143390),
        ('i_dobj prep pobj', 143390),
        ('dobj', 138795),
        ('i_dobj', 138795),
        ('i_conj', 124810),
        ('conj', 124810),
        ('i_compound i_pobj i_prep', 123735),
        ('prep pobj compound', 123735),
        ('i_amod i_pobj i_prep', 115432),
        ('prep pobj amod', 115432),
        ('i_dobj nsubj', 107806),
        ('i_nsubj dobj', 107806),
        ('i_pobj i_prep prep pobj', 106960),
        ('i_amod prep pobj', 77102),
        ('i_pobj i_prep amod', 77102),
        ('i_pobj', 77050),
        ('pobj', 77050),
        ('prep pobj prep pobj', 68813),
        ('i_pobj i_prep i_pobj i_prep', 68813),
        ('i_nsubj prep pobj', 52061),
        ('i_pobj i_prep nsubj', 52061),
        ('i_amod amod', 48060),
        ('i_dobj i_pcomp i_prep', 46328),
        ('prep pcomp dobj', 46328)
        ```
        + The top 30 frequent paths take around 35% of the whole paths
    6. There are many overlaps between the top frequent paths collected from closely related pairs and less related pairs

## JULY 1

### This week's goal
> Do some survey and think about how we can use the original sentence in our score function.
>
> Think about what else features besides dependency path can we use in our score function.

### This week's progress

1. About other possible features for score function.
    + Evaluate the importance of the relation with respect to the whole sentence.
    + Perhaps some keyword phrases should not be splitted, like "python library" should not be considered as good expression for relation between "python" and "library"
    + The position or the role of the keyword. Like, it would be better if the keyword is or is inside the subject or object of the sentence.

2. Implemented part of the idea from paper "Open Relation Extraction and Grounding" to see if the "average commute time" could be helpful for our score function.

3. Thought about using PageRank for collecting training data for score function or selecting sentence candidates.

## JULY 8th

### This week's goal
> Try the 1st sentences from wikipedia page as the positive samples to train the score function model

### This week's progress

1. Trained a baseline score function model using the pretrained model of Bert.
    + Dataset: 
        + 5000 positive sentences from the "1st_wiki_new.json" file and,
        + 5000 negative sentences from the "CS publications in the arxiv dataset by Cornell" (called "filtered_arxiv.json" on the Slack)
    + Training:
        + 8000 samples for training, 2000 samples for validation check
        + Use Bert model to encode the sentences and do a binary classification
    + Result:
        + 1989 sentences out of 2000 sentences are correctly classified.
    + Conclusion:
        + There might exist some learnable patterns or semantic meaning to distinguish the wikipedia page's first sentences from other sentences.
        + It is likely that the misclassified sentences are good sentences in the random corpus.
        + It would be better if we could collect more good sentences and learn more useful patterns.
        
2. Run pagerank algorithm on a set of sentences containing a pair of keywords to see if the selected sentences are good sentences showing the relation
    + Dataset:
        + Sentences in the "filtered_arxiv.json" file
    + Learning:
        + Collect sentences which containing two keywords
        + Encode the sentences using Bert
        + Build a graph among the sentences, taking cosine similarity as the weight of edges
        + Remove the edges pointing to itself or having weight smaller than a threshold
        + Run pagerank and pick the top sentences

## JULY 15th

### This week's goal
> Re-train the score function based on modified training dataset
> Redo the pagerank test based on weight calculation from the paper

### This week's progress

1. Re-train the score function
    + Dataset: 
        + 3325 positive, 4708 negative, each in the format like: <CLS> python is a kind of programming language <SEP> <HEAD_ENT> python <TAIL_ENT> programming language <SEP>
        + terms-cs-cfl-epoch200.txt
    + Result:
        + 1560 out of 1607 items are classified correctly
        + Most of the "wrongly classified" items look like noises.
        
2. Run pagerank algorithm based on changed weight calculation
    + Dataset:
        + Sentences in the "filtered_arxiv.json" file or sentences from wikipedia page containing two keywords
    + Result:
        + wikipedia page may not be enough

### This week's progress

1. Dataset
    + sentences on edge: Wikipedia sentences with 2 keywords co-occur
    + gold sentences: Wikipedia sentences in "summary" section with specific manually defined structures, like "A be B", "A such as B", ...
    + query sentences: "What is the relationship between A and B"

2. Training (E.g., python and programming language)
    1. Starting from "python", get its neighbours, except for "programming language", from each edge, retrieve k sentences
        + one bert model encodes sentences on edge, 
        + one bert model encodes query for present edge
        + the final query is a combination of present query and the overall query
        + e.g., present query: "What is the relationship between python and java"
                overall query: "What is the relationship between python and programming language"

    2. Record the k sentences, move to the next node, repeat step 1, until we reach "programming language"
        + The step for choosing the next node should be guided by some algorithms

    3. Repeat step 1 and 2, starting from "programming language"
    4. Suppose we go through n nodes starting from "python" to "programming language", and m nodes from "programming language" to "python", 
        we have (m+n)k sentences. Use BART to generate l output sequences, compute the loss between the outputs and the gold sentences.