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
