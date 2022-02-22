1. modify full span finding code. (amod not include, only compound)?

   - use spacy to find phrase
   - **only use "compound"**
2. coverage calculation

   1. AUX: modifiers in spacy document, refer to stanford nlp
   2. maybe some more dep to be included in AUX edges, like "compound"
   3. conj combine with the parent
      - e.g. "historically , information science is associated with computer science , data science , psychology , technology and intelligence agency ."
3. Dependency parse: spacy or stanford NLP? **Use large model of Spacy**?
4. Preprocess: remove () and extra spaces

   - remove too short or too long sentence. (5< or >50?)
5. check score calculation code

   - test "other highly regarded top computer science award include ieee john von neumann medal awarded by the ieee board of director , and the japan kyoto prize for information science ."
6. reject sentence if full span and entity not exactly match

   - e.g. "other highly regarded top computer science award include ieee john von neumann medal awarded by the ieee board of director , and the japan kyoto prize for information science ."
7. score calculation:

   1. pattern frequency - for intuitive
   2. dependency coverage - for informative
      take mean

---

2/10/2022

1. 细化dependency哪些要哪些不要
2. 关于过长的modifier的weight计算
   1. only keep subtrees with modifier dependency for the first off-path link
   2. For each token calculate the shortest path to the main path
   3. Calculate the frequency for each "sub shortest path" and use the frequency as weight
3. wikipedia2vec similarity threshold set to 0.5

2/14/2022

1. tokens with dependency "appos", "compound" and "conj" have the same score with its parent
2. "punct" is removed from both original path length and the subpath length

2/15/2022

1. Collect some example for some sub path pattern to see if more frequent pattern is more explicit for human to understand.
   1. "Machine learning explores the study and construction of algorithms that can learn from and make predictions on data which is manually collected by human."
   2. "Machine learning explores the study and construction of algorithms that can learn from and make predictions on data, which is widely applied to downstream applications, such as image recognition."
