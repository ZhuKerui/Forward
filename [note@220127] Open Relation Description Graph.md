# [note@220127] Open Relation Description Graph

- Re-draw demonstration (recalculate score for each sentence in the graph) on Google drive
- Change the encoding scheme
   - e1: machine leanring
   - e2: deep learning
   - w1: artificial intelligence
   - s1: artificial neural network ( ann ) is a subfield of the research area machine learning
   - s2: deep learning consists of multiple hidden layer in an artificial neural network.
   

​		e1--s1--w1--s2--e2 => `entity1: e1 entity2: e2 path: e1; w1; e2 sentence1: s1 sentence2: s2`

- Dataset split 
  
   - Total: 107684
   - Train: 85%
   - Valid: 5%
   - Test: 10%
   
- Baselines (Try 1,3,4 First)
  
   1. `entity1: e1 entity2: e2`
   2. `entity1: e1 entity2: e2 path: e1; w1; e2`
   3. k=1
   4. k=3
   5. k=5
   
- Evaluation
  
   - Follow Open Relation Modeling: BLEU, ROUGE, METEOR, BERTScore
   
     ```
     # Command
     bash RM-scorer.sh path_to_model_output path_to_ground_truth
     ```
   
     ```
     # File RM-scorer.sh
     echo BLEU
     cat $1 | sacrebleu --width 4 $2
     echo
     
     echo ROUGE
     rouge -f -a $1 $2 --stats F --ignore_empty
     echo
     
     echo METEOR
     java -Xmx2G -jar /scratch/jeffhj/research/relation_modeling/meteor/meteor-*.jar $1 $2 -norm -lower -noPunct | tail -10
     echo
     
     echo BERTSCORE
     bert-score -r $2 -c $1 --lang en --model roberta-large-mnli
     echo
     ```
   
     requirement: (pip install xxx)
   
     ```
     sacrebleu
     rouge
     bert-score
     ```
   
- Leave GraphWriter for now