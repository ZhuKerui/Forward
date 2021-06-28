# python print_dep_path.py sent keyword_1 keyword_2
import sys
sys.path.append('..')

from tools.TextProcessing import nlp, find_dependency_path_from_tree
doc = nlp(sys.argv[1])
path = find_dependency_path_from_tree(doc, sys.argv[2], sys.argv[3])
if path:
    print(path)