# python build_wordtree.py keyword_file wordtree_file
import sys

sys.path.append('..')
from tools.TextProcessing import build_word_tree_v2

build_word_tree_v2(sys.argv[1], sys.argv[2], sys.argv[3])