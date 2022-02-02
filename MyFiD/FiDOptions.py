from src.options import Options

class FiDOptions(Options):
    def add_eval_options(self):
        super().add_eval_options()
        self.parser.add_argument('--write_results', type=bool, default=False, help='save results')
        self.parser.add_argument('--write_crossattention_scores', type=bool, default=False, 
                        help='save dataset with cross-attention scores')


    def initialize_parser(self):
        super().initialize_parser()
        self.parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=-1,
                        help="Main port (for multi-node SLURM jobs)")
        
        
    def add_model_specific_options(self):
        self.parser.add_argument('--model_size', type=str, default='base')
        self.parser.add_argument('--n_context', type=int, default=1)
        self.parser.add_argument('--text_maxlength', type=int, default=200, 
                        help='maximum number of tokens in text segments (question+passage)')
        self.parser.add_argument('--answer_maxlength', type=int, default=-1, 
                        help='maximum number of tokens used to train the model, no truncation if -1')
        self.parser.add_argument('--no_sent', type=bool, default=False, help='no sentence in context')
        self.parser.add_argument('--no_path', type=bool, default=False, help='no path information')
        
        