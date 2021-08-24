from transformers import PreTrainedModel

class ScoreFunction1(PreTrainedModel):
    
    base_model_prefix = 'bert_model'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)