import copy

from transformers import PretrainedConfig, AutoConfig

class JointConfig(PretrainedConfig):
    model_type = "joint"
    is_composition = True

    def __init__(
        self,
        vocab_size=None,
        is_encoder_decoder=True,
        prefix=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        decoder_start_token_id=None,
        title_sep=" / ",
        doc_sep=" // ",
        n_docs=5,
        max_combined_length=300,
        retrieval_vector_size=768,
        retrieval_batch_size=8,
        dataset="wiki_dpr",
        dataset_split="train",
        index_name="compressed",
        index_path=None,
        passages_path=None,
        use_dummy_dataset=False,
        reduce_loss=False,
        label_smoothing=0.0,
        do_deduplication=True,
        exclude_bos_score=False,
        do_marginalize=False,
        output_retrieved=False,
        use_cache=True,
        forced_eos_token_id=None,
        **kwargs
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            prefix=prefix,
            vocab_size=vocab_size,
            **kwargs,
        )
        assert (
            "question_encoder" in kwargs and "generator" in kwargs
        ), "Config has to be initialized with question_encoder and generator config"
        question_encoder_config = kwargs.pop("question_encoder")
        question_encoder_model_type = question_encoder_config.pop("model_type")
        decoder_config = kwargs.pop("generator")
        decoder_model_type = decoder_config.pop("model_type")

        self.question_encoder = AutoConfig.for_model(question_encoder_model_type, **question_encoder_config)
        self.generator = AutoConfig.for_model(decoder_model_type, **decoder_config)

        self.reduce_loss = reduce_loss
        self.label_smoothing = label_smoothing
        self.exclude_bos_score = exclude_bos_score
        self.do_marginalize = do_marginalize

        self.title_sep = title_sep
        self.doc_sep = doc_sep
        self.n_docs = n_docs
        self.max_combined_length = max_combined_length

        self.dataset = dataset
        self.dataset_split = dataset_split
        self.index_name = index_name

        self.retrieval_vector_size = retrieval_vector_size
        self.retrieval_batch_size = retrieval_batch_size
        self.passages_path = passages_path
        self.index_path = index_path
        self.use_dummy_dataset = use_dummy_dataset

        self.output_retrieved = output_retrieved

        self.do_deduplication = do_deduplication

        self.use_cache = use_cache

        if self.forced_eos_token_id is None:
            self.forced_eos_token_id = getattr(self.generator, "forced_eos_token_id", None)

    @classmethod
    def from_question_encoder_generator_configs(
        cls, question_encoder_config: PretrainedConfig, generator_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a :class:`~transformers.EncoderDecoderConfig` (or a derived class) from a pre-trained encoder model
        configuration and decoder model configuration.

        Returns:
            :class:`EncoderDecoderConfig`: An instance of a configuration object
        """
        return cls(question_encoder=question_encoder_config.to_dict(), generator=generator_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default
        :meth:`~transformers.PretrainedConfig.to_dict`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["question_encoder"] = self.question_encoder.to_dict()
        output["generator"] = self.generator.to_dict()
        output["model_type"] = self.__class__.model_type
        return output