def load_kwargs(config: dict) -> dict: 
    """ Load optional parameters into kwargs structure.

    """
    kwargs = {}
    optional = ['getHidden', 'precision', 'device', 
                'PLL_type', 'id2label', 'addPadToken', 'doLower',
                'addPrefixSpace', 'loadAll', 'checkFileColumns', 
               'batchSize', 'verbose', 
               'loadPretrained', 'numLabels', 
                'stride', 
                # dataset args
                'seed',
                'samplePercent',
                'textLabel',
                'pairLabel',
                'tokensLabel',
                'tagsLabel',
                # training args
                'modelfpath',
                'epochs',
                'eval_strategy',
                'eval_steps',
                'batchSize',
                'learning_rate',
                'weight_decay',
                'save_strategy',
                'save_steps',
                'load_best_model_at_end',
                'wholeWordMasking',
                'maskProbability',
                'maxTrainSequenceLength',
                # analysis args
                'predfpath',
                'datafpath',
                'condfpath',
                'resultsfpath',
                'sep',
                'pred_measure',
                'word_summary',
                'roi_summary',
                'k_lemmas',
                'punctuation',
                # evaluate args
                'giveAllLabels',
               ]
    for option in optional:
        if option in config:
            kwargs[option] = config[option]
    return kwargs
