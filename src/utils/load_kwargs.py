def load_kwargs(config: dict) -> dict: 
    """ Load optional parameters into kwargs structure.

    """
    kwargs = {}
    optional = ['getHidden', 'precision', 'device', 'showSpecialTokens', 
                'PLL_type', 'id2label', 'addPadToken', 'doLower',
                'addPrefixSpace', 'loadAll', 'checkFileColumns', 
               'batchSize', 'verbose', 
                'maxSequenceLength',
               'loadPretrained', 'numLabels', 
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
                'maxSequenceLength',
               ]
    for option in optional:
        if option in config:
            kwargs[option] = config[option]
    return kwargs
