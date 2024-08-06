def load_kwargs(config: dict) -> dict: 
    """ Load optional parameters into kwargs structure.

    """
    kwargs = {}
    optional = ['getHidden', 'precision', 'device', 'showSpecialTokens', 
                'PLL_type', 'id2label', 'addPadToken', 'doLower',
                'addPrefixSpace', 'loadAll', 'checkFileColumns', 
               'batchSize', 'verbose', 
               'loadPretrained', 'num_labels']
    for option in optional:
        if option in config:
            kwargs[option] = config[option]
    return kwargs
