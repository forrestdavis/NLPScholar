import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.load_tokenizers import load_tokenizers
from src.tokenizers.hf_tokenizer import HFTokenizer

tokenizer = HFTokenizer('bert-base-cased')
text = ['the man is happy.', 'Rudolf is a red nosed reindeer']
print(tokenizer.align_words_ids(text))
#print(tokenizer.align_words_ids([text, text]))

#print(tokenizer.convert_tokens_to_ids('.'))
#print(tokenizer.id_is_punct(119))

config = {'tokenizers':
            {'hf': ['bert-base-cased']}, 
          'device': 'mps'}

tokenizer = load_tokenizers(config)[0]
print(tokenizer)
