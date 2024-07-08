import os
from Korpora import Korpora
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer

def write_lines(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write(line + '\n')


nsmc = Korpora.load('nsmc', force_download=True)
train_txt_path, test_txt_path = './train.txt', './test.txt'
bbpe_dir = './bbpe'

if not os.path.exists(train_txt_path):
    write_lines(train_txt_path, nsmc.train.get_all_texts())
if not os.path.exists(test_txt_path):
    write_lines(test_txt_path, nsmc.test.get_all_texts())
os.makedirs(bbpe_dir, exist_ok=True)

# GPT Tokenizer
bytebpe_tokenizer = ByteLevelBPETokenizer()
bytebpe_tokenizer.train(
    files=[train_txt_path, test_txt_path],
    vocab_size=10000,
    special_tokens=["[PAD]"])

# BERT Tokenizer
wordpiece_tokenizer = BertWordPieceTokenizer(lowercase=False)
wordpiece_tokenizer.train(
    files=[train_txt_path, test_txt_path],
    vocab_size=10000,
)
wordpiece_tokenizer.save_model("./wordpiece")
