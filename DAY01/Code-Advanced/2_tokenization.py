from transformers import GPT2Tokenizer, BertTokenizer
gpt_tokenizer = GPT2Tokenizer.from_pretrained('./bbpe')
gpt_tokenizer.pad_token = '[PAD]'

sentences = [
    "아 더빙.. 진짜 짜증나네요 목소리",
    "흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나",
    "별루 였다..",
]
tokenized_sentences = [gpt_tokenizer.tokenize(sentence) for sentence in sentences]
batch_inputs = gpt_tokenizer(
    sentences,
    padding="max_length",
    max_length=12,
    truncation=True,
)

print(batch_inputs)


bert_tokenizer = BertTokenizer.from_pretrained(
    "./wordpiece",
    do_lower_case=False,
)

tokenized_sentences = [bert_tokenizer.tokenize(sentence) for sentence in sentences]
batch_inputs = bert_tokenizer(
    sentences,
    padding="max_length",
    max_length=12,
    truncation=True,
)

print(batch_inputs)
