from tokenization.tiny_tokenizer import TinyLMTokenizer
from tok_testsuite import test_suite, MESSAGES, CHAT_MESSAGES

with open(f"datasets/{input("Enter file name: ")}", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = TinyLMTokenizer(
    vocab_size=int(input("Enter vocab size: "))
)
tokenizer.train(text)

tokenizer.save("tiny_tokenizer.json")

test_suite(tokenizer, MESSAGES)
test_suite(tokenizer, CHAT_MESSAGES)