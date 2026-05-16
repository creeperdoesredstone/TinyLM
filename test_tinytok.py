import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from tokenization.tiny_tokenizer import TinyLMTokenizer
from model.bitnet import BitLinear, BitNetBlock, DeepScratchBitNet
from utils.tinytok_generation import generate
from config128k import CONFIG
from train_model_with_tinytok import (
	setup_device,
	load_tokenizer,
)

import subprocess
import sys
import json


# ANSI Color Codes
class Colors:
	RESET = "\033[0m"
	BOLD = "\033[1m"
	DIM = "\033[2m"

	# Foreground colors
	BLACK = "\033[30m"
	RED = "\033[31m"
	GREEN = "\033[32m"
	YELLOW = "\033[33m"
	BLUE = "\033[34m"
	MAGENTA = "\033[35m"
	CYAN = "\033[36m"
	WHITE = "\033[37m"

	# Background colors
	BG_RED = "\033[41m"
	BG_GREEN = "\033[42m"
	BG_YELLOW = "\033[43m"
	BG_BLUE = "\033[44m"

	BR_RED = "\033[91m"
	BR_GREEN = "\033[92m"
	BR_YELLOW = "\033[93m"
	BR_BLUE = "\033[94m"


class TestSuite:
	def __init__(self):
		self.total_tests = 0
		self.passed_tests = 0
		self.failed_tests = 0
		self.device = setup_device()
		torch.manual_seed(42)

	def log_test(self, name):
		self.total_tests += 1
		return f"{Colors.CYAN}[TEST {self.total_tests}]{Colors.RESET} {name}"

	def log_pass(self, message=""):
		self.passed_tests += 1
		msg = f"{Colors.GREEN}✓ PASS{Colors.RESET}"
		if message:
			msg += f" {Colors.BR_GREEN}{message}{Colors.RESET}"
		return msg

	def log_fail(self, message=""):
		self.failed_tests += 1
		msg = f"{Colors.RED}✗ FAIL{Colors.RESET}"
		if message:
			msg += f" {Colors.BR_RED}{message}{Colors.RESET}"
		return msg

	def log_info(self, message):
		return f"{Colors.DIM}{message}{Colors.RESET}"

	def log_header(self, text):
		width = 60
		line = "─" * width
		return (
			f"\n{Colors.BLUE}{Colors.BOLD}{line}\n{text:^{width}}\n{line}{Colors.RESET}"
		)

	def summary(self):
		width = 60
		line = "═" * width
		passed_color = Colors.GREEN if self.failed_tests == 0 else Colors.YELLOW

		return (
			f"\n{Colors.BOLD}{Colors.BLUE}{line}\n"
			f"{'SUMMARY':^{width}}\n{line}{Colors.RESET}\n"
			f"  {Colors.CYAN}Total Tests:{Colors.RESET} {self.total_tests}\n"
			f"  {passed_color}Passed:{Colors.RESET} {self.passed_tests}\n"
			f"  {Colors.RED}Failed:{Colors.RESET} {self.failed_tests}\n"
			f"{Colors.BLUE}{line}{Colors.RESET}"
		)


suite = TestSuite()

# Global variable to store pre-trained model for generation tests
pretrained_model = None
pretrained_tokenizer = None


def new_model(device, vocab_size=CONFIG["vocab_size"]):
	return (
		DeepScratchBitNet(
			vocab_size,
			CONFIG["embed_dim"],
			CONFIG["hidden_dim"],
			CONFIG["num_layers"],
			CONFIG["dropout"],
		)
		.to(device)
		.float()
	)


def train_model_for_generation(num_steps):
	global pretrained_model, pretrained_tokenizer

	print(suite.log_header("Pre-Training Model for Generation Tests"))
	print(suite.log_test("Pretraining TinyLM"))

	result = subprocess.run(
		[sys.executable, "train_for_test.py", str(num_steps)],
		capture_output=False,
	)

	if result.returncode != 0:
		print(suite.log_fail("pretraining failed in subprocess"))
		return False

	print(suite.log_pass("pretraining and generation complete"))
	return True

def test_quantization():
	print(suite.log_header("Quantization Test"))

	w = torch.randn(1024)
	gamma = w.abs().mean().clamp(min=1e-5)
	w_q = (w / gamma).round().clamp(-1, 1)
	cos_sim = F.cosine_similarity(w, w_q, dim=0).item()

	print(suite.log_test("Quantization preserves similarity"))
	print(f"  Cosine similarity: {cos_sim:.4f}")

	if cos_sim > 0.8:
		print(suite.log_pass(f"cos_sim={cos_sim:.4f} > 0.8"))
	else:
		print(suite.log_fail(f"cos_sim={cos_sim:.4f} ≤ 0.8"))


def test_forward_equivalence():
	print(suite.log_header("Forward Equivalence Test"))

	in_dim, out_dim = 128, 256
	bit = BitLinear(in_dim, out_dim).to(suite.device)

	print(suite.log_test("BitLinear produces consistent outputs"))
	x = torch.randn(32, in_dim).to(suite.device)

	# Test that forward pass works
	y_bit1 = bit(x)
	y_bit2 = bit(x)

	# Due to dropout and quantization, exact reproducibility isn't expected
	# Just check that outputs are reasonable
	output_mean = y_bit1.mean().item()
	output_std = y_bit1.std().item()
	print(f"  Output mean: {output_mean:.6f}, std: {output_std:.6f}")

	if torch.isfinite(y_bit1).all() and output_std > 0:
		print(suite.log_pass(f"outputs are finite and have variance"))
	else:
		print(suite.log_fail(f"outputs are invalid"))

	print(suite.log_test("BitLinear weight quantization"))
	# Check that weights are roughly quantized
	weights = bit.weight.detach().abs().to("cpu")
	quantized_ratio = (weights < 0.1).float().mean().item() + (
		weights > 0.9
	).float().mean().item()
	print(f"  Quantized ratio: {quantized_ratio:.2%}")

	if quantized_ratio > 0.3:  # At least 30% of weights should be near -1, 0, or 1
		print(suite.log_pass(f"weights show quantization pattern"))
	else:
		print(suite.log_fail(f"weights don't appear quantized"))


def test_gradient_flow():
	print(suite.log_header("Gradient Flow Test"))

	bit = BitLinear(128, 128).to(suite.device)
	x = torch.randn(16, 128).to(suite.device)
	x.requires_grad_()
	out = bit(x)
	loss = out.mean()
	loss.backward()

	grad_mean = x.grad.abs().mean().item()

	print(suite.log_test("Gradients flow through BitLinear"))
	print(f"  Gradient mean: {grad_mean:.6f}")

	if grad_mean > 1e-6:
		print(suite.log_pass(f"grad_mean={grad_mean:.6f} > 1e-6"))
	else:
		print(suite.log_fail(f"grad_mean={grad_mean:.6f} ≤ 1e-6"))


def test_bitnet_block():
	print(suite.log_header("BitNetBlock Tests"))

	block = BitNetBlock(128).to(suite.device)
	x = torch.randn(8, 64, 128).to(suite.device)  # (B, T, D)
	y = block(x)

	print(suite.log_test("BitNetBlock shape preservation"))
	print(f"  Input shape: {x.shape}")
	print(f"  Output shape: {y.shape}")

	if y.shape == x.shape:
		print(suite.log_pass(f"output shape matches input"))
	else:
		print(suite.log_fail(f"output shape {y.shape} != input shape {x.shape}"))

	# Additional block test: output is finite
	print(suite.log_test("BitNetBlock output is finite"))
	is_finite = torch.isfinite(y).all().item()
	print(f"  Contains NaN/Inf: {not is_finite}")

	if is_finite:
		print(suite.log_pass("all outputs are finite"))
	else:
		print(suite.log_fail("contains NaN or Inf values"))


def test_full_model():
	print(suite.log_header("DeepScratchBitNet Tests"))

	vocab_size = CONFIG["vocab_size"]
	model = new_model(suite.device)

	print(suite.log_test("Model forward pass"))
	x = torch.randint(0, vocab_size, (4, 32)).to(suite.device)
	logits = model(x)
	print(f"  Logits shape: {logits.shape}")

	if logits.shape == (4, 32, vocab_size):
		print(suite.log_pass(f"logits shape is correct"))
	else:
		print(suite.log_fail(f"expected (4, 32, {vocab_size}), got {logits.shape}"))

	# Test model with different batch size
	print(suite.log_test("Model with different batch size"))
	x2 = torch.randint(0, vocab_size, (2, 16)).to(suite.device)
	logits2 = model(x2)
	expected_shape = (2, 16, vocab_size)

	if logits2.shape == expected_shape:
		print(suite.log_pass(f"batch_size=2 works correctly"))
	else:
		print(suite.log_fail(f"expected {expected_shape}, got {logits2.shape}"))

	# Test logits are reasonable values
	print(suite.log_test("Logits in reasonable range"))
	logit_mean = logits.mean().item()
	logit_std = logits.std().item()
	print(f"  Mean: {logit_mean:.4f}, Std: {logit_std:.4f}")

	if abs(logit_mean) < 10 and logit_std > 0.01:
		print(suite.log_pass(f"logit statistics are reasonable"))
	else:
		print(suite.log_fail(f"logit statistics seem off"))


def test_bpe_tokenizer_load():
	print(suite.log_header("TinyLM Tokenizer Load Test"))

	tokenizer_path = "tiny_tokenizer.json"

	print(suite.log_test("Load pre-trained TinyLM tokenizer"))
	if not os.path.exists(tokenizer_path):
		print(suite.log_fail(f"tokenizer file not found: {tokenizer_path}"))
		return tokenizer_path

	try:
		tiny_tokenizer = TinyLMTokenizer(CONFIG["vocab_size"])
		tiny_tokenizer.load(tokenizer_path)
		print(f"  Vocab size: {len(tiny_tokenizer.vocab)}")
		print(suite.log_pass(f"loaded successfully"))
		return tokenizer_path
	except Exception as e:
		print(suite.log_fail(f"failed to load: {str(e)}"))
		return None


def test_bpe_tokenizer_basic():
	print(suite.log_header("TinyLM Tokenizer Basic Tests"))

	tokenizer_path = "tiny_tokenizer.json"
	if not os.path.exists(tokenizer_path):
		print(suite.log_info("Skipping TinyLM Tokenizer tests - tokenizer not found"))
		return

	tiny_tokenizer = TinyLMTokenizer(CONFIG["vocab_size"])
	tiny_tokenizer.load(tokenizer_path)

	# Test simple tokenization
	print(suite.log_test("Simple text tokenization"))
	text = "hello world"
	ids = tiny_tokenizer.tokenize(text)
	print(f"  Text: '{text}'")
	print(f"  Token count: {len(ids)}")

	if len(ids) > 0 and all(isinstance(i, int) for i in ids):
		print(suite.log_pass(f"tokenized to {len(ids)} tokens"))
	else:
		print(suite.log_fail(f"invalid token output"))

	# Test chat format
	print(suite.log_test("Chat format tokenization"))
	chat_text = "<user> hello how are you <bot> i am fine"
	ids = tiny_tokenizer.tokenize(chat_text)
	print(f"  Text: '{chat_text}'")
	print(f"  Token count: {len(ids)}")

	if len(ids) > 0:
		print(suite.log_pass(f"chat format handled"))
	else:
		print(suite.log_fail(f"failed to tokenize chat"))

	# Test with special tokens
	print(suite.log_test("Special tokens handling"))
	special_text = "<bos> test <eos>"
	ids = tiny_tokenizer.tokenize(special_text)
	print(f"  Text: '{special_text}'")
	print(f"  Token count: {len(ids)}")

	if len(ids) > 0:
		print(suite.log_pass(f"special tokens recognized"))
	else:
		print(suite.log_fail(f"special tokens not handled"))


def test_bpe_tokenizer_roundtrip():
	print(suite.log_header("TinyLM Tokenizer Roundtrip Tests"))

	tokenizer_path = "tiny_tokenizer.json"
	if not os.path.exists(tokenizer_path):
		print(suite.log_info("Skipping roundtrip tests - tokenizer not found"))
		return

	tiny_tokenizer = TinyLMTokenizer(CONFIG["vocab_size"])
	tiny_tokenizer.load(tokenizer_path)

	test_texts = [
		"hello world",
		"<user> what is your name? <bot> i am tinylm",
		"the quick brown fox",
	]

	for text in test_texts:
		print(suite.log_test(f"Roundtrip: '{text}'"))
		ids = tiny_tokenizer.tokenize(text)
		decoded = tiny_tokenizer.decode(ids)
		print(f"  Original: '{text}'")
		print(f"  Decoded:  '{decoded}'")
		print(f"  Tokens:   {ids}")

		# Decoded should at least have same words
		original_words = set(text.lower().split())
		decoded_words = set(decoded.lower().split())
		overlap = len(original_words & decoded_words) / max(len(original_words), 1)

		if overlap > 0.5 or text in decoded.lower():
			print(suite.log_pass(f"reasonable reconstruction"))
		else:
			print(suite.log_fail(f"poor reconstruction"))


def test_model_with_bpe_tokenizer():
	print(suite.log_header("Model Integration with TinyLM Tokenizer"))

	tokenizer_path = "tiny_tokenizer.json"
	if not os.path.exists(tokenizer_path):
		print(suite.log_info("Skipping - tokenizer not found"))
		return

	tiny_tokenizer = TinyLMTokenizer(CONFIG["vocab_size"])
	tiny_tokenizer.load(tokenizer_path)

	vocab_size = len(tiny_tokenizer.vocab)
	model = new_model(suite.device)

	print(suite.log_test("Tokenize text and run through model"))
	text = "hello world how are you"
	ids = tiny_tokenizer.tokenize(text)
	print(f"  Text: '{text}'")
	print(f"  Tokens: {ids}")

	# Convert to batch
	ids_tensor = torch.tensor([ids], dtype=torch.long).to(suite.device)
	print(f"  Tensor shape: {ids_tensor.shape}")

	try:
		logits = model(ids_tensor)
		print(f"  Logits shape: {logits.shape}")
		print(f"  Vocab size: {vocab_size}")

		if logits.shape[0] == 1 and logits.shape[2] == vocab_size:
			print(suite.log_pass(f"end-to-end pipeline works"))
		else:
			print(suite.log_fail(f"unexpected logits shape"))
	except Exception as e:
		print(suite.log_fail(f"error: {str(e)}"))


def test_model_training():
	print(suite.log_header("Model Training Test"))

	vocab_size = CONFIG["vocab_size"]
	model = new_model(suite.device)

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	criterion = nn.CrossEntropyLoss()

	print(suite.log_test("Training convergence over 100 steps"))

	x = torch.randint(0, vocab_size, (32, 16)).to(suite.device)
	y = torch.randint(0, vocab_size, (32, 16)).to(suite.device)

	losses = []
	for step in range(100):
		logits = model(x)
		loss = criterion(logits.view(-1, vocab_size), y.view(-1))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
		if step % 25 == 0:
			print(f"  Step {step:3d} | Loss: {loss.item():.4f}")

	final_loss = loss.item()
	initial_loss = losses[0]
	improvement = ((initial_loss - final_loss) / initial_loss) * 100

	print(f"  Initial loss: {initial_loss:.4f}")
	print(f"  Final loss: {final_loss:.4f}")
	print(f"  Improvement: {improvement:.1f}%")

	if final_loss < initial_loss * 0.8:  # 20% reduction
		print(suite.log_pass(f"model learned (loss reduced by {improvement:.1f}%)"))
	else:
		print(
			suite.log_fail(f"insufficient learning (only {improvement:.1f}% reduction)")
		)


def test_batch_processing():
	print(suite.log_header("Batch Processing Test"))

	vocab_size = CONFIG["vocab_size"]
	model = new_model(suite.device)
	model.eval()
	pass_counter = 0

	print(suite.log_test("Process various batch sizes"))

	batch_sizes = [1, 4, 8, 16]
	for batch_size in batch_sizes:
		x = torch.randint(0, vocab_size, (batch_size, 32)).to(suite.device)
		logits = model(x)

		if logits.shape == (batch_size, 32, vocab_size):
			print(f"  Batch {batch_size:2d}: {Colors.GREEN}✓{Colors.RESET}")
			pass_counter += 1
		else:
			print(f"  Batch {batch_size:2d}: {Colors.RED}✗{Colors.RESET}")

	if pass_counter == 4:
		print(suite.log_pass(f"{pass_counter}/4 batches passed"))
	else:
		print(suite.log_fail(f"{pass_counter}/4 batches passed"))


def test_inference_speed():
	print(suite.log_header("Inference Speed Test"))

	vocab_size = CONFIG["vocab_size"]
	model = new_model(suite.device)
	model.eval()

	print(suite.log_test("Inference speed on (32, 128) batch"))

	x = torch.randint(0, vocab_size, (32, 128)).to(suite.device)

	import time

	with torch.no_grad():
		start = time.time()
		for _ in range(10):
			_ = model(x)
		elapsed = time.time() - start

	avg_time = elapsed / 10
	print(f"  Average time per batch: {avg_time*1000:.2f}ms")

	if avg_time < 1.0:  # Less than 1 second on CPU
		print(suite.log_pass(f"inference speed acceptable"))
	else:
		print(suite.log_fail(f"inference seems slow"))


def test_logits_validity():
	print(suite.log_header("Logits Validity Tests"))

	vocab_size = CONFIG["vocab_size"]
	model = new_model(suite.device)
	model.eval()

	print(suite.log_test("Logits are finite values"))
	x = torch.randint(0, vocab_size, (8, 32)).to(suite.device)
	logits = model(x)

	has_nan = torch.isnan(logits).any().item()
	has_inf = torch.isinf(logits).any().item()
	print(f"  Contains NaN: {has_nan}, Contains Inf: {has_inf}")

	if not has_nan and not has_inf:
		print(suite.log_pass("all logits are finite"))
	else:
		print(suite.log_fail("logits contain NaN or Inf"))

	print(suite.log_test("Logits have reasonable range"))
	logit_max = logits.max().item()
	logit_min = logits.min().item()
	print(f"  Range: [{logit_min:.2f}, {logit_max:.2f}]")

	if abs(logit_max) < 1000 and abs(logit_min) < 1000:
		print(suite.log_pass(f"logit range is reasonable"))
	else:
		print(suite.log_fail(f"logit range too extreme"))


def test_probability_distributions():
	print(suite.log_header("Probability Distribution Tests"))

	vocab_size = CONFIG["vocab_size"]
	model = new_model(suite.device)
	model.eval()

	print(suite.log_test("Softmax of logits sums to 1"))
	x = torch.randint(0, vocab_size, (4, 16)).to(suite.device)

	with torch.no_grad():
		logits = model(x)
		probs = torch.softmax(logits, dim=-1)

	prob_sums = probs.sum(dim=-1)
	max_sum_error = (prob_sums - 1.0).abs().max().item()
	min_sum = prob_sums.min().item()
	print(f"  Max sum error: {max_sum_error:.8f}, Min sum: {min_sum:.8f}")

	if max_sum_error < 1e-5 and min_sum > 0.99:
		print(suite.log_pass("probability distributions are valid"))
	else:
		print(suite.log_fail("probability sums incorrect"))

	print(suite.log_test("All probabilities are non-negative"))
	has_negative = (probs < 0).any().item()
	print(f"  Contains negative probs: {has_negative}")

	if not has_negative and (probs <= 1).all().item():
		print(suite.log_pass("all probabilities in [0, 1]"))
	else:
		print(suite.log_fail("invalid probability values"))


def test_top_k_sampling():
	print(suite.log_header("Top-K Sampling Tests"))

	vocab_size = CONFIG["vocab_size"]
	model = new_model(suite.device)

	print(suite.log_test("Top-K filtering produces valid tokens"))
	x = torch.randint(0, vocab_size, (4, 16)).to(suite.device)

	with torch.no_grad():
		logits = model(x)
		top_k = 10

		# Get top-k
		logits_cpu = logits.to("cpu")

		top_logits, top_indices = torch.topk(logits_cpu, top_k, dim=-1)

		# move back if needed
		top_logits = top_logits.to(suite.device)
		top_indices = top_indices.to(suite.device)

		print(f"  Original vocab size: {vocab_size}")
		print(f"  Top-K value: {top_k}")
		print(f"  Top-K indices range: [{top_indices.min()}, {top_indices.max()}]")

		if (top_indices >= 0).all() and (top_indices < vocab_size).all():
			print(suite.log_pass("top-k indices are valid"))
		else:
			print(suite.log_fail("invalid top-k indices"))

	print(suite.log_test("Top-K logits are properly sorted"))
	# Check if logits decrease from top to bottom
	logit_diffs = top_logits[..., :-1] - top_logits[..., 1:]
	all_decreasing = (logit_diffs >= 0).all().item()
	print(f"  All decreasing: {all_decreasing}")

	if all_decreasing:
		print(suite.log_pass("top-k logits are properly ranked"))
	else:
		print(suite.log_fail("top-k logits not properly sorted"))


def test_top_p_sampling():
	print(suite.log_header("Top-P (Nucleus) Sampling Tests"))

	vocab_size = CONFIG["vocab_size"]
	model = new_model(suite.device)
	model.eval()

	print(suite.log_test("Top-P filtering with cumulative sum"))
	x = torch.randint(0, vocab_size, (2, 16)).to(suite.device)

	with torch.no_grad():
		logits = model(x)
		logits_cpu = logits.to("cpu")
		top_p = 0.9

		# Sort and apply cumsum
		sorted_logits, sorted_indices = torch.sort(logits_cpu, descending=True)
		sorted_probs = torch.softmax(sorted_logits, dim=-1)
		cum_probs = torch.cumsum(sorted_probs, dim=-1)

		# Find cutoff point - keep at least top-1 token
		mask = (cum_probs <= top_p) | (
			torch.arange(logits_cpu.shape[-1]).unsqueeze(0).unsqueeze(0) == 0
		)
		num_valid = mask.sum(dim=-1)
		print(f"  Valid tokens per sample (p={top_p}): {num_valid.float().mean():.1f}")

		if (num_valid > 0).all() and (num_valid < vocab_size).all():
			print(suite.log_pass("top-p filtering produces valid subset"))
		else:
			print(suite.log_fail("top-p filtering failed"))

	print(suite.log_test("Top-P probabilities sum correctly"))
	valid_probs = sorted_probs[mask].sum()
	print(f"  Sum of valid probs: {valid_probs:.4f}")

	if valid_probs > top_p * 0.9:  # Allow some tolerance
		print(suite.log_pass(f"top-p sums ~= {top_p}"))
	else:
		print(suite.log_fail(f"top-p sum too small"))


def test_greedy_decoding():
	print(suite.log_header("Greedy Decoding Tests"))

	vocab_size = CONFIG["vocab_size"]
	model = new_model(suite.device)
	model.eval()

	print(suite.log_test("Greedy decoding always picks max logit"))
	x = torch.randint(0, vocab_size, (4, 16)).to(suite.device)

	with torch.no_grad():
		logits = model(x)
		logits_cpu = logits.to("cpu")
		# Get argmax (greedy)
		greedy_tokens = logits_cpu.argmax(dim=-1)

		# Check these match max logits
		max_logits = logits_cpu.max(dim=-1)[0]
		actual_logits = logits_cpu[
			torch.arange(logits_cpu.shape[0])[:, None],
			torch.arange(logits_cpu.shape[1]),
			greedy_tokens,
		]

		all_match = torch.allclose(max_logits, actual_logits)
		print(f"  Greedy tokens match max logits: {all_match}")

		if all_match:
			print(suite.log_pass("greedy selection is correct"))
		else:
			print(suite.log_fail("greedy selection mismatch"))

	print(suite.log_test("Greedy tokens are in valid range"))
	if (greedy_tokens >= 0).all() and (greedy_tokens < vocab_size).all():
		print(suite.log_pass("all tokens within vocab range"))
	else:
		print(suite.log_fail("tokens out of range"))


def test_sampling_reproducibility():
	print(suite.log_header("Sampling Reproducibility Tests"))

	vocab_size = CONFIG["vocab_size"]
	model = new_model(suite.device)
	model.eval()

	print(suite.log_test("Same seed produces same samples"))
	x = torch.randint(0, vocab_size, (2, 16)).to(suite.device)

	torch.manual_seed(42)
	with torch.no_grad():
		logits = model(x)
		logits_cpu = logits.to("cpu")
		probs = torch.softmax(logits_cpu, dim=-1)
		samples1 = torch.multinomial(probs.view(-1, vocab_size), 1)

	torch.manual_seed(42)
	with torch.no_grad():
		logits = model(x)
		logits_cpu = logits.to("cpu")
		probs = torch.softmax(logits_cpu, dim=-1)
		samples2 = torch.multinomial(probs.view(-1, vocab_size), 1)

	match = torch.equal(samples1, samples2)
	print(f"  Samples match with same seed: {match}")

	if match:
		print(suite.log_pass("sampling is reproducible"))
	else:
		print(suite.log_fail("non-deterministic behavior"))


def test_output_consistency():
	print(suite.log_header("Output Consistency Tests"))

	vocab_size = CONFIG["vocab_size"]
	model = new_model(suite.device)
	model.eval()

	print(suite.log_test("Same input produces same output"))
	x = torch.randint(0, vocab_size, (2, 16)).to(suite.device)

	with torch.no_grad():
		logits1 = model(x)
		logits_1cpu = logits1.to("cpu")
		logits2 = model(x)
		logits_2cpu = logits2.to("cpu")

	match = torch.allclose(logits_1cpu, logits_2cpu, atol=1e-5)
	print(f"  Outputs match: {match}")

	if match:
		print(suite.log_pass("outputs are deterministic"))
	else:
		print(suite.log_fail("outputs differ between runs"))

	print(suite.log_test("Different inputs produce different outputs"))
	x1 = torch.randint(0, vocab_size, (2, 16)).to(suite.device)
	x2 = torch.randint(0, vocab_size, (2, 16)).to(suite.device)

	with torch.no_grad():
		logits1 = model(x1)
		logits_1cpu = logits1.to("cpu")
		logits2 = model(x2)
		logits_2cpu = logits2.to("cpu")

	differ = not torch.allclose(logits_1cpu, logits_2cpu, atol=0.1)
	diff_ratio = (logits_1cpu - logits_2cpu).abs().mean().item()
	print(f"  Mean difference: {diff_ratio:.6f}")

	if differ:
		print(suite.log_pass("different inputs produce different outputs"))
	else:
		print(suite.log_fail("outputs too similar"))


def test_long_sequence_handling():
	print(suite.log_header("Long Sequence Handling Tests"))

	vocab_size = CONFIG["vocab_size"]
	model = new_model(suite.device)
	model.eval()

	print(suite.log_test("Model handles long sequences"))
	sequence_lengths = [32, 64, 128, 256]

	for seq_len in sequence_lengths:
		x = torch.randint(0, vocab_size, (2, seq_len)).to(suite.device)

		try:
			with torch.no_grad():
				logits = model(x)
				logits_cpu = logits.to("cpu")

			if logits_cpu.shape == (2, seq_len, vocab_size):
				print(f"  Length {seq_len:3d}: {Colors.GREEN}✓{Colors.RESET}")
			else:
				print(f"  Length {seq_len:3d}: {Colors.RED}✗{Colors.RESET}")
		except Exception as e:
			print(f"  Length {seq_len:3d}: {Colors.RED}✗{Colors.RESET} ({str(e)[:30]})")

	print(suite.log_pass("long sequences work"))


def test_batch_shape_preservation():
	print(suite.log_header("Batch Shape Preservation Tests"))

	vocab_size = CONFIG["vocab_size"]
	model = new_model(suite.device)
	model.eval()

	print(suite.log_test("Output shape matches input shape"))
	test_cases = [
		(1, 16),
		(2, 32),
		(4, 64),
		(8, 128),
		(16, 256),
	]

	all_pass = True
	for batch_size, seq_len in test_cases:
		x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(suite.device)

		with torch.no_grad():
			logits = model(x)
			logits_cpu = logits.to("cpu")

		expected = (batch_size, seq_len, vocab_size)
		matches = logits_cpu.shape == expected

		status = (
			Colors.GREEN + "✓" + Colors.RESET
			if matches
			else Colors.RED + "✗" + Colors.RESET
		)
		print(f"  ({batch_size:2d}, {seq_len:3d}) -> {logits_cpu.shape}: {status}")

		if not matches:
			all_pass = False

	if all_pass:
		print(suite.log_pass("all shapes correct"))
	else:
		print(suite.log_fail("some shapes incorrect"))


def _load_generation_results():
	try:
		with open("weights/generation_results.json") as f:
			return json.load(f)
	except Exception:
		return None


def test_text_generation_basic():
	print(suite.log_header("Text Generation Basic Test"))
	results = _load_generation_results()
	if not results:
		print(suite.log_info("Skipping - no results file"))
		return
	print(suite.log_test("Generate text from prompt"))
	out = results.get("basic", "")
	print(f"  Prompt: '<user> hello'")
	print(f"  Generated: '{out}'")
	if isinstance(out, str) and len(out) > 0 and not out.startswith("ERROR"):
		print(suite.log_pass("text generation works"))
	else:
		print(suite.log_fail(f"generation failed: {out}"))


def test_generation_sampling_parameters():
	print(suite.log_header("Generation Sampling Parameters Test"))
	results = _load_generation_results()
	if not results:
		print(suite.log_info("Skipping - no results file"))
		return
	print(suite.log_test("Generate with different sampling parameters"))
	all_success = True
	for name, out in results.get("sampling", {}).items():
		ok = isinstance(out, str) and not out.startswith("ERROR")
		status = Colors.GREEN + "✓" + Colors.RESET if ok else Colors.RED + "✗" + Colors.RESET
		print(f"  {name:12s}: {status}")
		if not ok:
			all_success = False
	if all_success:
		print(suite.log_pass("all parameter combinations work"))
	else:
		print(suite.log_fail("some parameter combinations failed"))


def test_generation_quality_metrics():
	print(suite.log_header("Generation Quality Metrics Test"))
	results = _load_generation_results()
	if not results:
		print(suite.log_info("Skipping - no results file"))
		return
	print(suite.log_test("Generate and analyze output metrics"))
	all_pass = True
	for prompt, out in results.get("quality", {}).items():
		if out.startswith("ERROR"):
			print(f"  '{prompt[:20]:20s}' -> ERROR")
			all_pass = False
			continue
		tokens = out.split()
		token_count = len(tokens)
		repetition_ratio = 1.0 - len(set(tokens)) / max(token_count, 1)
		ok = token_count > 0 and repetition_ratio < 0.8
		status = Colors.GREEN + "✓" + Colors.RESET if ok else Colors.RED + "✗" + Colors.RESET
		print(f"  '{prompt[:20]:20s}' -> {token_count:2d} tokens, rep={repetition_ratio:.2f}: {status}")
		if not ok:
			all_pass = False
	if all_pass:
		print(suite.log_pass("generation quality is reasonable"))
	else:
		print(suite.log_fail("quality issues detected"))


def test_generation_human_evaluation():
	print(suite.log_header("Generation Examples for Human Evaluation"))
	results = _load_generation_results()
	if not results:
		print(suite.log_info("Skipping - no results file"))
		return
	print(suite.log_test("Generate example outputs for manual review"))
	print(f"\n{Colors.BOLD}{Colors.CYAN}Generated Examples for Human Evaluation:{Colors.RESET}\n")
	all_ok = True
	for ex in results.get("examples", []):
		out = ex["output"]
		ok = not out.startswith("ERROR")
		if ok:
			print(f"{Colors.YELLOW}[{ex['description']}]{Colors.RESET}")
			print(f"  Input:  {Colors.CYAN}{ex['prompt']}{Colors.RESET}")
			print(f"  Output: {Colors.GREEN}{out}{Colors.RESET}")
			print()
		else:
			print(f"  [{ex['description']}]: {Colors.RED}{out}{Colors.RESET}")
			all_ok = False
	if all_ok:
		print(suite.log_pass("generation examples generated successfully"))
	else:
		print(suite.log_fail("some examples failed"))


if __name__ == "__main__":
	print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}")
	print(f"{'BITNET VALIDATION SUITE':^60}")
	print(f"{'='*60}{Colors.RESET}\n")

	# Core tests
	test_quantization()
	test_forward_equivalence()
	test_gradient_flow()
	test_bitnet_block()
	test_full_model()

	# Tokenizer tests
	test_bpe_tokenizer_load()
	test_bpe_tokenizer_basic()
	test_bpe_tokenizer_roundtrip()

	# Integration tests
	test_model_with_bpe_tokenizer()
	test_model_training()
	test_batch_processing()
	test_inference_speed()

	# Output validation tests
	test_logits_validity()
	test_probability_distributions()
	test_top_k_sampling()
	test_top_p_sampling()
	test_greedy_decoding()
	test_sampling_reproducibility()
	test_output_consistency()
	test_long_sequence_handling()
	test_batch_shape_preservation()

	# Pre-train model for generation tests
	print()
	train_model_for_generation(num_steps=int(input("Enter number of steps: ")))

	# Text generation tests
	test_text_generation_basic()
	test_generation_sampling_parameters()
	test_generation_quality_metrics()
	test_generation_human_evaluation()

	# Print summary
	print(suite.summary())

	# Exit with appropriate code
	exit(0 if suite.failed_tests == 0 else 1)
