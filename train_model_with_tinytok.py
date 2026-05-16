import torch
import torch.nn as nn
import os
import math
import sys
import multiprocessing as mp

from config import CONFIG
from tokenization.tiny_tokenizer import TinyLMTokenizer
from model.bitnet import BitLinear, DeepScratchBitNet
from utils.export import export_packed_bitnet
from torch.utils.data import DataLoader

CHECKPOINT_DIR = "weights/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

torch.set_float32_matmul_precision("high")


class ANSI:
	RESET = "\033[0m"
	DIM = "\033[2m"
	BOLD = "\033[1m"

	GREEN = "\033[92m"
	CYAN = "\033[96m"
	YELLOW = "\033[93m"
	RED = "\033[91m"
	PURPLE = "\033[95m"
	ORANGE = "\033[38;5;208m"


class ProgressBar:
	def __init__(self, total, width=40):
		self.total = total
		self.width = width * 2
		self.ema_loss = None

	def update(self, step, loss, lr, epoch, opt_step):
		# EMA smoothing
		if self.ema_loss is None:
			self.ema_loss = loss
		else:
			self.ema_loss = 0.9 * self.ema_loss + 0.1 * loss

		ratio = step / self.total
		filled = ratio * self.width

		full = int(filled // 2)
		half = (filled % 2) >= 1

		bar = ANSI.GREEN + "█" * full
		if half:
			bar += "▌"

		remaining = self.width // 2 - full - (1 if half else 0)
		bar += ANSI.DIM + " " * remaining + ANSI.RESET

		# Loss color
		if self.ema_loss < 1.5:
			c = ANSI.CYAN
		elif self.ema_loss < 2.0:
			c = ANSI.GREEN
		elif self.ema_loss < 3.0:
			c = ANSI.YELLOW
		elif self.ema_loss < 4.0:
			c = ANSI.ORANGE
		elif self.ema_loss < 5.0:
			c = ANSI.RED
		else:
			c = ANSI.PURPLE

		msg = (
			f"\r{ANSI.CYAN}Epoch {epoch:2}{ANSI.RESET} "
			f"[{bar}] "
			f"{ANSI.BOLD}{step:5}/{self.total:5}{ANSI.RESET} "
			f"({ratio*100:5.1f}%) | "
			f"Loss: {c}{self.ema_loss:.4f}{ANSI.RESET} | "
			f"LR: {ANSI.CYAN}{lr:.6f}{ANSI.RESET} | "
			f"{opt_step:5}   "
		)

		sys.stdout.write(msg)
		sys.stdout.flush()

	def close(self):
		print()


def print_header(text):
	print(f"\n{ANSI.CYAN}{'─'*50}")
	print(f"{ANSI.BOLD}{text.center(50)}{ANSI.RESET}")
	print(f"{ANSI.CYAN}{'─'*50}{ANSI.RESET}\n")


# -------------------------------
# DEVICE
# -------------------------------
def setup_device():
	if mp.current_process().name != "MainProcess":
		return torch.device("cpu")
	try:
		if hasattr(torch, "xpu") and torch.xpu.is_available():
			device = torch.device("xpu")
		elif torch.cuda.is_available():
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")
	except:
		device = torch.device("cpu")

	device = torch.device("cpu")
	print("Using:", device)
	return device


# -------------------------------
# TOKENIZER
# -------------------------------
def load_tokenizer(CONFIG):
	tokenizer = TinyLMTokenizer(vocab_size=CONFIG["vocab_size"])
	tokenizer.load("tiny_tokenizer.json")
	return tokenizer


# -------------------------------
# DATA
# -------------------------------
from concurrent.futures import ThreadPoolExecutor, as_completed

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


def _tokenize_batch(args):
	import os

	os.environ["CUDA_VISIBLE_DEVICES"] = ""

	samples, tokenizer = args
	results = []

	for sample in samples:
		try:
			# Normalize spacing
			sample = sample.strip()

			if "<user>" not in sample or "<bot>" not in sample:
				continue

			# Split cleanly
			parts = sample.split("<bot>")
			if len(parts) != 2:
				continue

			user_part = parts[0].replace("<user>", "").strip()
			bot_part = parts[1].replace("<end>", "").strip()

			if not user_part or not bot_part:
				continue

			full_text = f"<user> {user_part} <bot> {bot_part} <end>"
			ids = tokenizer.tokenize(full_text)

			if isinstance(ids, list) and len(ids) > 2:
				results.append(ids)

		except Exception:
			continue

	return results


def load_prompt_response_dataset(path, tokenizer, num_workers=None):
	with open(path, "r", encoding="utf-8") as f:
		text = f.read()

	samples = [s for s in text.split("<end>") if "<user>" in s and "<bot>" in s]

	if not samples:
		print("No valid samples found.")
		return []

	if num_workers is None:
		num_workers = max(1, mp.cpu_count() - 1)

	print(f"Tokenizing {len(samples)} samples across {num_workers} workers...")

	batch_size = max(1, len(samples) // num_workers)
	batches = [
		(samples[i : i + batch_size], tokenizer)
		for i in range(0, len(samples), batch_size)
	]

	sequences = []

	try:
		with ProcessPoolExecutor(max_workers=num_workers) as executor:
			for batch_result in executor.map(_tokenize_batch, batches):
				sequences.extend(batch_result)
	except Exception:
		print("Multiprocessing unavailable, falling back to single-threaded...")
		for batch in batches:
			sequences.extend(_tokenize_batch(batch))

	print(f"Loaded {len(sequences)} valid sequences.")
	return sequences


def preprocess_dataset(dataset, seq_len, bot_token_id):
    processed = []

    for ids in dataset:
        if bot_token_id not in ids:
            continue  # skip bad samples early

        bot_pos = ids.index(bot_token_id)

        # Ensure <bot> is inside the window
        start = max(0, bot_pos - seq_len // 2)
        seq = ids[start:start + seq_len]

        # If still missing (edge case), skip
        if bot_token_id not in seq:
            continue

        x = seq[:-1]
        y = seq[1:]

        bot_pos = seq.index(bot_token_id)

        mask = [0.0] * bot_pos + [1.0] * (len(seq) - bot_pos - 1)

        target_len = seq_len - 1

        def pad(seq, val=0):
            return seq + [val] * (target_len - len(seq))

        processed.append((pad(x), pad(y), pad(mask, 0.0)))

    return processed

def load_and_prepare_dataset(tokenizer):
	filename = "datasets/" + input("Enter dataset file: ")
	dataset = load_prompt_response_dataset(filename, tokenizer)

	print("Preprocessing dataset...")
	bot_id = tokenizer.tokenize("<bot>")[0]

	processed = preprocess_dataset(dataset, CONFIG["seq_len"], bot_id)

	dataset = [
		(
			torch.tensor(x, dtype=torch.long),
			torch.tensor(y, dtype=torch.long),
			torch.tensor(m, dtype=torch.float32),
		)
		for x, y, m in processed
	]

	return dataset


# -------------------------------
# VERIFY TOKENIZER
# -------------------------------
def verify_tokenizer(tokenizer, dataset):
	print("\nTokenizer verification:")
	print("Vocab size:", len(tokenizer.vocab))

	print("\nDecoded samples:")
	for i in range(min(3, len(dataset))):
		x, y, _ = dataset[i]
		print("\nInput :", tokenizer.decode(x.tolist())[:100])
		print("Target:", tokenizer.decode(y.tolist())[:100])


# -------------------------------
# MODEL
# -------------------------------
def build_model(vocab_size, device):
	model = DeepScratchBitNet(
		vocab_size,
		CONFIG["embed_dim"],
		CONFIG["hidden_dim"],
		CONFIG["num_layers"],
		dropout=CONFIG.get("dropout", 0.1),
		# use_checkpointing=CONFIG.get("use_gradient_checkpointing", False),
	).to(device)

	# try:
	# 	model = torch.compile(model)
	# except Exception as e:
	# 	print(f"Compilation skipped: {e}")

	return model


# -------------------------------
# OPTIMIZER
# -------------------------------
def build_optimizer(model, device):
	# if device.type == "xpu":
	# 	return torch.optim.SGD(
	# 		model.parameters(),
	# 		lr=CONFIG["max_lr"],
	# 		momentum=0.9,
	# 		weight_decay=0.01,
	# 	)
	# else:
	# 	return torch.optim.AdamW(
	# 		model.parameters(),
	# 		lr=CONFIG["max_lr"],
	# 		betas=(0.9, 0.95),
	# 		weight_decay=0.01,
	# 	)
	return torch.optim.AdamW(
		model.parameters(),
		lr=CONFIG["max_lr"],
		betas=(0.9, 0.95),
		weight_decay=0.01,
	)


# -------------------------------
# LR SCHEDULE
# -------------------------------
def get_lr(step, total, CONFIG):
	warmup = CONFIG["warmup_steps"]
	max_lr = CONFIG["max_lr"]
	min_lr = CONFIG["min_lr"]

	if step < warmup:
		return max_lr * step / warmup

	progress = (step - warmup) / (total - warmup)
	progress = min(max(progress, 0.0), 1.0)

	cosine = 0.5 * (1 + math.cos(math.pi * progress))

	lr = min_lr + (max_lr - min_lr) * cosine

	# Prevent LR from becoming uselessly small too early
	return max(lr, min_lr)


def collate_fn(batch, pad_id=0, seq_len=32):
	xs, ys, ms = zip(*batch)

	def pad(seq):
		if len(seq) > seq_len:
			return seq[:seq_len]

		pad_size = seq_len - len(seq)
		pad_tensor = torch.full((pad_size,), pad_id, dtype=seq.dtype)

		return torch.cat([seq, pad_tensor])

	xs = [pad(x) for x in xs]
	ys = [pad(y) for y in ys]
	ms = [pad(m) for m in ms]

	return (
		torch.stack(xs),
		torch.stack(ys),
		torch.stack(ms),
	)

# -------------------------------
# TRAIN
# -------------------------------
def train(model, dataset, optimizer, device, label_smoothing, CONFIG):
	print(CONFIG)
	batch_size = CONFIG["batch_size"]
	epochs = int(input("# of epochs: "))

	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=0,
		pin_memory=False,
		collate_fn=lambda b: collate_fn(b, pad_id=0, seq_len=CONFIG["seq_len"]),
	)

	total_steps = len(loader) * epochs

	# 🔥 MUCH simpler loss
	criterion = nn.CrossEntropyLoss(
		ignore_index=0, label_smoothing=label_smoothing  # padding
	)

	num_samples = len(dataset)
	bar = ProgressBar(total=num_samples)

	global_step = 0
	scaler = torch.amp.GradScaler(device=device.type)

	print_header("BEGIN TRAINING")

	# autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)

	for epoch in range(epochs):
		seen_samples = 0

		for xb, yb, mb in loader:
			xb = xb.to(device)
			yb = yb.to(device)
			mb = mb.to(device)

			lr = get_lr(global_step, total_steps, CONFIG)
			optimizer.param_groups[0]["lr"] = lr

			optimizer.zero_grad(set_to_none=True)

			# with autocast_ctx:
			logits = model(xb)
			logits = torch.clamp(logits, -20, 20)

			loss_raw = criterion(
				logits.reshape(-1, logits.size(-1)),
				yb.reshape(-1),
			)
			mask = mb.reshape(-1)
			loss = (loss_raw * mask).sum() / mask.sum().clamp(min=1.0)

			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)

			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

			scaler.step(optimizer)
			scaler.update()

			global_step += 1
			seen_samples += xb.size(0)

			if global_step % 16 == 0:
				bar.update(
					step=seen_samples,
					loss=loss.item(),
					lr=lr,
					epoch=epoch + 1,
					opt_step=global_step,
				)

			if global_step % 100 == 0 and device.type == "xpu":
				torch.xpu.empty_cache()

		bar.update(
			step=seen_samples,
			loss=loss.item(),
			lr=lr,
			epoch=epoch + 1,
			opt_step=global_step,
		)
		bar.close()

	return model


# -------------------------------
# SAVE
# -------------------------------
def save_and_export(model):
	for module in model.modules():
		if isinstance(module, BitLinear):
			module.quantize()

	os.makedirs("weights", exist_ok=True)
	torch.save(model.state_dict(), "weights/model.pt")

	model = model.to("cpu")
	export_packed_bitnet(model)

	print("Model saved + exported")


# -------------------------------
# MAIN
# -------------------------------
def main():
	device = setup_device()

	tokenizer = load_tokenizer(CONFIG)
	dataset = load_and_prepare_dataset(tokenizer)

	verify_tokenizer(tokenizer, dataset)

	vocab_size = len(tokenizer.vocab)

	model = build_model(vocab_size, device)
	optimizer = build_optimizer(model, device)

	if device.type == "xpu":
		import intel_extension_for_pytorch as ipex

		model, optimizer = ipex.optimize(
			model, optimizer=optimizer, dtype=torch.bfloat16
		)

	model = train(model, dataset, optimizer, device, 0.1, CONFIG)

	save_and_export(model)


if __name__ == "__main__":
	main()
