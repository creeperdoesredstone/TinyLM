"""Standalone training + generation script invoked by test_tinytok.py in a subprocess."""

import sys
import math
import os
import json
import torch

from config128k import CONFIG
from train_model_with_tinytok import (
    setup_device,
    build_optimizer,
    train,
    load_tokenizer,
    load_prompt_response_dataset,
    preprocess_dataset,
)
from model.bitnet import DeepScratchBitNet
from utils.tinytok_generation import generate
import builtins


def main():
    num_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

    device = setup_device()
    tokenizer = load_tokenizer(CONFIG)
    vocab_size = len(tokenizer.vocab)

    model = (
        DeepScratchBitNet(
            vocab_size,
            CONFIG["embed_dim"],
            CONFIG["hidden_dim"],
            CONFIG["num_layers"],
            CONFIG.get("dropout", 0.1),
        )
        .to(device)
        .float()
    )

    optimizer = build_optimizer(model, device)

    model = torch.compile(model, backend="aot_eager", fullgraph=False, dynamic=False)

    num_samples = 14500
    batch_size = CONFIG["batch_size"]
    loader_len = math.ceil(num_samples / batch_size)
    epochs = max(1, math.ceil(num_steps / max(1, loader_len)))

    raw = load_prompt_response_dataset("datasets/combined_chat_small.txt", tokenizer)
    raw = raw[:num_samples]
    print("[DEBUG] First raw sample:", raw[0][:50])
    bot_tok_id = tokenizer.tokenize("<bot>")[0]
    print(f"Bot Token ID: {bot_tok_id}")
    processed = preprocess_dataset(raw, CONFIG["seq_len"], bot_tok_id)
    print("Raw samples:", len(raw))
    print("Processed samples:", len(processed))

    dataset = []
    for x, y, m in processed:
        if not isinstance(x, list) or not isinstance(y, list):
            continue
        if not all(isinstance(i, int) for i in x + y):
            continue

        dataset.append(
            (
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(m, dtype=torch.float32),
            )
        )
    print("Final dataset size:", len(dataset))

    builtins.input = lambda prompt="": str(epochs)
    model = train(model, dataset, optimizer, device, CONFIG["label_smoothing"], CONFIG)

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/test_model.pt")
    print("Checkpoint saved.")

    # Run generation tests in this subprocess while XPU is still healthy
    model.eval()
    results = {
        "basic": None,
        "sampling": {},
        "quality": {},
        "examples": [],
    }

    # Basic generation
    try:
        with torch.no_grad():
            out = generate(
                model,
                tokenizer,
                "<user> hello! <bot> ",
                device,
                length=20,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
            )
        results["basic"] = out
    except Exception as e:
        results["basic"] = f"ERROR: {e}"

    # Sampling parameters
    for name, temp, top_p in [
        ("Conservative", 0.5, 0.9),
        ("Balanced", 1.0, 0.95),
        ("Creative", 1.5, 0.99),
    ]:
        try:
            with torch.no_grad():
                out = generate(
                    model,
                    tokenizer,
                    "<user> how are you? <bot> ",
                    device,
                    length=15,
                    temperature=temp,
                    top_p=top_p,
                    top_k=0,
                )
            results["sampling"][name] = out
        except Exception as e:
            results["sampling"][name] = f"ERROR: {e}"

    # Quality metrics
    for prompt in [
        "<user> hello <bot> ",
        "<user> what is your name <bot> ",
        "<bot> i am good ",
    ]:
        try:
            with torch.no_grad():
                out = generate(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    length=20,
                    temperature=0.8,
                    top_p=0.92,
                    top_k=0,
                )
            results["quality"][prompt] = out
        except Exception as e:
            results["quality"][prompt] = f"ERROR: {e}"

    # Human evaluation examples
    for prompt, description in [
        ("<user> hello, how are you today? <bot> ", "Standard greeting"),
        ("<user> what is ", "Incomplete query"),
        ("<bot> the answer is ", "Statement continuation"),
        ("<user> tell me about ", "Information request"),
    ]:
        try:
            with torch.no_grad():
                out = generate(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    temperature=0.9,
                    top_p=0.92,
                    top_k=0,
                    repetition_penalty=1.1,
                    freq_penalty=0.1,
                    length=40,
                )
            results["examples"].append(
                {
                    "description": description,
                    "prompt": prompt,
                    "output": out,
                }
            )
        except Exception as e:
            results["examples"].append(
                {
                    "description": description,
                    "prompt": prompt,
                    "output": f"ERROR: {e}",
                }
            )

    with open("weights/generation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Generation results saved.")


if __name__ == "__main__":
    main()
