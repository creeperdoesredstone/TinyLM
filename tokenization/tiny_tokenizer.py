import regex as re
import json
from collections import Counter


def clean_corpus(raw_text):
    lines = raw_text.split("\n")
    cleaned_lines = [line for line in lines if "\u0019" not in line]
    text = "\n".join(cleaned_lines)
    # keep only printable ASCII + newlines
    text = "".join(char for char in text if 32 <= ord(char) <= 126 or char == "\n")
    return text


class TinyLMTokenizer:
    def __init__(self, vocab_size=1024):
        self.vocab_size = vocab_size

        self.special_tokens = [
            "<pad>",
            "<unk>",
            "<user>",
            "<bot>",
            "<end>",
            "!!!",
            "...",
            "```",
            "???",
            "---",
        ]
        self.num_special = len(self.special_tokens)

        self.base_bytes = list(range(32, 127))
        self.num_base = len(self.base_bytes)  # should be 95

        # raw byte value -> vocab ID
        self.byte_to_id = {}
        self.vocab = {}
        self._build_initial_vocab()

        self.merges = {}

        special_pat = "|".join(
            sorted(
                (r" ?" + re.escape(t) for t in self.special_tokens),
                key=len,
                reverse=True,
            )
        )
        self.pat = re.compile(
            special_pat
            + r"""|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.IGNORECASE,
        )

    def _build_initial_vocab(self):
        self.vocab = {}
        self.byte_to_id = {}

        for i, token in enumerate(self.special_tokens):
            self.vocab[i] = token.encode("utf-8")

        for i, byte_val in enumerate(self.base_bytes):
            vocab_id = self.num_special + i
            self.vocab[vocab_id] = bytes([byte_val])
            self.byte_to_id[byte_val] = vocab_id

    def _encode_bytes_to_ids(self, raw_bytes):
        ids = []
        for b in raw_bytes:
            if b in self.byte_to_id:
                ids.append(self.byte_to_id[b])
        return ids

    def train(self, text):
        text = text.lower()
        text = clean_corpus(text)
        print(f"Training BPE... Target: {self.vocab_size}")

        raw_words = re.findall(self.pat, text)
        words = [w for w in raw_words if w not in self.special_tokens]
        word_counts = Counter(words)

        ids_dict = {}
        for w, count in word_counts.items():
            ids = self._encode_bytes_to_ids(w.encode("utf-8"))
            if ids:
                ids_dict[tuple(ids)] = ids_dict.get(tuple(ids), 0) + count

        current_vocab_size = self.num_special + self.num_base
        num_merges = self.vocab_size - current_vocab_size

        for i in range(num_merges):
            stats = Counter()
            for ids, count in ids_dict.items():
                for pair in zip(ids, ids[1:]):
                    stats[pair] += count

            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            new_idx = current_vocab_size + i
            self.merges[best_pair] = new_idx

            new_ids_dict = {}
            for ids, count in ids_dict.items():
                new_ids = []
                j = 0
                while j < len(ids):
                    if j < len(ids) - 1 and (ids[j], ids[j + 1]) == best_pair:
                        new_ids.append(new_idx)
                        j += 2
                    else:
                        new_ids.append(ids[j])
                        j += 1
                new_ids_dict[tuple(new_ids)] = count
            ids_dict = new_ids_dict

            if (i + 1) % 100 == 0:
                print(f"Merge {i+1}/{num_merges} complete: {best_pair} -> {new_idx}")

        for (p1, p2), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p1] + self.vocab[p2]

        print(f"Training complete. Final vocab size: {len(self.vocab)}")

    def tokenize(self, text):
        text = text.lower()
        chunks = re.findall(self.pat, text)

        final_ids = []
        special_map = {t.encode("utf-8"): i for i, t in enumerate(self.special_tokens)}

        for chunk in chunks:
            chunk_bytes = chunk.encode("utf-8")
            if chunk_bytes in special_map:
                final_ids.append(special_map[chunk_bytes])
            elif chunk.strip().encode("utf-8") in special_map:
                stripped = chunk.strip()

                # preserve leading space(s)
                if chunk.startswith(" "):
                    space_ids = self._encode_bytes_to_ids(b" ")
                    final_ids.extend(space_ids)

                final_ids.append(special_map[stripped.encode("utf-8")])
            else:
                ids = self._encode_bytes_to_ids(chunk_bytes)

                for pair, idx in self.merges.items():
                    new_ids = []
                    i = 0
                    while i < len(ids):
                        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                            new_ids.append(idx)
                            i += 2
                        else:
                            new_ids.append(ids[i])
                            i += 1
                    ids = new_ids
                final_ids.extend(ids)
        return final_ids

    def decode(self, ids):
        b_arr = b"".join(self.vocab.get(idx, b"") for idx in ids)
        return b_arr.decode("utf-8", errors="replace")

    def save(self, path):
        serializable_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        readable_vocab = {}
        for idx, token_bytes in self.vocab.items():
            try:
                readable_vocab[idx] = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                readable_vocab[idx] = str(token_bytes)

        data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "base_bytes": self.base_bytes,
            "merges": serializable_merges,
            "vocab_map": readable_vocab,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Tokenizer saved to {path}")

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.special_tokens = data["special_tokens"]
        self.num_special = len(self.special_tokens)
        self.base_bytes = data.get("base_bytes", list(range(32, 127)))
        self.num_base = len(self.base_bytes)

        self.merges = {
            tuple(map(int, k.split(","))): v for k, v in data["merges"].items()
        }

        self._build_initial_vocab()
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        for (p1, p2), idx in sorted_merges:
            self.vocab[idx] = self.vocab[p1] + self.vocab[p2]

        print(f"Tokenizer loaded from {path}. Vocab size: {len(self.vocab)}")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["pat"]  # compiled regex isn't picklable
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        special_pat = "|".join(
            sorted(
                (r" ?" + re.escape(t) for t in self.special_tokens),
                key=len,
                reverse=True,
            )
        )
        self.pat = re.compile(
            special_pat
            + r"""|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.IGNORECASE,
        )
