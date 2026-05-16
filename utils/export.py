import os

SAFE_CHARS = [chr(i) for i in range(33, 127)]  # printable ASCII
BASE = len(SAFE_CHARS)
CHAR_TO_VAL = {c: i for i, c in enumerate(SAFE_CHARS)}

def encode_baseN(data: bytes) -> str:
    num = int.from_bytes(data, "big")

    if num == 0:
        return SAFE_CHARS[0]

    chars = []
    while num > 0:
        num, rem = divmod(num, BASE)
        chars.append(SAFE_CHARS[rem])

    return "".join(reversed(chars))


def pack_base3_fixed(weights, chunk_size=162):
    packed = bytearray()

    for i in range(0, len(weights), chunk_size):
        chunk = weights[i:i+chunk_size]

        value = 0
        for w in chunk:
            value = value * 3 + w

        byte_len = (value.bit_length() + 7) // 8
        packed.append(byte_len)
        packed.extend(value.to_bytes(byte_len, "big"))

        # store length so we can decode later

    return bytes(packed)

def export_packed_bitnet(
    model,
    weight_file="weights/packed_weights.txt",
    map_file="weights/model_map.txt"
):
    os.makedirs(os.path.dirname(weight_file), exist_ok=True)

    print("--- Exporting Weights ---")

    with open(weight_file, "w", encoding="utf-8") as fw, \
         open(map_file, "w", encoding="utf-8") as fm:

        fm.write("Layer_Name, Offset, Params, ChunkSize\n")

        offset = 0
        chunk_size = 162

        for name, param in model.state_dict().items():
            weights = param.detach().cpu().numpy().flatten()
            print(f"\rWriting: {name}", end="", flush=True)

            mapped = []
            for w in weights:
                if w >= 0.5:
                    mapped.append(2)
                elif w <= -0.5:
                    mapped.append(0)
                else:
                    mapped.append(1)

            fm.write(f"{name},{offset},{len(mapped)},{chunk_size}\n")

            packed_bytes = pack_base3_fixed(mapped, chunk_size)
            encoded = encode_baseN(packed_bytes)

            fw.write(encoded + "\n")

            offset += len(encoded)

    print("\nSuccess!")
    print(f"File size: {os.path.getsize(weight_file)/(1024*1024):.2f} MB")
