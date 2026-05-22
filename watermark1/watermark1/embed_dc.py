#!/usr/bin/env python3
import argparse
import os
from typing import List

import cv2
import numpy as np

HEADER_BITS = 32


def bytes_to_bits(data):
    bits = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def int_to_bits(value, width):
    bits = []
    for i in range(width - 1, -1, -1):
        bits.append((value >> i) & 1)
    return bits


def nearest_q_with_bit(q, bit):
    if q % 2 == bit:
        return q

    c1 = q - 1
    c2 = q + 1

    candidates = []
    if c1 % 2 == bit:
        candidates.append(c1)
    if c2 % 2 == bit:
        candidates.append(c2)

    if not candidates:
        return q

    return min(candidates, key=lambda x: abs(x - q))


def iter_block_positions(height, width):
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            yield y, x


def embed_bits_in_image(gray, bits, delta):
    h, w = gray.shape
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8

    if h8 < 8 or w8 < 8:
        raise ValueError("Ảnh quá nhỏ, không đủ block 8x8 để nhúng.")

    out = gray.copy().astype(np.float32)
    capacity = (h8 // 8) * (w8 // 8)

    if len(bits) > capacity:
        raise ValueError(
            "Thông điệp quá dài. Cần {} bit nhưng ảnh chỉ chứa được {} bit.".format(
                len(bits), capacity
            )
        )

    bit_idx = 0
    for y, x in iter_block_positions(h8, w8):
        if bit_idx >= len(bits):
            break

        block = out[y:y + 8, x:x + 8].copy()
        block_shifted = block - 128.0
        dct_block = cv2.dct(block_shifted)

        dc = float(dct_block[0, 0])
        q = int(round(dc / delta))
        q_new = nearest_q_with_bit(q, bits[bit_idx])

        dct_block[0, 0] = q_new * delta
        rec = cv2.idct(dct_block) + 128.0
        out[y:y + 8, x:x + 8] = np.clip(np.round(rec), 0, 255)

        bit_idx += 1

    return out.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Nhúng watermark text vào ảnh bằng DCT/DC parity quantization watermarking."
    )
    parser.add_argument("input_image", help="Ảnh đầu vào, ví dụ: frames/frame_0001.png")
    parser.add_argument("message", help="Thông điệp cần nhúng, ví dụ: Tung")
    parser.add_argument(
        "-o",
        "--output",
        default="output/watermarked_frame.png",
        help="Ảnh đầu ra (mặc định: output/watermarked_frame.png)"
    )
    parser.add_argument(
        "-d",
        "--delta",
        type=float,
        default=40.0,
        help="Bước lượng tử delta (mặc định: 40.0)"
    )
    args = parser.parse_args()

    gray = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError("Không đọc được ảnh: {}".format(args.input_image))

    msg_bytes = args.message.encode("utf-8")
    header_bits = int_to_bits(len(msg_bytes), HEADER_BITS)
    payload_bits = bytes_to_bits(msg_bytes)
    bits = header_bits + payload_bits

    h, w = gray.shape
    capacity = (h // 8) * (w // 8)

    print("[+] Kích thước ảnh: {}x{}".format(w, h))
    print("[+] Sức chứa lý thuyết: {} bit".format(capacity))
    print("[+] Thông điệp: {}".format(args.message))
    print("[+] Số byte thông điệp: {}".format(len(msg_bytes)))
    print("[+] Tổng số bit cần nhúng: {}".format(len(bits)))
    print("[+] Delta: {}".format(args.delta))

    watermarked = embed_bits_in_image(gray, bits, args.delta)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    ok = cv2.imwrite(args.output, watermarked)
    if not ok:
        raise RuntimeError("Không ghi được ảnh đầu ra: {}".format(args.output))

    print("[+] Đã lưu ảnh chứa watermark: {}".format(args.output))


if __name__ == "__main__":
    main()

