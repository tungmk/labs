#!/usr/bin/env python3
import argparse

import cv2
import numpy as np

HEADER_BITS = 32


def bits_to_int(bits):
    value = 0
    for b in bits:
        value = (value << 1) | b
    return value


def bits_to_bytes(bits):
    if len(bits) % 8 != 0:
        raise ValueError("Số bit payload không chia hết cho 8.")
    out = bytearray()
    for i in range(0, len(bits), 8):
        b = 0
        for bit in bits[i:i + 8]:
            b = (b << 1) | bit
        out.append(b)
    return bytes(out)


def iter_block_positions(height, width):
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            yield y, x


def extract_all_bits(gray, delta):
    h, w = gray.shape
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8

    bits = []
    img = gray.astype(np.float32)

    for y, x in iter_block_positions(h8, w8):
        block = img[y:y + 8, x:x + 8]
        block_shifted = block - 128.0
        dct_block = cv2.dct(block_shifted)

        dc = float(dct_block[0, 0])
        q = int(round(dc / delta))
        bit = q % 2
        bits.append(bit)

    return bits


def main():
    parser = argparse.ArgumentParser(
        description="Trích watermark text từ ảnh bằng DCT/DC parity quantization watermarking."
    )
    parser.add_argument("input_image", help="Ảnh cần detect, ví dụ: output/watermarked_frame.png")
    parser.add_argument(
        "-d",
        "--delta",
        type=float,
        default=40.0,
        help="Bước lượng tử delta (mặc định: 40.0)"
    )
    parser.add_argument(
        "--show-bits",
        action="store_true",
        help="In thêm header bit và payload bit"
    )
    args = parser.parse_args()

    gray = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError("Không đọc được ảnh: {}".format(args.input_image))

    h, w = gray.shape
    capacity = (h // 8) * (w // 8)

    print("[+] Kích thước ảnh: {}x{}".format(w, h))
    print("[+] Sức chứa lý thuyết: {} bit".format(capacity))
    print("[+] Delta: {}".format(args.delta))

    all_bits = extract_all_bits(gray, args.delta)

    if len(all_bits) < HEADER_BITS:
        raise ValueError("Ảnh không đủ dữ liệu để đọc header 32 bit.")

    header_bits = all_bits[:HEADER_BITS]
    msg_len_bytes = bits_to_int(header_bits)
    payload_bits_needed = msg_len_bytes * 8
    total_bits_needed = HEADER_BITS + payload_bits_needed

    print("[+] Độ dài thông điệp đọc từ header: {} byte".format(msg_len_bytes))
    print("[+] Tổng số bit cần đọc: {}".format(total_bits_needed))

    if total_bits_needed > len(all_bits):
        raise ValueError(
            "Header yêu cầu {} bit nhưng chỉ đọc được {} bit.".format(
                total_bits_needed, len(all_bits)
            )
        )

    payload_bits = all_bits[HEADER_BITS:total_bits_needed]

    if args.show_bits:
        print("[DEBUG] Header bits : {}".format("".join(map(str, header_bits))))
        print("[DEBUG] Payload bits: {}".format("".join(map(str, payload_bits))))

    msg_bytes = bits_to_bytes(payload_bits)

    try:
        message = msg_bytes.decode("utf-8")
    except UnicodeDecodeError:
        message = msg_bytes.decode("utf-8", errors="replace")

    print("[+] Message bytes: {}".format(msg_bytes))
    print("[+] Message text : {}".format(message))


if __name__ == "__main__":
    main()

