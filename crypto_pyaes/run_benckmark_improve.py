#!/usr/bin/env python
"""
AES block-cipher benchmark using the fast C-based cryptography library.

Benchmark AES in CTR mode with the same payload as the original pyaes example.
"""

import pyperf
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# 23,000 bytes
CLEARTEXT = b"This is a test. What could possibly go wrong? " * 500

# 128-bit key (16 bytes)
KEY = b'\xa1\xf6%\x8c\x87}_\xcd\x89dHE8\xbf\xc9,'

# 16-byte nonce for CTR mode (can be all zeros for benchmark purposes)
NONCE = b'\x00' * 16


def bench_fast_aes(loops):
    range_it = range(loops)
    t0 = pyperf.perf_counter()

    for _ in range_it:
        # Create cipher object for this iteration
        cipher = Cipher(algorithms.AES(KEY), modes.CTR(NONCE), backend=default_backend())

        # Encrypt
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(CLEARTEXT) + encryptor.finalize()

        # Decrypt
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        # Verify correctness
        if plaintext != CLEARTEXT:
            raise Exception("decrypt error!")

    dt = pyperf.perf_counter() - t0
    return dt


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.metadata['description'] = (
        "AES benchmark in CTR mode using fast C-based cryptography library"
    )
    runner.bench_time_func('crypto_fast_aes', bench_fast_aes)
