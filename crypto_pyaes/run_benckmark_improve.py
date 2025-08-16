#!/usr/bin/env python
"""
Pure-Python Implementation of the AES block-cipher.

Benchmark AES in CTR mode using the pyaes module.
"""

import pyperf
import pyaes

# 23,000 bytes
CLEARTEXT = b"This is a test. What could possibly go wrong? " * 500

# 128-bit key (16 bytes)
KEY = b'\xa1\xf6%\x8c\x87}_\xcd\x89dHE8\xbf\xc9,'


def bench_pyaes(loops):
    range_it = range(loops)
    t0 = pyperf.perf_counter()

    for _ in range_it:
        # Create one AES object for encryption and one for decryption
        aes_enc = pyaes.AESModeOfOperationCTR(KEY)
        ciphertext = aes_enc.encrypt(CLEARTEXT)

        aes_dec = pyaes.AESModeOfOperationCTR(KEY)
        plaintext = aes_dec.decrypt(ciphertext)

        if plaintext != CLEARTEXT:
            raise Exception("decrypt error!")

    dt = pyperf.perf_counter() - t0
    return dt


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.metadata['description'] = (
        "Pure-Python Implementation of the AES block-cipher"
    )
    runner.bench_time_func('crypto_pyaes', bench_pyaes)
