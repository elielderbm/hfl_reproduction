import os, struct

def _rotl32(v, n):
    return ((v << n) & 0xffffffff) | (v >> (32 - n))

def _salsa20_hash(b: bytes):
    if len(b) != 64:
        raise ValueError(f"Salsa20 precisa de 64 bytes de entrada, recebido {len(b)}")
    x = list(struct.unpack("<16I", b))
    z = x[:]
    for _ in range(10):
        # Column rounds
        z[4] ^= _rotl32((z[0] + z[12]) & 0xffffffff, 7)
        z[8] ^= _rotl32((z[4] + z[0]) & 0xffffffff, 9)
        z[12] ^= _rotl32((z[8] + z[4]) & 0xffffffff, 13)
        z[0] ^= _rotl32((z[12] + z[8]) & 0xffffffff, 18)

        z[9] ^= _rotl32((z[5] + z[1]) & 0xffffffff, 7)
        z[13] ^= _rotl32((z[9] + z[5]) & 0xffffffff, 9)
        z[1] ^= _rotl32((z[13] + z[9]) & 0xffffffff, 13)
        z[5] ^= _rotl32((z[1] + z[13]) & 0xffffffff, 18)

        z[14] ^= _rotl32((z[10] + z[6]) & 0xffffffff, 7)
        z[2] ^= _rotl32((z[14] + z[10]) & 0xffffffff, 9)
        z[6] ^= _rotl32((z[2] + z[14]) & 0xffffffff, 13)
        z[10] ^= _rotl32((z[6] + z[2]) & 0xffffffff, 18)

        z[3] ^= _rotl32((z[15] + z[11]) & 0xffffffff, 7)
        z[7] ^= _rotl32((z[3] + z[15]) & 0xffffffff, 9)
        z[11] ^= _rotl32((z[7] + z[3]) & 0xffffffff, 13)
        z[15] ^= _rotl32((z[11] + z[7]) & 0xffffffff, 18)

        # Row rounds
        z[1] ^= _rotl32((z[0] + z[3]) & 0xffffffff, 7)
        z[2] ^= _rotl32((z[1] + z[0]) & 0xffffffff, 9)
        z[3] ^= _rotl32((z[2] + z[1]) & 0xffffffff, 13)
        z[0] ^= _rotl32((z[3] + z[2]) & 0xffffffff, 18)

        z[6] ^= _rotl32((z[5] + z[4]) & 0xffffffff, 7)
        z[7] ^= _rotl32((z[6] + z[5]) & 0xffffffff, 9)
        z[4] ^= _rotl32((z[7] + z[6]) & 0xffffffff, 13)
        z[5] ^= _rotl32((z[4] + z[7]) & 0xffffffff, 18)

        z[11] ^= _rotl32((z[10] + z[9]) & 0xffffffff, 7)
        z[8] ^= _rotl32((z[11] + z[10]) & 0xffffffff, 9)
        z[9] ^= _rotl32((z[8] + z[11]) & 0xffffffff, 13)
        z[10] ^= _rotl32((z[9] + z[8]) & 0xffffffff, 18)

        z[12] ^= _rotl32((z[15] + z[14]) & 0xffffffff, 7)
        z[13] ^= _rotl32((z[12] + z[15]) & 0xffffffff, 9)
        z[14] ^= _rotl32((z[13] + z[12]) & 0xffffffff, 13)
        z[15] ^= _rotl32((z[14] + z[13]) & 0xffffffff, 18)
    return struct.pack("<16I", *[(z[i] + x[i]) & 0xffffffff for i in range(16)])

def _expand32(key: bytes, nonce: bytes, counter: bytes):
    if len(key) != 32:
        raise ValueError(f"Chave inválida: esperado 32 bytes, recebido {len(key)}")
    if len(nonce) != 8:
        raise ValueError(f"Nonce inválido: esperado 8 bytes, recebido {len(nonce)}")
    sigma = b"expand 32-byte k"
    return sigma[:4] + key[:16] + sigma[4:8] + nonce + counter + sigma[8:12] + key[16:] + sigma[12:16]

def salsa20_stream(key: bytes, nonce: bytes, length: int, counter: int = 0):
    output = b""
    while len(output) < length:
        block = _salsa20_hash(_expand32(key, nonce, counter.to_bytes(8, "little")))
        output += block
        counter += 1
    return output[:length]

def salsa20_xor(key: bytes, nonce: bytes, data: bytes, counter: int = 0):
    ks = salsa20_stream(key, nonce, len(data), counter)
    return bytes(a ^ b for a, b in zip(data, ks))

def encrypt(key_hex: str, data: bytes) -> bytes:
    key = bytes.fromhex(key_hex)
    if len(key) != 32:
        raise ValueError(f"Chave inválida: esperado 32 bytes (64 hex), recebido {len(key)}")
    nonce = os.urandom(8)  # nonce sempre 8 bytes
    ct = salsa20_xor(key, nonce, data, counter=0)
    return nonce + ct

def decrypt(key_hex: str, enc: bytes) -> bytes:
    key = bytes.fromhex(key_hex)
    if len(key) != 32:
        raise ValueError(f"Chave inválida: esperado 32 bytes (64 hex), recebido {len(key)}")
    nonce, ct = enc[:8], enc[8:]
    return salsa20_xor(key, nonce, ct, counter=0)
