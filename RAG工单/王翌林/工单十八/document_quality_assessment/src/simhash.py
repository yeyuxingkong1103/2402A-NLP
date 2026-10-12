# -*- coding: utf-8 -*-
"""SimHash：用于识别高相似度文档（不同版本的同一份文件），输出“待确认版本冲突列表”。"""
import hashlib


class SimHash:
    def __init__(self, hash_bits=64):
        self.hash_bits = hash_bits

    @staticmethod
    def _tokenize(text):
        # 按字符 3-gram 分词（对中英文均鲁棒）
        t = text.strip().lower()
        if len(t) < 3:
            return [t]
        return [t[i:i + 3] for i in range(len(t) - 2)]

    def hash(self, text):
        v = [0] * self.hash_bits
        for tok in self._tokenize(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            for i in range(self.hash_bits):
                v[i] += 1 if (h >> i) & 1 else -1
        fingerprint = 0
        for i in range(self.hash_bits):
            if v[i] >= 0:
                fingerprint |= (1 << i)
        return fingerprint

    @staticmethod
    def hamming(a, b):
        return bin(a ^ b).count("1")
