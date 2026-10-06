import base64
import hashlib
import hmac
import logging
import os
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

_KEY_BYTES = 32
_NONCE_BYTES = 12
_KEY_VERSION = "v1"


@dataclass(frozen=True)
class EncryptedValue:
    ciphertext: str
    nonce: str
    encrypted_data_key: str
    key_version: str


def _b64_encode(value: bytes) -> str:
    # 使用 URL 安全 Base64，便于后续存入数据库或 JSON。
    return base64.urlsafe_b64encode(value).decode("ascii")


def _b64_decode(value: str) -> bytes:
    # 只在解密入口解码，避免业务层接触二进制密文。
    return base64.urlsafe_b64decode(value.encode("ascii"))


def _settings_secret(name: str) -> str:
    # 优先读取当前环境变量，确保测试 monkeypatch 和运行期配置生效。
    env_value = os.getenv(name)
    if env_value:
        return env_value
    # 回退到配置对象，保持与 Task 1 的 settings 接口兼容。
    return getattr(get_settings(), name, "")


def _require_secret(name: str) -> bytes:
    # 密钥缺失直接失败，不使用弱默认值。
    secret = _settings_secret(name)
    if not secret:
        logger.error("crypto secret missing", extra={"secret_name": name})
        raise ValueError(f"{name} is required")
    # 仅返回 UTF-8 字节，不记录密钥内容。
    return secret.encode("utf-8")


def _derive_key(secret: bytes, purpose: str, usage: str) -> bytes:
    # HKDF 将主密钥按用途隔离，避免不同业务复用同一派生密钥。
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_BYTES,
        salt=f"legal-rag:{usage}".encode("utf-8"),
        info=f"purpose:{purpose}".encode("utf-8"),
    )
    # 返回固定长度 AES-256/HMAC 派生密钥。
    return hkdf.derive(secret)


def encrypt_text(plaintext: str, purpose: str) -> EncryptedValue:
    # 明文仅在函数入参和加密调用内短暂存在，绝不写入日志。
    master_key = _require_secret("APP_MASTER_KEY")
    # 按业务目的派生 AES-GCM 密钥。
    data_key = _derive_key(master_key, purpose, "aes-gcm")
    # AES-GCM 推荐 96-bit nonce，每次加密必须随机生成。
    nonce = os.urandom(_NONCE_BYTES)
    # AAD 绑定用途与版本，防止密文被挪用到其他目的解密。
    associated_data = f"{_KEY_VERSION}:{purpose}".encode("utf-8")
    # cryptography 库负责认证加密和 tag 生成。
    ciphertext = AESGCM(data_key).encrypt(nonce, plaintext.encode("utf-8"), associated_data)
    logger.info("text encrypted", extra={"purpose": purpose, "key_version": _KEY_VERSION})
    # encrypted_data_key 当前记录派生方案版本，不存储真实数据密钥。
    return EncryptedValue(
        ciphertext=_b64_encode(ciphertext),
        nonce=_b64_encode(nonce),
        encrypted_data_key="derived-hkdf-sha256",
        key_version=_KEY_VERSION,
    )


def decrypt_text(value: EncryptedValue, purpose: str) -> str:
    # 解密使用与加密相同的主密钥和用途派生密钥。
    master_key = _require_secret("APP_MASTER_KEY")
    data_key = _derive_key(master_key, purpose, "aes-gcm")
    associated_data = f"{value.key_version}:{purpose}".encode("utf-8")
    # 仅解码 nonce 与密文，不记录任何密文或明文。
    plaintext = AESGCM(data_key).decrypt(_b64_decode(value.nonce), _b64_decode(value.ciphertext), associated_data)
    logger.info("text decrypted", extra={"purpose": purpose, "key_version": value.key_version})
    return plaintext.decode("utf-8")


def hmac_digest(value: str, purpose: str) -> str:
    # HMAC 使用独立环境密钥，避免与加密主密钥混用。
    hmac_key = _require_secret("APP_HMAC_KEY")
    digest_key = _derive_key(hmac_key, purpose, "hmac")
    # 输出十六进制摘要，可稳定用于去重或查找，不暴露原文。
    digest = hmac.new(digest_key, value.encode("utf-8"), hashlib.sha256).hexdigest()
    logger.info("hmac digest created", extra={"purpose": purpose})
    return digest
