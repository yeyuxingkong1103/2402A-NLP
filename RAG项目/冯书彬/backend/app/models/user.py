from dataclasses import dataclass
from datetime import datetime


@dataclass
class User:
    # 用户主键使用内部随机 ID，不暴露手机号。
    id: str
    # 手机号密文和加密元数据分列保存，数据库中不落明文手机号。
    encrypted_phone: str
    phone_nonce: str
    phone_encrypted_data_key: str
    phone_key_version: str
    # HMAC 仅用于稳定查找和去重，不可反推出手机号。
    phone_hmac: str
    # 协议确认字段供后续合规流程复用，本任务只设置默认版本。
    agreement_version: str
    privacy_policy_version: str
    adult_confirmed: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class DeviceSession:
    # 每个设备会话对应一个 refresh token 摘要。
    id: str
    user_id: str
    refresh_token_hash: str
    device_identifier: str
    user_agent: str
    login_at: datetime
    last_activity_at: datetime
    province_location: str | None
    expires_at: datetime
    revoked: bool = False
    revoked_at: datetime | None = None
