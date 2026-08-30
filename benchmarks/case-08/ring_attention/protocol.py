"""Binary protocol shared by the case-8 USB coordinator and ESP32 workers."""

from __future__ import annotations

import enum
import struct
import zlib
from dataclasses import dataclass


MAGIC = 0x38474E52  # "RNG8" when viewed as little-endian bytes.
VERSION = 1
REPLY_BIT = 0x80
HEADER = struct.Struct("<IBBHIII")
STATUS = struct.Struct("<HHI")


class Message(enum.IntEnum):
    HELLO = 1
    STAGE_BEGIN = 2
    STAGE_CHUNK = 3
    STAGE_COMMIT = 4
    SET_NORM = 5
    RUN_NORM = 6
    RUN_LINEAR = 7
    ATTN_BEGIN = 8
    ATTN_BLOCK = 9
    ATTN_END = 10
    PING = 11


class Status(enum.IntEnum):
    OK = 0
    BAD_HEADER = 1
    BAD_LENGTH = 2
    BAD_CRC = 3
    BAD_STATE = 4
    BAD_SHAPE = 5
    FLASH_ERROR = 6
    INTERNAL_ERROR = 7


@dataclass(frozen=True)
class Frame:
    kind: int
    request_id: int
    flags: int
    payload: bytes


def encode_frame(kind: int, request_id: int, payload: bytes = b"", flags: int = 0) -> bytes:
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    return HEADER.pack(
        MAGIC, VERSION, int(kind), flags, request_id, len(payload), checksum
    ) + payload


def decode_header(data: bytes) -> tuple[int, int, int, int, int]:
    if len(data) != HEADER.size:
        raise ValueError(f"header must be {HEADER.size} bytes")
    magic, version, kind, flags, request_id, length, checksum = HEADER.unpack(data)
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08x}")
    if version != VERSION:
        raise ValueError(f"unsupported protocol version {version}")
    return kind, flags, request_id, length, checksum


def decode_frame(data: bytes) -> Frame:
    kind, flags, request_id, length, checksum = decode_header(data[: HEADER.size])
    payload = data[HEADER.size :]
    if len(payload) != length:
        raise ValueError(f"payload length {len(payload)} != declared {length}")
    if zlib.crc32(payload) & 0xFFFFFFFF != checksum:
        raise ValueError("payload CRC mismatch")
    return Frame(kind, request_id, flags, payload)


def decode_status(payload: bytes) -> tuple[Status, int, bytes]:
    if len(payload) < STATUS.size:
        raise ValueError("short status payload")
    code, _reserved, elapsed_us = STATUS.unpack_from(payload)
    return Status(code), elapsed_us, payload[STATUS.size :]
