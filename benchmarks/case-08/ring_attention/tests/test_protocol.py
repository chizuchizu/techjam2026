from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol import HEADER, MAGIC, Message, decode_frame, encode_frame


class ProtocolTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        encoded = encode_frame(Message.PING, 42, b"case8", flags=7)
        decoded = decode_frame(encoded)
        self.assertEqual(decoded.kind, Message.PING)
        self.assertEqual(decoded.request_id, 42)
        self.assertEqual(decoded.flags, 7)
        self.assertEqual(decoded.payload, b"case8")

    def test_bad_crc(self) -> None:
        encoded = bytearray(encode_frame(Message.PING, 1, b"abc"))
        encoded[-1] ^= 0x01
        with self.assertRaisesRegex(ValueError, "CRC"):
            decode_frame(bytes(encoded))

    def test_bad_magic(self) -> None:
        encoded = bytearray(encode_frame(Message.PING, 1))
        encoded[:4] = (MAGIC ^ 1).to_bytes(4, "little")
        with self.assertRaisesRegex(ValueError, "magic"):
            decode_frame(bytes(encoded))

    def test_short_payload(self) -> None:
        encoded = encode_frame(Message.PING, 1, b"abc")[:-1]
        with self.assertRaisesRegex(ValueError, "length"):
            decode_frame(encoded)


if __name__ == "__main__":
    unittest.main()
