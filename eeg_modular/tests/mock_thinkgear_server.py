"""Minimal streaming ThinkGear TCP server for protocol rehearsals."""

from __future__ import annotations

import argparse
import json
import socket
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--rate", type=float, default=128.0)
    args = parser.parse_args()
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        connection, _ = server.accept()
        with connection:
            connection.recv(4096)
            started = time.monotonic()
            index = 0
            while time.monotonic() - started < args.max_seconds:
                packet = {"rawEeg": (index % 401) - 200}
                if index % max(1, int(args.rate)) == 0:
                    packet.update(
                        {
                            "poorSignalLevel": 0,
                            "eSense": {"attention": 50, "meditation": 50},
                        }
                    )
                try:
                    connection.sendall(
                        (json.dumps(packet, separators=(",", ":")) + "\r").encode()
                    )
                except (BrokenPipeError, ConnectionResetError):
                    break
                index += 1
                time.sleep(1.0 / args.rate)


if __name__ == "__main__":
    main()
