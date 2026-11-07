#!/usr/bin/env python3
"""Resumable SFTP upload for the KIT archive."""
import os
import sys
import time

import paramiko

HOST = "211.81.55.164"
PORT = 41190
USER = "ljx"
KEY = r"C:\Users\34459\.ssh\id_lab_ljx"
LOCAL = r"C:\Users\34459\Downloads\KIT.tar.bz2"
REMOTE = "/data/ljx/datasets/KIT.tar.bz2"
CHUNK = 4 * 1024 * 1024


def main() -> int:
    local_size = os.path.getsize(LOCAL)
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, key_filename=KEY,
                timeout=30, banner_timeout=30, auth_timeout=30)
    sftp = cli.open_sftp()
    try:
        try:
            remote_size = sftp.stat(REMOTE).st_size
        except FileNotFoundError:
            remote_size = 0
        print(f"local={local_size} remote={remote_size}", flush=True)
        if remote_size > local_size:
            print("remote is larger than local; abort", flush=True)
            return 2
        mode = "r+" if remote_size > 0 else "wb"
        start = time.time()
        last = remote_size
        with open(LOCAL, "rb") as src:
            src.seek(remote_size)
            with sftp.open(REMOTE, mode) as dst:
                if remote_size > 0:
                    dst.seek(remote_size)
                while True:
                    chunk = src.read(CHUNK)
                    if not chunk:
                        break
                    dst.write(chunk)
                    written = src.tell()
                    if written - last >= 128 * 1024 * 1024:
                        elapsed = max(time.time() - start, 1e-6)
                        speed = (written - remote_size) / elapsed / 1024 / 1024
                        print(f"{written}/{local_size} ({100.0*written/local_size:.1f}%) {speed:.1f} MB/s", flush=True)
                        last = written
                dst.flush()
        final = sftp.stat(REMOTE).st_size
        print(f"done remote={final}", flush=True)
        return 0 if final == local_size else 3
    finally:
        sftp.close()
        cli.close()


if __name__ == "__main__":
    sys.exit(main())
