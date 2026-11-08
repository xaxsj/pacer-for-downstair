#!/usr/bin/env python3
"""Resumable SFTP upload with reconnects for the TA 3090 server."""
from __future__ import annotations

import os
import sys
import time

import paramiko

HOST = "172.23.234.141"
PORT = 60022
USER = "lvjianxiong"
KEY = r"C:\Users\34459\.ssh\id_ed25519"
LOCAL = r"C:\Users\34459\Desktop\pacer-main-hoi-20261108.tar.zst"
REMOTE = "/home/lvjianxiong/data/pacer-main-hoi-20261108.tar.zst"
CHUNK = 4 * 1024 * 1024


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, key_filename=KEY,
                timeout=30, banner_timeout=30, auth_timeout=30)
    return cli, cli.open_sftp()


def main() -> int:
    total = os.path.getsize(LOCAL)
    sent = 0
    for attempt in range(1, 9):
        cli = sftp = None
        try:
            cli, sftp = connect()
            try:
                offset = sftp.stat(REMOTE).st_size
            except FileNotFoundError:
                offset = 0
            if offset > total:
                print("remote larger than local; truncating", flush=True)
                sftp.remove(REMOTE)
                offset = 0
            print(f"attempt {attempt} resume at {offset}/{total}", flush=True)
            with open(LOCAL, "rb") as fh, sftp.open(REMOTE, "r+b") as out:
                fh.seek(offset)
                out.seek(offset)
                sent = offset
                while sent < total:
                    data = fh.read(CHUNK)
                    if not data:
                        break
                    out.write(data)
                    sent += len(data)
                    if sent % (64 * CHUNK) < CHUNK:
                        print(f"  {sent}/{total} ({100.0 * sent / total:.1f}%)", flush=True)
                out.flush()
            remote_size = sftp.stat(REMOTE).st_size
            print(f"attempt {attempt} finished remote={remote_size}", flush=True)
            if remote_size == total:
                print("UPLOAD_OK", flush=True)
                sftp.close(); cli.close()
                return 0
        except Exception as exc:
            print(f"attempt {attempt} error: {type(exc).__name__}: {exc}", flush=True)
        finally:
            try:
                if sftp is not None:
                    sftp.close()
                if cli is not None:
                    cli.close()
            except Exception:
                pass
        time.sleep(5)
    print("UPLOAD_FAILED", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
