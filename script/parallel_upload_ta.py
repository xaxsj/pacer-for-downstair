#!/usr/bin/env python3
"""Parallel ranged SFTP upload to resume the TA archive quickly."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import paramiko

HOST = "172.23.234.141"
PORT = 60022
USER = "lvjianxiong"
KEY = r"C:\Users\34459\.ssh\id_ed25519"
LOCAL = r"C:\Users\34459\Desktop\pacer-main-hoi-20261108.tar.zst"
BASE = "/home/lvjianxiong/data/pacer-main-hoi-20261108.tar.zst"
N = 4
CHUNK = 8 * 1024 * 1024


def conn():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, key_filename=KEY,
                timeout=30, banner_timeout=30, auth_timeout=30)
    return cli, cli.open_sftp()


def upload_part(i, start, end, total):
    part = f"{BASE}.part{i}"
    for attempt in range(1, 7):
        cli = sftp = None
        try:
            cli, sftp = conn()
            try:
                sftp.remove(part)
            except Exception:
                pass
            with open(LOCAL, "rb") as src, sftp.open(part, "wb") as dst:
                src.seek(start)
                done = start
                while done < end:
                    data = src.read(min(CHUNK, end - done))
                    if not data:
                        break
                    dst.write(data)
                    done += len(data)
                    if (done - start) % (256 * 1024 * 1024) < CHUNK:
                        print(f"part{i}: {done}/{end} ({100.0 * done / total:.1f}%)", flush=True)
            size = sftp.stat(part).st_size
            if size == end - start:
                print(f"part{i} OK {size}", flush=True)
                sftp.close(); cli.close()
                return i, part, size
            print(f"part{i} bad size {size}", flush=True)
        except Exception as exc:
            print(f"part{i} attempt {attempt} error: {type(exc).__name__}: {exc}", flush=True)
        finally:
            try:
                if sftp is not None: sftp.close()
                if cli is not None: cli.close()
            except Exception:
                pass
        time.sleep(3)
    raise RuntimeError(f"part {i} failed")


def main() -> int:
    total = os.path.getsize(LOCAL)
    cli, sftp = conn()
    remote_size = sftp.stat(BASE).st_size
    sftp.close(); cli.close()
    print(f"local={total} remote={remote_size}", flush=True)
    if remote_size >= total:
        print("ALREADY_COMPLETE", flush=True)
        return 0
    remaining = total - remote_size
    per = (remaining + N - 1) // N
    ranges = []
    for i in range(N):
        a = remote_size + i * per
        b = min(total, a + per)
        if a < b:
            ranges.append((i, a, b))
    print("ranges", ranges, flush=True)
    with ThreadPoolExecutor(max_workers=len(ranges)) as ex:
        futs = [ex.submit(upload_part, i, a, b, total) for i, a, b in ranges]
        parts = [f.result() for f in futs]
    print("parts done", [(p[0], p[2]) for p in parts], flush=True)
    cli, sftp = conn()
    part_names = [f"{BASE}.part{i}" for i, _, _ in ranges]
    cat = "cat " + " ".join(part_names) + f" >> {BASE}"
    _, out, err = cli.exec_command(cat, timeout=600)
    rc = out.channel.recv_exit_status()
    print("cat rc", rc, err.read().decode("utf-8", "replace")[:200], flush=True)
    final = sftp.stat(BASE).st_size
    print("final", final, "expected", total, flush=True)
    for name in part_names:
        try: sftp.remove(name)
        except Exception: pass
    sftp.close(); cli.close()
    print("UPLOAD_OK" if final == total else "UPLOAD_SIZE_MISMATCH", flush=True)
    return 0 if final == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
