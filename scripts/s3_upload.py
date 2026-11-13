#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3>=1.40,<2", "rich>=13,<15"]
# ///
"""Upload one file to S3 with threaded multipart transfer and live statistics."""

from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path
import time
from urllib.parse import urlsplit

import boto3
from boto3.s3.transfer import TransferConfig, create_transfer_manager
from botocore.config import Config
from rich.console import Console
from rich.filesize import decimal
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Column
from s3transfer.subscribers import BaseSubscriber


DEFAULT_DESTINATION = "s3://flare-models-555824081510-us-west-2/"
MIB = 1024**2


def destination_key(source: Path, destination: str) -> tuple[str, str]:
    uri = urlsplit(destination)
    if uri.scheme != "s3" or not uri.netloc or uri.query or uri.fragment:
        raise ValueError("destination must be s3://BUCKET[/KEY-OR-PREFIX] without query or fragment")
    prefix = uri.path.lstrip("/")
    if not source.is_file():
        raise ValueError(f"source is not a regular file: {source}")
    key = prefix + source.name if not prefix or prefix.endswith("/") else prefix
    return uri.netloc, key


class UploadProgress(BaseSubscriber):
    def __init__(self, progress: Progress, task: int):
        self.progress = progress
        self.task = task

    def on_progress(self, future, bytes_transferred, **kwargs):
        # Retry callbacks subtract bytes already reported for the failed attempt.
        self.progress.advance(self.task, bytes_transferred)


def transfer_config(threads: int, part_size_mib: int) -> TransferConfig:
    if threads < 1:
        raise ValueError("threads must be positive")
    if not 5 <= part_size_mib <= 5120:
        raise ValueError("part size must be between 5 and 5120 MiB")
    return TransferConfig(
        multipart_threshold=part_size_mib * MIB,
        multipart_chunksize=part_size_mib * MIB,
        max_concurrency=threads,
        use_threads=True,
        preferred_transfer_client="classic",
    )


def upload_file(client, source: Path, bucket: str, key: str, *, threads: int = 16,
                part_size_mib: int = 64, console: Console | None = None) -> None:
    config = transfer_config(threads, part_size_mib)
    console = console or Console()
    total_bytes = source.stat().st_size
    console.print(f"Destination: s3://{bucket}/{key}", markup=False)
    console.print(f"{threads} transfer threads · "
                  f"{part_size_mib} MiB parts (SDK increases size for large objects)")
    progress = Progress(
        TextColumn("{task.description}", markup=False,
                   table_column=Column(no_wrap=True, max_width=min(40, max(12, console.width // 5)))),
        BarColumn(), TaskProgressColumn(table_column=Column(no_wrap=True)),
        DownloadColumn(table_column=Column(no_wrap=True)),
        TransferSpeedColumn(table_column=Column(no_wrap=True)),
        TimeElapsedColumn(table_column=Column(no_wrap=True)),
        TimeRemainingColumn(table_column=Column(no_wrap=True)),
        console=console, refresh_per_second=5,
    )
    started = time.monotonic()
    with progress:
        task = progress.add_task(source.name, total=total_bytes)
        with create_transfer_manager(client, config) as manager:
            manager.upload(str(source), bucket, key,
                           subscribers=[UploadProgress(progress, task)]).result()
    elapsed = time.monotonic() - started
    console.print(f"[bold green]Uploaded {decimal(total_bytes)} "
                  f"in {elapsed:.2f}s · {decimal(total_bytes / max(elapsed, 1e-9))}/s average[/bold green]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="file to upload")
    parser.add_argument("destination", nargs="?", default=DEFAULT_DESTINATION,
                        help=f"S3 key or prefix (default: {DEFAULT_DESTINATION})")
    parser.add_argument("--threads", type=int, default=16, help="multipart transfer worker count (default: 16)")
    parser.add_argument("--part-size-mib", type=int, default=64, help="multipart threshold and part size (default: 64 MiB)")
    parser.add_argument("--profile", help="AWS shared-config profile; otherwise use the SDK credential chain")
    parser.add_argument("--region", help="override the AWS profile's region")
    parser.add_argument("--dry-run", action="store_true", help="show destination key and size without AWS calls")
    args = parser.parse_args()
    console = Console()
    try:
        transfer_config(args.threads, args.part_size_mib)
        source = args.source.expanduser()
        bucket, key = destination_key(source, args.destination)
        if args.dry_run:
            console.print(f"{source} → s3://{bucket}/{key} ({decimal(source.stat().st_size)})", markup=False)
            return 0
        session = boto3.Session(profile_name=args.profile, region_name=args.region)
        with closing(session.client("s3", config=Config(
            max_pool_connections=args.threads,
            retries={"mode": "standard", "total_max_attempts": 6},
            connect_timeout=10, read_timeout=60,
        ))) as client:
            upload_file(client, source, bucket, key, threads=args.threads,
                        part_size_mib=args.part_size_mib, console=console)
    except KeyboardInterrupt:
        console.print("Interrupted; upload cancelled and multipart cleanup attempted.", style="yellow")
        return 130
    except Exception as error:
        console.print(f"Error: {error}", style="red", markup=False)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
