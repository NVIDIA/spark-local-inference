# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import closing
from io import StringIO
import pytest

boto3 = pytest.importorskip("boto3")
pytest.importorskip("rich")

from botocore.config import Config
from botocore.exceptions import ClientError
from botocore.stub import ANY, Stubber
from rich.console import Console
from rich.progress import Progress

from scripts.s3_upload import (
    MIB,
    UploadProgress,
    destination_key,
    transfer_config,
    upload_file,
)


def test_file_destination_keys(tmp_path):
    source = tmp_path / "model.flare"
    source.write_bytes(b"model")
    for destination, key in [
        ("s3://models", "model.flare"),
        ("s3://models/cores/", "cores/model.flare"),
        ("s3://models/exact.flare", "exact.flare"),
    ]:
        assert destination_key(source, destination) == ("models", key)


def test_empty_file(tmp_path):
    source = tmp_path / "empty"
    source.touch()
    assert destination_key(source, "s3://models/") == ("models", "empty")


@pytest.mark.parametrize("destination", ["https://models/", "s3:///key", "s3://models/key?query", "s3://models/key#fragment"])
def test_invalid_destinations(tmp_path, destination):
    with pytest.raises(ValueError, match="destination"):
        destination_key(tmp_path, destination)


def test_missing_file_and_directory_rejected(tmp_path):
    for source in [tmp_path / "missing", tmp_path]:
        with pytest.raises(ValueError, match="not a regular file"):
            destination_key(source, "s3://models/")


@pytest.mark.parametrize("threads,part_size", [(0, 64), (1, 4), (1, 5121)])
def test_invalid_transfer_settings(threads, part_size):
    with pytest.raises(ValueError):
        transfer_config(threads, part_size)


def test_progress_subtracts_retried_bytes():
    progress = Progress()
    file = progress.add_task("file", total=100)
    subscriber = UploadProgress(progress, file)
    for count in [75, -75, 100]:
        subscriber.on_progress(None, count)
    assert progress.tasks[0].completed == 100


def test_failed_part_aborts_multipart_upload(tmp_path):
    source = tmp_path / "weights.bin"
    with source.open("wb") as stream:
        stream.truncate(6 * MIB)
    client = boto3.client(
        "s3", region_name="us-west-2", aws_access_key_id="testing",
        aws_secret_access_key="testing", config=Config(retries={"max_attempts": 0}),
    )
    with closing(client), Stubber(client) as stub:
        stub.add_response("create_multipart_upload", {"UploadId": "upload-id"},
                          {"Bucket": "models", "Key": "weights.bin", "ChecksumAlgorithm": "CRC32"})
        stub.add_client_error("upload_part", service_error_code="AccessDenied",
                              service_message="part denied", http_status_code=403,
                              expected_params={"Bucket": "models", "Key": "weights.bin",
                                               "UploadId": "upload-id", "PartNumber": 1,
                                               "Body": ANY, "ChecksumAlgorithm": "CRC32"})
        stub.add_response("abort_multipart_upload", {},
                          {"Bucket": "models", "Key": "weights.bin", "UploadId": "upload-id"})
        with pytest.raises(ClientError, match="part denied"):
            upload_file(client, source, "models", "weights.bin", threads=1,
                        part_size_mib=5, console=Console(file=StringIO()))
        stub.assert_no_pending_responses()
