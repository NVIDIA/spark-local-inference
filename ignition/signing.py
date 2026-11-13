# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encrypted Ed25519 signing keys and public publisher profiles."""

import getpass
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import warnings

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ignition.preparation.protection import validate_namespace


def passphrase(prompt):
    if not sys.stdin.isatty():
        raise ValueError('encrypted signing keys require an interactive terminal for the passphrase')
    with warnings.catch_warnings():
        warnings.simplefilter('error', getpass.GetPassWarning)
        try:
            return getpass.getpass(prompt).encode('utf-8')
        except getpass.GetPassWarning as error:
            raise ValueError('cannot disable terminal echo for the signing passphrase') from error


def load_key(path):
    path = Path(path).expanduser()
    with path.open('rb') as source:
        metadata = os.fstat(source.fileno())
        if os.name == 'nt':
            from ._windows_permissions import owner_only

            private = owner_only(path)
        else:
            private = metadata.st_uid == os.getuid() and not metadata.st_mode & 0o077
        if not stat.S_ISREG(metadata.st_mode) or not private:
            raise ValueError(f'signing key must be an owner-only regular file: {path}')
        if metadata.st_size > 16384:
            raise ValueError('signing key exceeds 16 KiB')
        encoded = source.read(16385)
        if len(encoded) > 16384:
            raise ValueError('signing key exceeds 16 KiB')
    password = (passphrase(f'Passphrase for {path}: ')
                if encoded.startswith(b'-----BEGIN ENCRYPTED PRIVATE KEY-----') else None)
    key = serialization.load_pem_private_key(encoded, password=password)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError('publishing requires an Ed25519 private key')
    return key


def generate(directory, name, namespace, password):
    """Create an encrypted key and a separate public profile in a private directory."""
    validate_namespace(namespace)
    if namespace == 'local-development':
        raise ValueError('production keys cannot use the local-development namespace')
    if not name or len(name.encode()) > 256 or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in name):
        raise ValueError('invalid publisher display name')
    if len(password) < 16:
        raise ValueError('use a unique signing passphrase of at least 16 bytes')
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    key = Ed25519PrivateKey.generate()
    encoded = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.BestAvailableEncryption(password))
    public = key.public_key().public_bytes_raw()
    profile = json.dumps([{'name': name, 'namespace': namespace, 'public_key': list(public),
                           'development': False}], indent=2).encode() + b'\n'
    for filename, contents in (('signing-key.pem', encoded), ('publisher.json', profile)):
        path = directory / filename
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as output:
            if os.name == 'nt':
                from ._windows_permissions import restrict_file

                restrict_file(path)
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
    return hashlib.sha256(public).hexdigest()


def change_password(path):
    """Replace the key's encryption after terminal authentication and confirmation."""
    path = Path(path).expanduser().resolve(strict=True)
    before = path.stat()
    key = load_key(path)
    password = passphrase('New signing-key passphrase: ')
    if password != passphrase('Confirm signing-key passphrase: '):
        raise ValueError('passphrases do not match')
    if len(password) < 16:
        raise ValueError('use a unique signing passphrase of at least 16 bytes')
    encoded = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.BestAvailableEncryption(password))
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}-', dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            if os.name == 'nt':
                from ._windows_permissions import restrict_file

                restrict_file(temporary)
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        after = path.stat()
        if any(getattr(before, field) != getattr(after, field)
               for field in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
            raise ValueError('signing key changed while entering the passphrase; retry')
        os.replace(temporary, path)
        if os.name == 'posix':
            try:
                directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except OSError as error:
                raise OSError('key passphrase was changed, but syncing its directory failed') from error
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()
