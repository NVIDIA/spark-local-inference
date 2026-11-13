# S3 model storage

The private model bucket is `s3://flare-models-555824081510-us-west-2/` in
`us-west-2`. Public access is blocked, object ownership is bucket-owner enforced,
and default encryption uses S3-managed AES-256 keys. Executable core storage uses
one zstd level-3 frame per resource; S3-managed encryption is separate
storage-at-rest protection. Incomplete multipart uploads
expire after seven days; completed objects do not expire.

Run the threaded uploader from the repository root. `uv` supplies its Boto3 and
Rich dependencies independently of the preparation environment:

```sh
uv run scripts/s3_upload.py /path/to/model.safetensors
uv run scripts/s3_upload.py /path/to/model.safetensors s3://flare-models-555824081510-us-west-2/cores/model.safetensors
uv run scripts/s3_upload.py /path/to/model.safetensors --threads 32 --part-size-mib 128
```

Without a destination, the object key is the source file name at the bucket
root. A destination ending in `/` is a prefix; otherwise it is the exact object key.
The source must be one file. To upload a core as one file, first create its
packed form with `ignition convert CACHE_ENTRY --form packed --output FILE`;
the uploader does not accept a directory-form cache entry.
An existing object with a matching key is replaced.
Keep the source file unchanged until upload completion.

Use `--dry-run` to inspect the key and size without AWS calls. `--profile` and
`--region` explicitly select AWS settings; otherwise Boto3 uses its standard
credential and region resolution, including the configured default AWS profile.

The uploader uses 16 transfer threads by default. Files of at least 64 MiB use
multipart upload with 64 MiB parts; the SDK increases
part size as needed to respect S3's part-count limit. Smaller files use a single
PUT. `--threads` and `--part-size-mib` control these settings.
File-backed parts are streamed rather than reading each entire part into memory.
Progress shows bytes, throughput, elapsed time, and ETA; the final summary reports
average end-to-end throughput.

Transient requests retry with exponential backoff, up to six attempts. Transfer
failures or Ctrl-C cancel pending work and attempt to abort the multipart upload.
There is no persistent resume across invocations; restart the command to upload
again. Forced termination or loss of connectivity
can leave incomplete uploads for the bucket lifecycle rule to remove.
