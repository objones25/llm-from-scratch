from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

_S3_SCHEME = "s3://"

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "llmtrain" / "s3"


def is_s3_uri(path: str) -> bool:
    return path.startswith(_S3_SCHEME)


def sibling_path(path_or_uri: str, filename: str) -> str:
    """`filename` in the same directory/prefix as path_or_uri, local or s3://."""
    if is_s3_uri(path_or_uri):
        parsed = urlsplit(path_or_uri)
        key = PurePosixPath(parsed.path.lstrip("/"))
        return f"s3://{parsed.netloc}/{key.parent / filename}"
    return str(Path(path_or_uri).parent / filename)


def resolve_local_path(path_or_uri: str, cache_dir: Path | None = None) -> Path:
    """A local filesystem path for path_or_uri, downloading from S3 first if needed.

    Non-s3:// inputs pass through unchanged. s3:// inputs are cached under
    cache_dir keyed by bucket/key, so repeated runs against the same checkpoint
    (e.g. iterating on --prompt/--temperature) don't re-download a multi-GB file
    every time — checkpoints are immutable once written, so a plain existence
    check is enough.
    """
    if not is_s3_uri(path_or_uri):
        return Path(path_or_uri)

    try:
        import boto3
    except ImportError as e:
        raise ImportError(
            "loading from an s3:// path requires the optional 's3' dependency "
            "group — install with `uv sync --extra s3`"
        ) from e

    parsed = urlsplit(path_or_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    local_path = (cache_dir or _DEFAULT_CACHE_DIR) / bucket / key
    if local_path.exists():
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    # endpoint_url/region come from AWS_ENDPOINT_URL_S3/AWS_DEFAULT_REGION env vars
    # (botocore >=1.31 resolves these automatically) — no custom client config
    # needed here, so this works against any S3-compatible provider via .env alone.
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(local_path))
    return local_path
