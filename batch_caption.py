#!/usr/bin/env python3
"""Batch-caption images through a local OpenAI-compatible vision server."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import mimetypes
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import quote

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "input"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "prompts" / "caption_prompt.txt"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}

DIRECT_DATA_URL_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

LOG_FIELDS = [
    "timestamp_utc",
    "source",
    "output",
    "status",
    "attempts",
    "caption_chars",
    "elapsed_seconds",
    "finish_reason",
    "response_field",
    "error",
    "raw_response",
    "model",
    "run_label",
    "endpoint",
    "prompt_file",
]


class CaptionError(RuntimeError):
    """An expected captioning failure with a useful diagnostic."""


@dataclass(frozen=True)
class CaptionResult:
    caption: str
    raw_response: str
    finish_reason: str
    response_field: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if endpoint.endswith("/v1"):
        return endpoint + "/chat/completions"
    return endpoint


def load_prompt(prompt_file: Path) -> str:
    try:
        prompt = prompt_file.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise CaptionError(f"Could not read prompt file '{prompt_file}': {exc}") from exc
    if not prompt:
        raise CaptionError(f"Prompt file is empty: {prompt_file}")
    return prompt


def discover_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator: Iterable[Path] = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def output_path_for(image_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative = image_path.relative_to(input_dir)
    return (output_dir / relative).with_suffix(".txt")


def find_output_collisions(
    image_paths: Iterable[Path], input_dir: Path, output_dir: Path
) -> dict[Path, list[Path]]:
    destinations: dict[Path, list[Path]] = {}
    for image_path in image_paths:
        destination = output_path_for(image_path, input_dir, output_dir)
        destinations.setdefault(destination, []).append(image_path)
    return {path: sources for path, sources in destinations.items() if len(sources) > 1}


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [flatten_text(part) for part in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value:
                text = flatten_text(value[key])
                if text:
                    return text
    return ""


def remove_reasoning_blocks(text: str) -> str:
    """Remove common visible reasoning wrappers while preserving the final answer."""

    cleaned = text.strip()
    for tag in ("think", "analysis", "reasoning"):
        cleaned = re.sub(
            rf"<{tag}\b[^>]*>.*?</{tag}>",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

    # Some servers return reasoning followed by only a closing tag and the final text.
    closing = re.compile(r"</(?:think|analysis|reasoning)>", re.IGNORECASE)
    matches = list(closing.finditer(cleaned))
    if matches:
        cleaned = cleaned[matches[-1].end() :].strip()

    cleaned = re.sub(
        r"^<(?:think|analysis|reasoning)\b[^>]*>\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```(?:text)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    return cleaned


def extract_caption(response_data: dict[str, Any]) -> CaptionResult:
    raw_response = json.dumps(response_data, ensure_ascii=False, separators=(",", ":"))
    try:
        choice = response_data["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise CaptionError("Server response did not contain choices[0]") from exc

    message = choice.get("message") or {}
    candidates = (
        ("content", message.get("content")),
        ("reasoning_content", message.get("reasoning_content")),
        ("reasoning", message.get("reasoning")),
        ("choice.text", choice.get("text")),
    )

    for field, value in candidates:
        text = flatten_text(value)
        if not text:
            continue
        caption = remove_reasoning_blocks(text)
        if caption:
            return CaptionResult(
                caption=caption,
                raw_response=raw_response,
                finish_reason=str(choice.get("finish_reason") or ""),
                response_field=field,
            )

    raise CaptionError("Model returned no usable caption text")


def _resize_image(image: Image.Image, max_megapixels: float) -> Image.Image:
    if max_megapixels <= 0:
        return image
    max_pixels = int(max_megapixels * 1_000_000)
    current_pixels = image.width * image.height
    if current_pixels <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / current_pixels)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def image_as_data_url(
    image_path: Path, max_megapixels: float = 0, jpeg_quality: int = 95
) -> str:
    suffix = image_path.suffix.lower()
    if suffix in DIRECT_DATA_URL_EXTENSIONS and max_megapixels <= 0:
        mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    try:
        with Image.open(image_path) as opened:
            opened.seek(0)
            image = ImageOps.exif_transpose(opened).copy()
    except Exception as exc:  # Pillow raises several format-specific exception types.
        raise CaptionError(f"Could not decode image '{image_path}': {exc}") from exc

    image = _resize_image(image, max_megapixels)
    buffer = BytesIO()
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        image.save(buffer, format="PNG", optimize=True)
        mime_type = "image/png"
    else:
        image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        mime_type = "image/jpeg"

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_as_file_url(image_path: Path, input_dir: Path) -> str:
    relative = image_path.relative_to(input_dir).as_posix()
    return "file://" + quote(relative, safe="/")


def make_payload(
    *,
    model: str,
    prompt: str,
    image_url: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }


def request_caption(
    *,
    endpoint: str,
    payload: dict[str, Any],
    timeout: float,
    api_key: str,
) -> CaptionResult:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib_request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        error_body = error_body.strip().replace("\n", " ")[:1200]
        raise CaptionError(f"Server returned HTTP {exc.code}: {error_body}") from exc
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        raise CaptionError(f"Request failed: {exc}") from exc

    try:
        response_data = json.loads(response_body)
    except json.JSONDecodeError as exc:
        diagnostic = response_body.strip().replace("\n", " ")[:1200]
        raise CaptionError(f"Server did not return JSON: {diagnostic}") from exc

    return extract_caption(response_data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Caption images one at a time through a local OpenAI-compatible "
            "multimodal server."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument(
        "--endpoint",
        "--base-url",
        default=os.getenv("CAPTION_ENDPOINT", DEFAULT_ENDPOINT),
        help="Full chat-completions endpoint, or a base URL ending in /v1.",
    )
    parser.add_argument(
        "--model", default=os.getenv("CAPTION_MODEL", "local-model")
    )
    parser.add_argument(
        "--api-key", default=os.getenv("CAPTION_API_KEY", "not-needed")
    )
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries after the first attempt (default: 2).",
    )
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan subfolders and mirror them under output/ (default: enabled).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-images", "--limit", type=int, default=0)
    parser.add_argument("--run-label", default="")
    parser.add_argument(
        "--image-transport",
        choices=("data-url", "file-url"),
        default="data-url",
        help=(
            "data-url is portable and needs no media whitelist; file-url requires "
            "llama-server --media-path to point at input/."
        ),
    )
    parser.add_argument(
        "--max-image-megapixels",
        type=float,
        default=0,
        help="Downscale larger images before upload; 0 preserves original dimensions.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--log-file", type=Path, default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_tokens < 1:
        raise CaptionError("--max-tokens must be at least 1")
    if args.retries < 0:
        raise CaptionError("--retries cannot be negative")
    if args.retry_delay < 0:
        raise CaptionError("--retry-delay cannot be negative")
    if args.max_images < 0:
        raise CaptionError("--max-images cannot be negative")
    if args.max_image_megapixels < 0:
        raise CaptionError("--max-image-megapixels cannot be negative")
    if not 1 <= args.jpeg_quality <= 100:
        raise CaptionError("--jpeg-quality must be between 1 and 100")


def make_log_path(log_file: Path | None) -> Path:
    if log_file is not None:
        return log_file
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_LOG_DIR / f"caption_run_{stamp}.csv"


def write_log_row(
    writer: csv.DictWriter[str], handle: Any, **values: Any
) -> None:
    writer.writerow({field: values.get(field, "") for field in LOG_FIELDS})
    handle.flush()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_args(args)
        input_dir = args.input_dir.resolve()
        output_dir = args.output_dir.resolve()
        prompt_file = args.prompt_file.resolve()
        endpoint = normalize_endpoint(args.endpoint)

        if not input_dir.is_dir():
            raise CaptionError(f"Input folder not found: {input_dir}")

        prompt = load_prompt(prompt_file)
        image_paths = discover_images(input_dir, args.recursive)
        if args.max_images:
            image_paths = image_paths[: args.max_images]

        if not image_paths:
            print(f"No supported images found in: {input_dir}")
            return 0

        collisions = find_output_collisions(image_paths, input_dir, output_dir)
        if collisions:
            details = []
            for destination, sources in collisions.items():
                source_names = ", ".join(str(path) for path in sources)
                details.append(f"{destination} <- {source_names}")
            raise CaptionError(
                "Multiple images would write the same caption file. Rename one of "
                "the colliding images:\n" + "\n".join(details)
            )

        if args.dry_run:
            print(f"Dry run: {len(image_paths)} image(s)")
            for image_path in image_paths:
                destination = output_path_for(image_path, input_dir, output_dir)
                print(f"{image_path} -> {destination}")
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = make_log_path(args.log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        counts = {"captioned": 0, "skipped": 0, "failed": 0}
        started = time.monotonic()

        with log_path.open("w", newline="", encoding="utf-8-sig") as log_handle:
            writer = csv.DictWriter(log_handle, fieldnames=LOG_FIELDS)
            writer.writeheader()

            total = len(image_paths)
            for index, image_path in enumerate(image_paths, start=1):
                destination = output_path_for(image_path, input_dir, output_dir)

                if (
                    not args.overwrite
                    and destination.exists()
                    and destination.stat().st_size > 0
                ):
                    counts["skipped"] += 1
                    if not args.quiet:
                        print(f"[{index}/{total}] Skipped: {image_path.name}")
                    write_log_row(
                        writer,
                        log_handle,
                        timestamp_utc=utc_now(),
                        source=str(image_path),
                        output=str(destination),
                        status="skipped",
                        attempts=0,
                        model=args.model,
                        run_label=args.run_label,
                        endpoint=endpoint,
                        prompt_file=str(prompt_file),
                    )
                    continue

                image_started = time.monotonic()
                attempts = 0
                last_error = ""
                raw_response = ""
                success_result: CaptionResult | None = None

                while attempts <= args.retries:
                    attempts += 1
                    try:
                        if args.image_transport == "data-url":
                            image_url = image_as_data_url(
                                image_path,
                                max_megapixels=args.max_image_megapixels,
                                jpeg_quality=args.jpeg_quality,
                            )
                        else:
                            image_url = image_as_file_url(image_path, input_dir)

                        payload = make_payload(
                            model=args.model,
                            prompt=prompt,
                            image_url=image_url,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                        )
                        success_result = request_caption(
                            endpoint=endpoint,
                            payload=payload,
                            timeout=args.timeout,
                            api_key=args.api_key,
                        )
                        raw_response = success_result.raw_response
                        break
                    except Exception as exc:
                        last_error = str(exc)
                        if attempts <= args.retries:
                            delay = args.retry_delay * (2 ** (attempts - 1))
                            if not args.quiet:
                                print(
                                    f"[{index}/{total}] Retry {attempts}/{args.retries} "
                                    f"for {image_path.name}: {last_error}"
                                )
                            if delay:
                                time.sleep(delay)

                elapsed = time.monotonic() - image_started
                if success_result is None:
                    counts["failed"] += 1
                    print(f"[{index}/{total}] Failed: {image_path.name}: {last_error}")
                    write_log_row(
                        writer,
                        log_handle,
                        timestamp_utc=utc_now(),
                        source=str(image_path),
                        output=str(destination),
                        status="error",
                        attempts=attempts,
                        elapsed_seconds=f"{elapsed:.3f}",
                        error=last_error,
                        raw_response=raw_response,
                        model=args.model,
                        run_label=args.run_label,
                        endpoint=endpoint,
                        prompt_file=str(prompt_file),
                    )
                    if args.fail_fast:
                        break
                    continue

                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(success_result.caption + "\n", encoding="utf-8")
                counts["captioned"] += 1
                if not args.quiet:
                    print(
                        f"[{index}/{total}] Captioned: {image_path.name} "
                        f"({elapsed:.1f}s)"
                    )
                write_log_row(
                    writer,
                    log_handle,
                    timestamp_utc=utc_now(),
                    source=str(image_path),
                    output=str(destination),
                    status="captioned",
                    attempts=attempts,
                    caption_chars=len(success_result.caption),
                    elapsed_seconds=f"{elapsed:.3f}",
                    finish_reason=success_result.finish_reason,
                    response_field=success_result.response_field,
                    raw_response=success_result.raw_response,
                    model=args.model,
                    run_label=args.run_label,
                    endpoint=endpoint,
                    prompt_file=str(prompt_file),
                )

        total_elapsed = time.monotonic() - started
        print(
            "Finished: "
            f"{counts['captioned']} captioned, "
            f"{counts['skipped']} skipped, "
            f"{counts['failed']} failed "
            f"in {total_elapsed:.1f}s."
        )
        print(f"Log: {log_path}")
        return 1 if counts["failed"] else 0

    except CaptionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
