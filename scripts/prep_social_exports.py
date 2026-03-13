"""Build a cleaned public-post corpus from LinkedIn and X export zips."""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import zipfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

try:
    from .experiment_config import (
        DATA_DIR,
        PUBLIC_POSTS_PATH,
    )
except ImportError:  # pragma: no cover - enables direct script execution
    from experiment_config import (
        DATA_DIR,
        PUBLIC_POSTS_PATH,
    )


HANDLE_RE = re.compile(r"(?<!\w)@([A-Za-z0-9_]{1,15})\b")
URL_RE = re.compile(r"https?://\S+")
TRAILING_TCO_RE = re.compile(r"(?:\s+https://t\.co/\w+)+\s*$", re.IGNORECASE)
PHONE_ONLY_RE = re.compile(r"^\+?[\d\s().-]{7,}$")
WHITESPACE_RE = re.compile(r"[ \t]+")
NEWLINE_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class PostRecord:
    id: str
    platform: str
    source_type: str
    created_at: str
    text: str
    char_len: int
    length_bucket: str


class HTMLStripper(HTMLParser):
    """Convert simple HTML fragments to plain text while preserving paragraphs."""

    BLOCK_TAGS = {"br", "p", "div", "li"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def strip(self, value: str) -> str:
        self.parts.clear()
        self.feed(value)
        self.close()
        return "".join(self.parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--linkedin-zip",
        type=Path,
        required=True,
        help="Path to the LinkedIn export zip.",
    )
    parser.add_argument(
        "--x-zip",
        type=Path,
        required=True,
        help="Path to the X export zip.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PUBLIC_POSTS_PATH,
        help="JSONL output for cleaned public posts.",
    )
    return parser.parse_args()


def read_csv_from_zip(zf: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    with zf.open(member) as raw:
        wrapper = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        return list(csv.DictReader(wrapper))


def read_x_ytd_json(zf: zipfile.ZipFile, member: str) -> list[dict[str, object]]:
    raw = zf.read(member).decode("utf-8")
    _, rhs = raw.split("=", 1)
    return json.loads(rhs.strip().rstrip(";"))


def strip_html(value: str) -> str:
    if "<" not in value or ">" not in value:
        return html.unescape(value)
    return html.unescape(HTMLStripper().strip(value))


def collapse_line_whitespace(value: str) -> str:
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in value.splitlines()]
    filtered = [line for line in lines if line]
    return "\n\n".join(filtered)


def normalize_export_quotes(value: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue
        line = line.replace('""', '"')
        if set(line) <= {'"'}:
            cleaned_lines.append("")
            continue
        if line.startswith('"'):
            line = line[1:]
        if line.endswith('"'):
            line = line[:-1]
        line = line.strip()
        if not line or set(line) <= {'"'}:
            cleaned_lines.append("")
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def is_low_signal_text(value: str) -> bool:
    if not value:
        return True
    normalized = value.strip()
    if normalized.lower() in {"n/a", "na", "-", ".", "...", "<url>", "<handle>"}:
        return True
    if PHONE_ONLY_RE.fullmatch(normalized):
        return True
    without_whitespace = re.sub(r"\s+", "", normalized)
    if URL_RE.fullmatch(normalized) or without_whitespace == "<URL>":
        return True
    return False


def normalized_text_key(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def length_bucket(char_len: int) -> str:
    if char_len <= 140:
        return "short"
    if char_len <= 320:
        return "medium"
    return "long"


def clean_text(raw_text: str) -> str:
    text = strip_html(raw_text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = normalize_export_quotes(text)
    text = TRAILING_TCO_RE.sub("", text)
    text = HANDLE_RE.sub("<HANDLE>", text)
    text = URL_RE.sub("<URL>", text)
    text = collapse_line_whitespace(text)
    text = NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def parse_linkedin_datetime(value: str) -> str:
    dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def parse_twitter_datetime(value: str) -> str:
    dt = datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso_datetime(value: str) -> str:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    ).isoformat().replace("+00:00", "Z")


def iter_linkedin_public_posts(linkedin_zip: Path) -> Iterable[PostRecord]:
    with zipfile.ZipFile(linkedin_zip) as zf:
        for row in read_csv_from_zip(zf, "Shares.csv"):
            cleaned = clean_text(row["ShareCommentary"])
            if len(cleaned) < 25 or is_low_signal_text(cleaned):
                continue
            created_at = parse_linkedin_datetime(row["Date"])
            share_link = row["ShareLink"].rstrip("/").split("/")[-1]
            yield PostRecord(
                id=f"linkedin_share_{share_link or row['Date'].replace(' ', '_')}",
                platform="linkedin",
                source_type="linkedin_share",
                created_at=created_at,
                text=cleaned,
                char_len=len(cleaned),
                length_bucket=length_bucket(len(cleaned)),
            )


def iter_x_public_posts(x_zip: Path) -> Iterable[PostRecord]:
    with zipfile.ZipFile(x_zip) as zf:
        tweets = read_x_ytd_json(zf, "data/tweets.js")
        for item in tweets:
            tweet = item["tweet"]
            full_text = tweet.get("full_text", "")
            if (
                tweet.get("in_reply_to_status_id")
                or tweet.get("retweeted")
                or full_text.startswith("RT @")
            ):
                continue
            cleaned = clean_text(full_text)
            if len(cleaned) < 25 or is_low_signal_text(cleaned):
                continue
            yield PostRecord(
                id=f"x_tweet_{tweet['id_str']}",
                platform="x",
                source_type="tweet_original",
                created_at=parse_twitter_datetime(tweet["created_at"]),
                text=cleaned,
                char_len=len(cleaned),
                length_bucket=length_bucket(len(cleaned)),
            )

        note_tweets = read_x_ytd_json(zf, "data/note-tweet.js")
        for item in note_tweets:
            note = item["noteTweet"]
            cleaned = clean_text(note["core"].get("text", ""))
            if len(cleaned) < 25 or is_low_signal_text(cleaned):
                continue
            yield PostRecord(
                id=f"x_note_{note['noteTweetId']}",
                platform="x",
                source_type="note_tweet",
                created_at=parse_iso_datetime(note["createdAt"]),
                text=cleaned,
                char_len=len(cleaned),
                length_bucket=length_bucket(len(cleaned)),
            )

        community = read_x_ytd_json(zf, "data/community-tweet.js")
        for item in community:
            tweet = item["tweet"]
            if tweet.get("in_reply_to_status_id"):
                continue
            cleaned = clean_text(tweet.get("full_text", ""))
            if len(cleaned) < 25 or is_low_signal_text(cleaned):
                continue
            yield PostRecord(
                id=f"x_community_{tweet['id_str']}",
                platform="x",
                source_type="community_post",
                created_at=parse_twitter_datetime(tweet["created_at"]),
                text=cleaned,
                char_len=len(cleaned),
                length_bucket=length_bucket(len(cleaned)),
            )


def build_public_posts(linkedin_zip: Path, x_zip: Path) -> list[PostRecord]:
    seen: set[str] = set()
    records: list[PostRecord] = []
    for record in list(iter_linkedin_public_posts(linkedin_zip)) + list(
        iter_x_public_posts(x_zip)
    ):
        key = normalized_text_key(record.text)
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
    return sorted(records, key=lambda row: (row.created_at, row.id))


def write_jsonl(path: Path, rows: Iterable[PostRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = build_public_posts(args.linkedin_zip, args.x_zip)
    write_jsonl(args.output, records)

    by_source: dict[str, int] = {}
    for record in records:
        by_source[record.source_type] = by_source.get(record.source_type, 0) + 1

    print(f"Wrote {len(records)} cleaned public posts to {args.output}")
    for source_type, count in sorted(by_source.items()):
        print(f"  {source_type}: {count}")


if __name__ == "__main__":
    main()
