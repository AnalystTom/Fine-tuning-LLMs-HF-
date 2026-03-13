from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

from scripts.prep_social_exports import build_public_posts, clean_text


def _write_csv_to_zip(zf: zipfile.ZipFile, name: str, rows: list[dict[str, str]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    zf.writestr(name, buffer.getvalue())


def _write_ytd_json(zf: zipfile.ZipFile, name: str, payload: list[dict[str, object]]) -> None:
    zf.writestr(name, "window.YTD.data = " + json.dumps(payload) + ";")


def test_clean_text_normalizes_links_and_handles() -> None:
    raw = "Hi @builder check this out https://example.com and trailing https://t.co/abc123"
    assert clean_text(raw) == "Hi <HANDLE> check this out <URL> and trailing"


def test_build_public_posts_only_uses_approved_sources(tmp_path: Path) -> None:
    linkedin_zip = tmp_path / "linkedin.zip"
    x_zip = tmp_path / "x.zip"

    with zipfile.ZipFile(linkedin_zip, "w") as zf:
        _write_csv_to_zip(
            zf,
            "Shares.csv",
            [
                {
                    "Date": "2026-03-03 22:42:44",
                    "ShareLink": "https://www.linkedin.com/feed/update/1",
                    "ShareCommentary": "Building an AI workflow that actually feels useful.",
                    "SharedUrl": "",
                    "MediaUrl": "",
                    "Visibility": "PUBLIC",
                }
            ],
        )
        _write_csv_to_zip(
            zf,
            "Comments.csv",
            [{"Date": "2026-03-03 22:44:03", "Link": "x", "Message": "Should not be used"}],
        )
        _write_csv_to_zip(
            zf,
            "messages.csv",
            [
                {
                    "CONVERSATION ID": "1",
                    "CONVERSATION TITLE": "DM",
                    "FROM": "Thomas Mann",
                    "SENDER PROFILE URL": "",
                    "TO": "Someone",
                    "RECIPIENT PROFILE URLS": "",
                    "DATE": "2026-03-03 22:44:03 UTC",
                    "SUBJECT": "",
                    "CONTENT": "Should not be used",
                    "FOLDER": "INBOX",
                    "ATTACHMENTS": "",
                }
            ],
        )

    with zipfile.ZipFile(x_zip, "w") as zf:
        _write_ytd_json(
            zf,
            "data/tweets.js",
            [
                {
                    "tweet": {
                        "id_str": "100",
                        "full_text": "Public original post about agent workflows and deployment.",
                        "created_at": "Wed Mar 05 09:16:21 +0000 2026",
                        "retweeted": False,
                        "in_reply_to_status_id": None,
                    }
                },
                {
                    "tweet": {
                        "id_str": "101",
                        "full_text": "Reply that should not be included.",
                        "created_at": "Wed Mar 05 09:17:21 +0000 2026",
                        "retweeted": False,
                        "in_reply_to_status_id": "10",
                    }
                },
            ],
        )
        _write_ytd_json(
            zf,
            "data/note-tweet.js",
            [
                {
                    "noteTweet": {
                        "noteTweetId": "200",
                        "createdAt": "2026-03-05T08:15:13.000Z",
                        "core": {"text": "A longer note about what changed once the system started using multiple models."},
                    }
                }
            ],
        )
        _write_ytd_json(
            zf,
            "data/community-tweet.js",
            [
                {
                    "tweet": {
                        "id_str": "300",
                        "full_text": "Community post asking for feedback on an AI product idea.",
                        "created_at": "Wed Mar 04 09:16:21 +0000 2026",
                        "in_reply_to_status_id": None,
                    }
                }
            ],
        )
        _write_ytd_json(
            zf,
            "data/direct-messages.js",
            [
                {
                    "dmConversation": {
                        "messages": [
                            {
                                "messageCreate": {
                                    "text": "Should not be used",
                                    "createdAt": "2026-03-05T08:15:13.000Z",
                                }
                            }
                        ]
                    }
                }
            ],
        )

    posts = build_public_posts(linkedin_zip, x_zip)
    assert [post.source_type for post in posts] == [
        "linkedin_share",
        "community_post",
        "note_tweet",
        "tweet_original",
    ]
