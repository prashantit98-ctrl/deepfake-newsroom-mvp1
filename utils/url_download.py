import os
import uuid
import yt_dlp


ALLOWED_URL_DOMAINS = ("twitter.com", "x.com")


def is_supported_url(url):
    """
    Quick check that this looks like an X/Twitter URL before we try
    to download anything. Scoped deliberately narrow for now — other
    sites can be added later, but each one is its own can of worms
    (different content policies, reliability, legal considerations),
    so starting with just X keeps this manageable.
    """
    url_lower = url.lower()
    return any(domain in url_lower for domain in ALLOWED_URL_DOMAINS)


def download_video_from_url(url, download_dir="uploads"):
    """
    Downloads a video from a supported URL (currently X/Twitter only)
    using yt-dlp, and returns the local file path plus the original
    filename-style label for display purposes.

    yt-dlp is a third-party, actively-maintained tool that reverse
    engineers how sites like X serve video — there's no official API
    for this. It can occasionally break if X changes something on
    their end; if downloads suddenly start failing, the fix is often
    just upgrading yt-dlp (`pip install -U yt-dlp`).

    Raises RuntimeError with a human-readable message on failure,
    rather than leaking a raw yt-dlp stack trace to the caller.
    """

    if not is_supported_url(url):
        raise RuntimeError(
            "Only X (Twitter) video links are supported right now."
        )

    os.makedirs(download_dir, exist_ok=True)

    upload_id = uuid.uuid4().hex
    output_template = os.path.join(download_dir, f"{upload_id}.%(ext)s")

    ydl_opts = {
        "outtmpl": output_template,
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Keep downloads reasonably sized — this is a screening tool,
        # not an archiving tool, and we don't need max resolution to
        # run frame-sampling checks on it.
        "format_sort": ["res:720"],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_path = ydl.prepare_filename(info)
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(
            f"Could not download this video. It may be private, deleted, or "
            f"the link may not point to a video. ({e})"
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error downloading video: {e}")

    if not os.path.exists(downloaded_path):
        raise RuntimeError("Download reported success but the file is missing.")

    display_name = info.get("title") or f"x_video_{upload_id}.mp4"

    return downloaded_path, display_name, upload_id
