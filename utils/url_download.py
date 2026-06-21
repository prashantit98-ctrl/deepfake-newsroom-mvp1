import os
import time
import uuid
import yt_dlp


ALLOWED_URL_DOMAINS = ("twitter.com", "x.com", "video.twimg.com", "pbs.twimg.com")

# If set, points to a cookies.txt file (Netscape format, exported from
# a logged-in browser session) that yt-dlp will use to authenticate
# requests to X. This is meaningfully more reliable than X's anonymous
# "guest" session, which is documented as intermittently flaky.
#
# SECURITY: this file contains live session credentials for whatever
# X account it was exported from — treat it like a password. Never
# commit it to git. On Railway, upload it as a mounted file/volume and
# point this env var at its path, or use a secrets file feature if
# available — do not paste its contents into a regular env var value.
COOKIE_FILE_PATH = os.environ.get("X_COOKIES_FILE")

# Transient "guest token" failures are a long-documented, intermittent
# issue with X's anonymous API access (not specific to this project —
# see yt-dlp's own issue tracker). A short retry loop resolves many of
# these without any user-visible delay worth worrying about.
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 3

# Error substrings that indicate a transient, retry-worthy failure
# (as opposed to a real "this isn't a video" or "this is private"
# failure, which retrying won't fix).
TRANSIENT_ERROR_SIGNATURES = (
    "bad guest token",
    "rate limit",
    "timed out",
    "timeout",
    "temporarily",
)


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


def _is_transient_error(error_message):
    error_lower = str(error_message).lower()
    return any(sig in error_lower for sig in TRANSIENT_ERROR_SIGNATURES)


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

    If X_COOKIES_FILE is set, authenticated cookies are used instead
    of X's anonymous guest-token flow, which is documented as
    intermittently unreliable. Either way, transient failures are
    retried a few times before giving up.

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

    if COOKIE_FILE_PATH:
        if os.path.exists(COOKIE_FILE_PATH):
            ydl_opts["cookiefile"] = COOKIE_FILE_PATH
        else:
            # Misconfigured env var shouldn't silently degrade to
            # anonymous access without any signal — but it also
            # shouldn't hard-crash the whole feature. Fall through to
            # anonymous access; the caller can check logs if downloads
            # seem less reliable than expected.
            pass

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_path = ydl.prepare_filename(info)

            if not os.path.exists(downloaded_path):
                raise RuntimeError("Download reported success but the file is missing.")

            display_name = info.get("title") or f"x_video_{upload_id}.mp4"
            return downloaded_path, display_name, upload_id

        except yt_dlp.utils.DownloadError as e:
            last_error = e
            if _is_transient_error(e) and attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            break

        except Exception as e:
            last_error = e
            break

    if last_error and _is_transient_error(last_error):
        raise RuntimeError(
            "X's service is temporarily rejecting this request — this is a "
            "known intermittent issue with X's anonymous video access, not "
            "specific to this link. Please try again in a moment."
        )

    raise RuntimeError(
        f"Could not download this video. It may be private, deleted, or "
        f"the link may not point to a video. ({last_error})"
    )
