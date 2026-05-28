#!/usr/bin/env python3
"""Download audio or video from YouTube using yt-dlp."""

import argparse
import sys
from pathlib import Path


def download(
    url: str,
    mode: str,
    output_dir: str,
    audio_format: str,
    video_quality: str,
) -> None:
    try:
        import yt_dlp
    except ImportError:
        print("yt-dlp not installed. Run: pip install --break-system-packages yt-dlp")
        sys.exit(1)

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    outtmpl = str(output_path / "%(title)s.%(ext)s")

    if mode == "audio":
        ydl_opts: dict = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": audio_format,
                    "preferredquality": "192",
                }
            ],
        }
    else:
        fmt = f"bestvideo[height<={video_quality}]+bestaudio/best[height<={video_quality}]"
        ydl_opts = {
            "format": fmt,
            "outtmpl": outtmpl,
            "merge_output_format": "mp4",
        }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get("title", "unknown") if info else "unknown"
        duration = info.get("duration_string", "?") if info else "?"
        print(f"Title   : {title}")
        print(f"Duration: {duration}")
        print(f"Saving to: {output_path}")
        print()
        ydl.download([url])

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download audio or video from YouTube.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download audio as MP3 (default)
  python yt_downloader.py https://youtu.be/vJlFxD_Uavk

  # Download audio as OGG
  python yt_downloader.py https://youtu.be/vJlFxD_Uavk --mode audio --audio-format vorbis

  # Download video at 1080p
  python yt_downloader.py https://youtu.be/vJlFxD_Uavk --mode video --quality 1080

  # Custom output directory
  python yt_downloader.py https://youtu.be/vJlFxD_Uavk -o ~/Music
        """,
    )
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument(
        "--mode",
        choices=["audio", "video"],
        default="audio",
        help="Download audio or video (default: audio)",
    )
    parser.add_argument(
        "--audio-format",
        choices=["mp3", "m4a", "opus", "vorbis", "wav", "flac"],
        default="mp3",
        help="Audio format when mode=audio (default: mp3)",
    )
    parser.add_argument(
        "--quality",
        choices=["360", "480", "720", "1080", "1440", "2160"],
        default="1080",
        help="Max video height in pixels when mode=video (default: 1080)",
    )
    parser.add_argument(
        "-o", "--output",
        default="~/Downloads",
        help="Output directory (default: ~/Downloads)",
    )

    args = parser.parse_args()
    download(
        url=args.url,
        mode=args.mode,
        output_dir=args.output,
        audio_format=args.audio_format,
        video_quality=args.quality,
    )


if __name__ == "__main__":
    main()
