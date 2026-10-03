"""Entry point:  python main.py [--config config.json] [--source video.mp4|rtsp://...]"""
import argparse
import signal

from src.config import Config
from src.logger import setup_logging
from src.pipeline import VisitorPipeline


def main():
    ap = argparse.ArgumentParser(description="Intelligent Face Tracker & Visitor Counter")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--source", help="override config source (file path or RTSP URL)")
    args = ap.parse_args()

    cfg = Config(args.config)
    if args.source:
        cfg.source = args.source
    logger = setup_logging(cfg.log_dir)
    pipe = VisitorPipeline(cfg, logger)

    def _sig(*_):                       # graceful stop => exits are still flushed
        pipe.stop = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    pipe.run()


if __name__ == "__main__":
    main()
