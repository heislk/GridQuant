"""Render the title-forecast animation to media/2026_title_forecast.mp4.

Needs ffmpeg on PATH. Run: python predict2026/viz/render.py
"""
import os, runpy
os.environ["RENDER"] = "1"
runpy.run_path(os.path.join(os.path.dirname(__file__), "animation.py"), run_name="__main__")
