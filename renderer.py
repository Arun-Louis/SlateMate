

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from code_sanitizer import sanitize

PYTHON_EXE = r"C:\Users\arunl\AppData\Local\Programs\Python\Python314\python.exe"


@dataclass
class RenderResult:
    success: bool
    video_path: Optional[str]
    error: Optional[str]


def render(code: str, output_dir: str = "output") -> RenderResult:
    Path(output_dir).mkdir(exist_ok=True)
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    # Sanitize code before writing
    code = sanitize(code)

    scene_file = temp_dir / "animation_scene.py"
    scene_file.write_text(code, encoding="utf-8")

    scene_file_abs = str(scene_file.resolve())
    output_dir_abs = str(Path(output_dir).resolve())

    bat_file = temp_dir / "run_manim.bat"
    bat_content = (
        f'@echo off\n'
        f'"{PYTHON_EXE}" -m manim "{scene_file_abs}" AnimationScene '
        f'-ql --media_dir "{output_dir_abs}" --disable_caching\n'
    )
    bat_file.write_text(bat_content, encoding="utf-8")

    print("Rendering animation...")
    ret = os.system(f'"{str(bat_file.resolve())}"')

    if ret != 0:
        return RenderResult(
            success=False,
            video_path=None,
            error=f"Manim render failed with exit code {ret}"
        )

    video_path = _find_output_video(output_dir)
    if not video_path:
        return RenderResult(
            success=False,
            video_path=None,
            error="Render completed but no video file found."
        )

    print(f"Animation rendered: {video_path}")
    return RenderResult(success=True, video_path=video_path, error=None)


def _find_output_video(output_dir: str) -> Optional[str]:
    mp4_files = list(Path(output_dir).rglob("*.mp4"))
    if not mp4_files:
        return None
    return str(max(mp4_files, key=lambda f: f.stat().st_mtime))


if __name__ == "__main__":
    test_code = """from manim import *
import numpy as np

class AnimationScene(Scene):
    def construct(self):
        title = Text("Eigenvalue Transformation", font_size=36)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        arrow = Arrow(start=ORIGIN, end=RIGHT*2, buff=0, color=YELLOW)
        label = MathTex(r"A\\vec{v} = \\lambda \\vec{v}")
        label.next_to(arrow, DOWN)
        self.play(Create(arrow), Write(label))
        self.wait(2)
"""
    result = render(test_code)
    if result.success:
        print(f"\nSuccess! Video at: {result.video_path}")
    else:
        print(f"\nRender failed:\n{result.error}")