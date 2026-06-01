

import re


def sanitize(code: str) -> str:
    """Apply all fixes in sequence."""
    code = fix_imports(code)
    code = fix_vector_class(code)
    code = fix_numberline(code)
    code = fix_numpy_arrays(code)
    code = fix_deprecated_methods(code)
    code = fix_dot_end(code)
    return code


def fix_imports(code: str) -> str:

    code = re.sub(r"^import numpy as np\n?", "", code, flags=re.MULTILINE)
    code = re.sub(r"^from numpy.*\n?", "", code, flags=re.MULTILINE)
    return code


def fix_vector_class(code: str) -> str:
    """
    Vector(start=ORIGIN, end=X) is wrong.
    Vector only takes a direction: Vector(direction).
    Replace with Arrow(start=ORIGIN, end=X, buff=0) which is correct.
    """
    # Vector(start=..., end=..., ...) → Arrow(start=..., end=..., buff=0, ...)
    def replace_vector(match):
        inner = match.group(1)
       
        if "buff=" in inner:
            return f"Arrow({inner})"
        else:
            return f"Arrow({inner}, buff=0)"

    code = re.sub(r"Vector\(([^)]+start=[^)]+end=[^)]+)\)", replace_vector, code)
    code = re.sub(r"Vector\(([^)]+end=[^)]+start=[^)]+)\)", replace_vector, code)
    return code


def fix_numberline(code: str) -> str:
    """NumberLine only takes x_range, not y_range."""
    code = re.sub(
        r"(NumberLine\([^)]*?)y_range=\[[^\]]*\],?\s*",
        r"\1",
        code
    )
  
    code = re.sub(r",\s*\)", ")", code)
    return code


def fix_numpy_arrays(code: str) -> str:
  
    def add_z(match):
        x = match.group(1).strip()
        y = match.group(2).strip()
        try:
            xf = float(x)
            yf = float(y)
            parts = []
            if xf != 0:
                parts.append(f"{xf}*RIGHT")
            if yf != 0:
                parts.append(f"{yf}*UP")
            return " + ".join(parts) if parts else "ORIGIN"
        except ValueError:
            return f"{x}*RIGHT + {y}*UP"

    code = re.sub(
        r"np\.array\(\[([^,\]]+),\s*([^,\]]+)\]\)",
        add_z,
        code
    )
    # Remove remaining np. references
    code = re.sub(r"np\.\w+\([^)]*\)\s*\*\s*\d+", "RIGHT*2", code)
    return code


def fix_deprecated_methods(code: str) -> str:
    """Replace deprecated/nonexistent Manim methods."""
    replacements = {
        r"ShowCreation\(": "Create(",
        r"ShowPassingFlash\(": "Create(",
        r"axes\.get_vector\(": "Arrow(ORIGIN, ",
        r"ax\.get_vector\(": "Arrow(ORIGIN, ",
    }
    for old, new in replacements.items():
        code = re.sub(old, new, code)
    return code


def fix_dot_end(code: str) -> str:
    """
    Dot(end=...) and accessing .end on arrows don't work.
    Replace with safe alternatives.
    """
    # Dot(end=X) → Dot(point=X)  or just Dot(X)
    code = re.sub(r"Dot\(end=([^,)]+)", r"Dot(point=\1", code)
    # arrow.end → arrow.get_end()
    code = re.sub(r"(\w+)\.end\b", r"\1.get_end()", code)
    return code


if __name__ == "__main__":
    test_code = """from manim import *
import numpy as np

class AnimationScene(Scene):
    def construct(self):
        line = NumberLine(x_range=[-4, 4], y_range=[-4, 4])
        i_hat = Vector(start=ORIGIN, end=RIGHT*3, buff=0, color=YELLOW)
        vec = np.array([3, -2])
        tip = Dot(end=i_hat.end, radius=0.1)
        self.play(ShowCreation(line))
"""
    print("Before:")
    print(test_code)
    print("\nAfter:")
    print(sanitize(test_code))