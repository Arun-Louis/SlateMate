

import re
import os
import logging
import anthropic
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are an expert in Manim Community Edition (ManimCE) v0.20.
Generate a complete, executable ManimCE Python script to animate a concept from a lecture transcript.

STRICT RULES — follow exactly:
1. Start with: from manim import *
2. Do NOT import numpy or any other library
3. Define exactly ONE class named AnimationScene(Scene)
4. Implement construct(self) method
5. Output ONLY raw Python code — no markdown, no explanation

ALLOWED objects: Text, MathTex, Arrow, Line, Circle, Square, Rectangle, Dot, NumberLine, Axes, VGroup
ALLOWED methods: self.play(), self.wait(), Write(), Create(), FadeIn(), FadeOut(), Transform(), ReplacementTransform(), animate.

FORBIDDEN — never use these, they cause errors:
- axes.get_vector() — does not exist
- axes.get_line() — does not exist
- ShowCreation() — use Create() instead
- ShowPassingFlash() — do not use
- Vector(start=..., end=...) — use Arrow(start=..., end=..., buff=0) instead
- NumberLine with y_range — NumberLine only takes x_range
- numpy arrays as positions — use UP, DOWN, LEFT, RIGHT, ORIGIN only
- import numpy — forbidden
- .end attribute on arrows — use .get_end() instead

CORRECT examples:
    # Arrow/vector:
    arrow = Arrow(start=ORIGIN, end=RIGHT*2, buff=0, color=YELLOW)
    self.play(Create(arrow))

    # Math equation:
    eq = MathTex(r"A\\vec{v} = \\lambda \\vec{v}")
    self.play(Write(eq))

    # Axes:
    ax = Axes(x_range=[-4, 4], y_range=[-3, 3])
    self.play(Create(ax))

    # Number line:
    nl = NumberLine(x_range=[-4, 4])
    self.play(Create(nl))

Keep it simple — max 5 animation steps. Short and clean."""

USER_PROMPT = """Here is what the professor said in the last 30 seconds:

"{transcript}"

Generate a simple ManimCE animation for the most important concept. Output only raw Python code."""


@dataclass
class GenerationResult:
    success: bool
    code: Optional[str]
    concept: Optional[str]
    error: Optional[str]


class ManimCodeGenerator:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
            )
        self.client = anthropic.Anthropic(api_key=key)
        logger.info(f"ManimCodeGenerator ready (model: {CLAUDE_MODEL})")

    def generate(self, transcript: str) -> GenerationResult:
        prompt = USER_PROMPT.format(transcript=transcript)
        try:
            logger.info("Sending transcript to Claude...")
            message = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            raw  = message.content[0].text
            code = self._extract_code(raw)

            if not code:
                return GenerationResult(
                    success=False, code=None, concept=None,
                    error="Model returned empty response."
                )

            if not self._is_valid_python(code):
                return GenerationResult(
                    success=False, code=None, concept=None,
                    error="Generated output failed Python syntax check."
                )

            concept = self._extract_concept(code)
            logger.info(f"Code generated successfully. Concept: {concept}")
            return GenerationResult(success=True, code=code, concept=concept, error=None)

        except anthropic.RateLimitError:
            return GenerationResult(
                success=False, code=None, concept=None,
                error="Rate limit hit. Wait a moment and try again."
            )
        except anthropic.AuthenticationError:
            return GenerationResult(
                success=False, code=None, concept=None,
                error="Invalid API key. Check your ANTHROPIC_API_KEY."
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return GenerationResult(success=False, code=None, concept=None, error=str(e))

    @staticmethod
    def _extract_code(raw: str) -> str:
        match = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        if "from manim import" in raw or "class AnimationScene" in raw:
            return raw.strip()
        return raw.strip()

    @staticmethod
    def _is_valid_python(code: str) -> bool:
        try:
            compile(code, "<string>", "exec")
            return True
        except SyntaxError as e:
            logger.warning(f"Syntax error: {e}")
            return False

    @staticmethod
    def _extract_concept(code: str) -> str:
        match = re.search(r"#.*?concept[:\s]+(.+)", code, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r'"""(.+?)"""', code, re.DOTALL)
        if match:
            return match.group(1).strip()[:80]
        return "STEM concept"


# ── Standalone test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_transcript = """
    So when we multiply a matrix by a special vector, something interesting happens.
    The vector doesn't rotate at all, it just gets scaled by a constant lambda.
    That constant is what we call the eigenvalue, and that special vector
    is called the eigenvector.
    """

    generator = ManimCodeGenerator()
    result    = generator.generate(test_transcript)

    if result.success:
        print(f"\nConcept: {result.concept}")
        print(f"\n── Generated Code ──\n{result.code}")
    else:
        print(f"\nFailed: {result.error}")