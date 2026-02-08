"""
Gemini AI Service for generating detailed OpenScore explanations
and quarterback performance analysis.
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiService:
    """Service that uses Google Gemini to generate rich, contextual explanations."""

    def __init__(self):
        self.model = None
        self._initialize()

    def _initialize(self):
        """Initialize the Gemini model with API key from environment."""
        if not GEMINI_AVAILABLE:
            print("[GeminiService] google-generativeai not installed. AI explanations disabled.")
            return

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or api_key == "your_gemini_api_key_here":
            print("[GeminiService] GEMINI_API_KEY not set. AI explanations disabled.")
            return

        try:
            self.model = genai.Client()
            print("[GeminiService] Gemini model initialized successfully.")
        except Exception as e:
            print(f"[GeminiService] Failed to initialize Gemini: {e}")
            self.model = None

    @property
    def is_available(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    # Per-player OpenScore explanation
    # ------------------------------------------------------------------

    async def explain_openscore(
        self,
        player_id: str,
        openscore: float,
        context: Dict[str, Any],
    ) -> str:
        """
        Generate a human-readable explanation for why a player's OpenScore
        is what it is.

        Args:
            player_id: e.g. "player_5"
            openscore: The computed OpenScore value (0-100)
            context: Dictionary with contextual metrics such as:
                - nearest_defender_distance (pixels)
                - num_nearby_defenders
                - closing_speed (pixels/sec, positive = closing in)
                - separation_efficiency (0-1)
                - coverage_radius_used (pixels)
                - field_diagonal (pixels)
                - avg_openscore, max_openscore, min_openscore, std_openscore
        Returns:
            A concise but detailed natural-language explanation.
        """
        if not self.is_available:
            return self._fallback_openscore_explanation(openscore, context)

        prompt = self._build_openscore_prompt(player_id, openscore, context)

        try:
            response = await asyncio.to_thread(
                self.model.models.generate_content,
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            print(f"[GeminiService] Gemini call failed: {e}")
            return self._fallback_openscore_explanation(openscore, context)

    def _build_openscore_prompt(
        self, player_id: str, openscore: float, ctx: Dict[str, Any]
    ) -> str:
        nearest_dist = ctx.get("nearest_defender_distance", "unknown")
        num_nearby = ctx.get("num_nearby_defenders", "unknown")
        closing_speed = ctx.get("closing_speed", "unknown")
        separation_eff = ctx.get("separation_efficiency", "unknown")
        avg_score = ctx.get("avg_openscore", openscore)
        max_score = ctx.get("max_openscore", openscore)
        min_score = ctx.get("min_openscore", openscore)

        # Convert pixel distance to a rough yard estimate (field ~53.3 yards wide ≈ 1920px)
        px_per_yard = 1920 / 53.3
        if isinstance(nearest_dist, (int, float)):
            nearest_yards = round(nearest_dist / px_per_yard, 1)
        else:
            nearest_yards = "unknown"

        return f"""You are a sports analyst describing unique receiver situations on the field. Each receiver's situation is different - focus on the SPECIFIC details to create a UNIQUE description.

Data for this receiver RIGHT NOW:
- **Nearest defender**: ~{nearest_yards} yards away
- **Defenders nearby**: {num_nearby}
- **Defender movement**: {closing_speed} px/sec (positive = rushing toward, negative = falling away)
- **Route quality**: {separation_eff} (1.0 = crisp, lower = disrupted)
- **This receiver's range this play**: avg {avg_score:.1f}, peak {max_score:.1f}, low {min_score:.1f}

Create a UNIQUE 1-2 sentence description that captures THIS specific receiver's situation. Use specific details from the data to make it unique, NOT generic templates. Vary your descriptions - use different sentence structures and framings each time.

Examples of good variety:
- "Sprinting into a gap between two converging defenders"
- "A defender is mirroring his routes with tight coverage"
- "Finding wide open grass on the perimeter with clear separation"
- "Carving a path through heavy coverage with efficient footwork"
- "The defender lost a step allowing serious separation opportunity"

Use natural football language. NO OpenScore mention. NO numbers. NO markdown.
"""

    def _fallback_openscore_explanation(
        self, openscore: float, ctx: Dict[str, Any]
    ) -> str:
        """Generate a simple fallback when Gemini is unavailable."""
        num_nearby = ctx.get("num_nearby_defenders", 0)
        nearest_dist = ctx.get("nearest_defender_distance", 0)
        closing_speed = ctx.get("closing_speed", 0)
        separation_eff = ctx.get("separation_efficiency", 0.5)
        
        px_per_yard = 1920 / 53.3
        nearest_yards = round(nearest_dist / px_per_yard, 1) if isinstance(nearest_dist, (int, float)) else 0
        
        # Variable descriptions based on the situation
        if openscore >= 80:
            if num_nearby == 0:
                return "Wide open space downfield with clear separation from defenders."
            else:
                return "Breaking free with only light coverage nearby."
        elif openscore >= 60:
            if isinstance(closing_speed, (int, float)) and closing_speed < 0:
                return "Defender falling back, creating expanding space for the catch."
            else:
                return "Moderate separation from coverage with time to make the play."
        elif openscore >= 40:
            if num_nearby >= 2:
                return "Multiple defenders in the area creating a congested coverage zone."
            else:
                return "Tight window with a defender in close proximity."
        else:
            if isinstance(closing_speed, (int, float)) and closing_speed > 100:
                return "Defender rapidly approaching in tight man coverage."
            else:
                return "Locked down in heavy coverage with very limited throwing window."

    # ------------------------------------------------------------------
    # Quarterback performance summary
    # ------------------------------------------------------------------

    async def explain_qb_performance(
        self,
        feedback_data: Dict[str, Any],
        openscore_summary: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Generate detailed AI-powered explanations for the QB performance summary.

        Returns a dict with keys:
            - summary: Detailed paragraph replacing the generic one-liner
            - strengths_analysis: Deeper analysis of strengths
            - improvement_analysis: Deeper analysis of weaknesses
            - play_reading: Assessment of the QB's read progression
        """
        if not self.is_available:
            print("gemini unavailable")
            return self._fallback_qb_explanation(feedback_data, openscore_summary)

        prompt = self._build_qb_prompt(feedback_data, openscore_summary)

        try:
            response = await asyncio.to_thread(
                self.model.models.generate_content,
                model="gemini-2.5-flash",
                contents=prompt,
            )
            print(response)
            return self._parse_qb_response(response.text.strip(), feedback_data)
        except Exception as e:
            print(f"[GeminiService] Gemini QB analysis failed: {e}")
            return self._fallback_qb_explanation(feedback_data, openscore_summary)

    def _build_qb_prompt(
        self,
        feedback: Dict[str, Any],
        openscore_summary: Dict[str, Any],
    ) -> str:
        # Build per-receiver stats (for context, but anonymized in prompt)
        num_receivers = len(openscore_summary)
        avg_scores = [data['avg_openscore'] for data in openscore_summary.values()]
        overall_avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0
        score_variance = max(avg_scores) - min(avg_scores) if avg_scores else 0

        return f"""You are an expert NFL quarterback coach providing a detailed performance breakdown after reviewing film analysis data.

**Overall Grade**: {feedback.get('overall_grade', 'N/A')} ({feedback.get('overall_score', 0)}/100)

**Key Metrics**:
- Number of pass catchers analyzed: {num_receivers}
- Average separation across all targets: {overall_avg:.1f}
- Range of separation quality: {score_variance:.1f} points

**Identified Strengths**: {json.dumps(feedback.get('strengths', []))}
**Identified Weaknesses**: {json.dumps(feedback.get('areas_for_improvement', []))}

Please provide a detailed analysis in the following JSON format (no markdown code fences, just raw JSON):
{{
  "summary": "A detailed 3-5 sentence paragraph explaining the quarterback's overall performance. Explain what the grade means in football terms, discuss whether targets were getting open or being locked down by the defense, and what the QB should focus on. Use generic references like 'targets', 'pass catchers', 'options' - NO specific receiver names or numbers. Focus on patterns and decision-making.",
  "strengths_analysis": "A 2-3 sentence detailed explanation expanding on the strengths. Avoid naming specific receivers. Explain what the QB did well in football terms and patterns.",
  "improvement_analysis": "A 2-3 sentence detailed explanation of areas for improvement. Be specific about what reads were missed and what defensive looks caused problems, but avoid specific receiver references.",
  "play_reading": "A 2-3 sentence assessment of the QB's ability to read the defense and progress through reads. Discuss overall decision-making without referencing specific receivers or numbers."
}}
"""

    def _parse_qb_response(
        self, text: str, feedback: Dict[str, Any]
    ) -> Dict[str, str]:
        """Parse Gemini's JSON response for QB analysis."""
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Remove first line and last line
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            parsed = json.loads(cleaned)
            return {
                "summary": parsed.get("summary", feedback.get("summary", "")),
                "strengths_analysis": parsed.get("strengths_analysis", ""),
                "improvement_analysis": parsed.get("improvement_analysis", ""),
                "play_reading": parsed.get("play_reading", ""),
            }
        except json.JSONDecodeError:
            # If JSON parsing fails, use the raw text as summary
            return {
                "summary": cleaned[:500] if cleaned else feedback.get("summary", ""),
                "strengths_analysis": "",
                "improvement_analysis": "",
                "play_reading": "",
            }

    def _fallback_qb_explanation(
        self, feedback: Dict[str, Any], openscore_summary: Dict[str, Any]
    ) -> Dict[str, str]:
        """Rule-based fallback when Gemini is unavailable."""
        grade = feedback.get("overall_grade", "N/A")
        score = feedback.get("overall_score", 0)

        # Gather stats (without naming specific receivers)
        avg_scores = [d["avg_openscore"] for d in openscore_summary.values()]
        if not avg_scores:
            return {
                "summary": "Insufficient data to provide a detailed analysis.",
                "strengths_analysis": "",
                "improvement_analysis": "",
                "play_reading": "",
            }

        overall_avg = sum(avg_scores) / len(avg_scores)
        best_avg = max(avg_scores)
        worst_avg = min(avg_scores)
        variance = best_avg - worst_avg

        summary_parts = [
            f"The quarterback earned a grade of {grade} with an overall score of {score}/100."
        ]

        if overall_avg >= 70:
            summary_parts.append(
                "Targets were consistently getting open across the field. The defense struggled to maintain coverage, "
                "creating multiple quality passing windows and scoring opportunities."
            )
        elif overall_avg >= 50:
            summary_parts.append(
                "Targets showed moderate separation from defenders. The defense provided reasonable resistance, "
                "requiring the QB to be selective with reads and identify the most favorable matchups."
            )
        else:
            summary_parts.append(
                "The defense dominated coverage across the board, with tight man-to-man and zone coverage limiting options. "
                "Quick reads and check-downs would have been the safest approach."
            )

        return {
            "summary": " ".join(summary_parts),
            "strengths_analysis": "Made effective decisions when targets created separation and identifed good opportunities when they were available.",
            "improvement_analysis": "Could improve by faster decision-making and better recognition of coverage looks to exploit available windows.",
            "play_reading": f"{'Showed good ability to find open receivers' if score >= 60 else 'Could improve in reading defensive coverage'} and progressing through reads efficiently.",
        }

    # ------------------------------------------------------------------
    # Batch: explain all players' OpenScores at once
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
# Batch: explain all players' OpenScores at once (single Gemini call)
# ------------------------------------------------------------------

    async def explain_all_openscores(
        self,
        openscore_summary: Dict[str, Any],
        player_contexts: Dict[str, Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Generate explanations for all players in one Gemini batch call.
        """

        if not self.is_available:
            # fallback per player
            out = {}
            for player_id, stats in openscore_summary.items():
                ctx = player_contexts.get(player_id, {})
                ctx.update({
                    "avg_openscore": stats.get("avg_openscore", 0),
                    "max_openscore": stats.get("max_openscore", 0),
                    "min_openscore": stats.get("min_openscore", 0),
                    "std_openscore": stats.get("std_openscore", 0),
                })
                out[player_id] = self._fallback_openscore_explanation(
                    stats.get("avg_openscore", 0), ctx
                )
            return out

        # ------------------------
        # Build batch payload
        # ------------------------

        batch_data = {}

        for player_id, stats in openscore_summary.items():
            ctx = player_contexts.get(player_id, {}).copy()
            ctx.update({
                "avg_openscore": stats.get("avg_openscore", 0),
                "max_openscore": stats.get("max_openscore", 0),
                "min_openscore": stats.get("min_opens_score", 0),
                "std_openscore": stats.get("std_openscore", 0),
            })
            batch_data[player_id] = ctx

        prompt = self._build_batch_openscore_prompt(batch_data)

        # ------------------------
        # Single Gemini call
        # ------------------------

        try:
            response = await asyncio.to_thread(
                self.model.models.generate_content,
                model="gemini-2.5-flash",
                contents=prompt,
            )

            return self._parse_batch_openscore_response(response.text, batch_data)

        except Exception as e:
            print(f"[GeminiService] Batch Gemini failed: {e}")

            # fallback safely
            fallback = {}
            for pid, ctx in batch_data.items():
                fallback[pid] = self._fallback_openscore_explanation(
                    ctx.get("avg_openscore", 0), ctx
                )
            return fallback
    def _build_batch_openscore_prompt(self, batch_data: Dict[str, Any]) -> str:
        return f"""
    You are an NFL analyst describing receiver separation.

    For EACH player below, write a UNIQUE 1-2 sentence explanation of how this did or did not get open based on his OpenScore context.
    Stay professional, you may use numbers but don't go over the top. Use yards, to convert the given units into yards please divide by 35.

    Rules:
    - No markdown
    - No OpenScore mention
    - Natural football language
    - Do not call the player by their ID (i.e. no Player 77)
    - Every player must sound different
    - Return JSON only

    Data:
    {json.dumps(batch_data, indent=2)}

    Return format:

    {{
    "player_id": "explanation",
    ...
    }}
    """

    def _parse_batch_openscore_response(
        self,
        text: str,
        batch_data: Dict[str, Any],
    ) -> Dict[str, str]:

        cleaned = text.strip()

        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            parsed = json.loads(cleaned)

            # ensure every player has output
            out = {}
            for pid, ctx in batch_data.items():
                out[pid] = parsed.get(
                    pid,
                    self._fallback_openscore_explanation(
                        ctx.get("avg_openscore", 0), ctx
                    ),
                )
            return out

        except json.JSONDecodeError:
            print("[GeminiService] JSON parse failed — fallback mode")

            fallback = {}
            for pid, ctx in batch_data.items():
                fallback[pid] = self._fallback_openscore_explanation(
                    ctx.get("avg_openscore", 0), ctx
                )
            return fallback




# Singleton instance
gemini_service = GeminiService()
