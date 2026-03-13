"""Shared constants for the brand-voice fine-tuning workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
GGUF_DIR = ARTIFACTS_DIR / "gguf"
MERGED_DIR = ARTIFACTS_DIR / "merged"
NOTEBOOK_PATH = ROOT / "notebooks" / "qwen35_brand_voice_llamacpp_experiment.ipynb"

# Export zip paths are intentionally not hardcoded.
# Pass explicit `--linkedin-zip` and `--x-zip` paths when running prep_social_exports.py.
DEFAULT_LINKEDIN_ZIP = Path("linkedin-export.zip")
DEFAULT_X_ZIP = Path("x-export.zip")

# Raw and legacy processed artifacts.
PUBLIC_POSTS_PATH = DATA_DIR / "public_posts.jsonl"
LABELED_POSTS_PATH = DATA_DIR / "public_posts_labeled.jsonl"
TRAIN_PATH = DATA_DIR / "train.jsonl"
EVAL_PATH = DATA_DIR / "eval.jsonl"
EVAL_PROMPTS_PATH = DATA_DIR / "eval_prompts.jsonl"
EVAL_TEXT_PATH = ARTIFACTS_DIR / "eval_text.txt"
BASELINE_OUTPUTS_PATH = ARTIFACTS_DIR / "baseline_outputs.jsonl"
FINETUNED_OUTPUTS_PATH = ARTIFACTS_DIR / "finetuned_outputs.jsonl"
EVAL_SUMMARY_PATH = ARTIFACTS_DIR / "eval_summary.md"
MEMORIZATION_REPORT_PATH = ARTIFACTS_DIR / "memorization_report.json"

# Run 1 curated dataset and evaluation artifacts.
PUBLIC_POSTS_REVIEW_PATH = DATA_DIR / "public_posts_review.csv"
CURATED_PUBLIC_POSTS_PATH = DATA_DIR / "public_posts_run1_curated.jsonl"
RUN1_LABELED_POSTS_PATH = DATA_DIR / "public_posts_labeled_run1.jsonl"
RUN1_TRAIN_PATH = DATA_DIR / "train_run1_qwen35_9b.jsonl"
RUN1_EVAL_PATH = DATA_DIR / "eval_run1_qwen35_9b.jsonl"
RUN1_RECONSTRUCTION_PROMPTS_PATH = DATA_DIR / "reconstruction_prompts_run1_qwen35_9b.jsonl"
RUN1_EVAL_SUITE_PATH = DATA_DIR / "eval_suite_run1_social_20x3.jsonl"

RUN1_EVAL_DIR = ARTIFACTS_DIR / "eval" / "run1"
RUN1_EVAL_TEXT_PATH = RUN1_EVAL_DIR / "eval_text.txt"
RUN1_BASELINE_OUTPUTS_PATH = RUN1_EVAL_DIR / "baseline_qwen35_9b_outputs.jsonl"
RUN1_FINETUNED_OUTPUTS_PATH = RUN1_EVAL_DIR / "finetuned_qwen35_9b_outputs.jsonl"
RUN1_RECONSTRUCTION_BASELINE_PATH = RUN1_EVAL_DIR / "reconstruction_baseline.jsonl"
RUN1_RECONSTRUCTION_FINETUNED_PATH = RUN1_EVAL_DIR / "reconstruction_finetuned.jsonl"
RUN1_PERPLEXITY_BASELINE_PATH = RUN1_EVAL_DIR / "perplexity_baseline.txt"
RUN1_PERPLEXITY_FINETUNED_PATH = RUN1_EVAL_DIR / "perplexity_finetuned.txt"
RUN1_MEMORIZATION_REPORT_PATH = RUN1_EVAL_DIR / "memorization_report.json"
RUN1_BLIND_REVIEW_PATH = RUN1_EVAL_DIR / "blind_review.csv"
RUN1_BLIND_REVIEW_KEY_PATH = RUN1_EVAL_DIR / "blind_review_key.csv"
RUN1_HUMAN_SCORES_PATH = RUN1_EVAL_DIR / "human_scores.csv"
RUN1_BLIND_EVAL_SUMMARY_PATH = RUN1_EVAL_DIR / "blind_eval_summary.md"
RUN1_SUMMARY_PATH = RUN1_EVAL_DIR / "summary.md"

QWEN35_9B_BASELINE_GGUF_PATH = GGUF_DIR / "qwen35_9b_baseline_q4_k_m.gguf"
QWEN35_9B_FINETUNED_GGUF_PATH = GGUF_DIR / "qwen35_9b_brand_voice_q4_k_m.gguf"
QWEN35_9B_LORA_DIR = ROOT / "qwen35_9b_brand_voice_lora"
QWEN35_9B_MERGED_DIR = MERGED_DIR / "qwen35_9b_brand_voice_merged"

ASSISTANT_SYSTEM_PROMPT = (
    "Match the author's public writing voice. Be direct, first-person, thoughtful, "
    "and specific. Keep the platform style appropriate. Do not copy exact names, "
    "links, or handles from prior posts."
)

EVAL_SYSTEM_PROMPT = (
    "You write public-facing content. Be direct, specific, readable, and original. "
    "Avoid copied phrasing, handles, or links unless explicitly requested. Match "
    "the requested platform and format."
)

POST_TYPES = (
    "build_update",
    "opinion",
    "observation",
    "question",
    "resource_share",
)

REVIEW_DROP_REASONS = (
    "job_search",
    "course_or_certificate",
    "employment_announcement",
    "event_only",
    "low_signal_short",
    "generic_resource_share",
    "off_topic_personal",
    "non_current_voice",
    "other",
)

RUN1_SUITE_NAME = "run1_social_voice_v1"
RUN1_SEEDS = (3407, 3408, 3409)
CURATION_RECENCY_CUTOFF = "2025-06-01T00:00:00Z"
CURATION_TRANSITION_CUTOFF = "2024-08-01T00:00:00Z"
CURATION_MIN_ROWS = 180
CURATION_TARGET_MIN = 200
CURATION_TARGET_MAX = 240
CURATION_MIN_LINKEDIN_ROWS = 30

BUCKET_WEIGHTS = {
    "x_topic": 0.35,
    "response": 0.30,
    "various_topic": 0.20,
    "blog": 0.15,
}

BUCKET_PRIMARY_METRIC = {
    "x_topic": "engagement_likelihood",
    "blog": "structure_depth",
    "various_topic": "topical_competence",
    "response": "conversational_value",
}

BUCKET_METRICS = {
    "x_topic": (
        "authenticity",
        "engagement_likelihood",
        "specificity",
        "platform_fit",
        "readability",
        "originality",
    ),
    "blog": (
        "authenticity",
        "structure_depth",
        "specificity",
        "readability",
        "originality",
        "coherence",
    ),
    "various_topic": (
        "authenticity",
        "topical_competence",
        "specificity",
        "voice_stability",
        "readability",
        "originality",
    ),
    "response": (
        "authenticity",
        "conversational_value",
        "judgment",
        "specificity",
        "platform_fit",
        "readability",
    ),
}

ALL_REVIEW_METRICS = tuple(
    sorted({metric for metrics in BUCKET_METRICS.values() for metric in metrics})
)


@dataclass(frozen=True)
class FixedPrompt:
    prompt_id: str
    platform: str
    prompt_text: str
    max_tokens: int


FIXED_PROMPTS = (
    FixedPrompt(
        prompt_id="suite_x_parallel_agents",
        platform="x",
        prompt_text=(
            "Write an X post about what changed after using multiple AI agents in "
            "parallel on one product."
        ),
        max_tokens=180,
    ),
    FixedPrompt(
        prompt_id="suite_x_ai_efficiency_layoffs",
        platform="x",
        prompt_text=(
            "Write an X post reacting to companies cutting jobs because of AI "
            "efficiency gains."
        ),
        max_tokens=180,
    ),
    FixedPrompt(
        prompt_id="suite_x_ai_debugging_debt",
        platform="x",
        prompt_text=(
            "Write an X post on when AI coding speeds you up and when it creates "
            "debugging debt."
        ),
        max_tokens=180,
    ),
    FixedPrompt(
        prompt_id="suite_x_agent_validation",
        platform="x",
        prompt_text=(
            "Write an X post asking builders how they validate agentic AI products "
            "before shipping."
        ),
        max_tokens=180,
    ),
    FixedPrompt(
        prompt_id="suite_linkedin_enterprise_startup_lessons",
        platform="linkedin",
        prompt_text=(
            "Write a LinkedIn post about a lesson from shipping AI systems across "
            "enterprise and startup environments."
        ),
        max_tokens=420,
    ),
    FixedPrompt(
        prompt_id="suite_linkedin_speed_eval_reliability",
        platform="linkedin",
        prompt_text=(
            "Write a LinkedIn post about balancing speed, evaluation, and "
            "reliability in LLM products."
        ),
        max_tokens=420,
    ),
    FixedPrompt(
        prompt_id="suite_linkedin_ai_engineering_roles",
        platform="linkedin",
        prompt_text=(
            "Write a LinkedIn post reflecting on how AI is changing engineering "
            "roles without sounding alarmist."
        ),
        max_tokens=420,
    ),
    FixedPrompt(
        prompt_id="suite_linkedin_side_project_update",
        platform="linkedin",
        prompt_text=(
            "Write a LinkedIn post sharing a recent build update from an AI side "
            "project and the practical insight it taught you."
        ),
        max_tokens=420,
    ),
)


@dataclass(frozen=True)
class PromptSpec:
    suite_name: str
    bucket: str
    prompt_id: str
    platform: str
    prompt_text: str
    max_tokens: int
    constraints: tuple[str, ...]
    seeds: tuple[int, ...] = RUN1_SEEDS


RUN1_PROMPT_SPECS = (
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="x_topic",
        prompt_id="x_parallel_agents",
        platform="x",
        prompt_text="Write a single X post about what changed after using multiple AI agents in parallel on one product.",
        max_tokens=180,
        constraints=(
            "One post only.",
            "Do not write a thread.",
            "Stay under 280 characters.",
            "Do not copy handles or links.",
            "Avoid hashtags unless they feel necessary.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="x_topic",
        prompt_id="x_debugging_debt",
        platform="x",
        prompt_text="Write a single X post on when AI coding speeds you up and when it creates debugging debt.",
        max_tokens=180,
        constraints=(
            "One post only.",
            "Do not write a thread.",
            "Stay under 280 characters.",
            "Do not copy handles or links.",
            "Avoid hashtags unless they feel necessary.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="x_topic",
        prompt_id="x_ai_efficiency_layoffs",
        platform="x",
        prompt_text="Write a single X post reacting to companies cutting jobs because of AI efficiency gains.",
        max_tokens=180,
        constraints=(
            "One post only.",
            "Do not write a thread.",
            "Stay under 280 characters.",
            "Do not copy handles or links.",
            "Avoid hashtags unless they feel necessary.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="x_topic",
        prompt_id="x_evals_vs_demos",
        platform="x",
        prompt_text="Write a single X post about why evals matter more than flashy demos in LLM products.",
        max_tokens=180,
        constraints=(
            "One post only.",
            "Do not write a thread.",
            "Stay under 280 characters.",
            "Do not copy handles or links.",
            "Avoid hashtags unless they feel necessary.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="x_topic",
        prompt_id="x_agent_validation",
        platform="x",
        prompt_text="Write a single X post asking builders how they validate agentic AI products before shipping.",
        max_tokens=180,
        constraints=(
            "One post only.",
            "Do not write a thread.",
            "Stay under 280 characters.",
            "Do not copy handles or links.",
            "Avoid hashtags unless they feel necessary.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="blog",
        prompt_id="blog_demo_to_prod",
        platform="blog",
        prompt_text='Write a blog post: "What actually breaks when you move an AI workflow from demo to production."',
        max_tokens=1400,
        constraints=(
            "Target 800-1200 words.",
            "Include a strong headline.",
            "Use clear section headings.",
            "Keep the tone practical, not generic.",
            "Avoid padded introductions.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="blog",
        prompt_id="blog_ai_engineering_roles",
        platform="blog",
        prompt_text='Write a blog post: "How AI is changing engineering roles without the usual hype or panic."',
        max_tokens=1400,
        constraints=(
            "Target 800-1200 words.",
            "Include a strong headline.",
            "Use clear section headings.",
            "Keep the tone practical, not generic.",
            "Avoid padded introductions.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="blog",
        prompt_id="blog_evals_guardrails",
        platform="blog",
        prompt_text='Write a blog post: "Why evals, guardrails, and fallback logic matter more than model size."',
        max_tokens=1400,
        constraints=(
            "Target 800-1200 words.",
            "Include a strong headline.",
            "Use clear section headings.",
            "Keep the tone practical, not generic.",
            "Avoid padded introductions.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="blog",
        prompt_id="blog_agent_ownership",
        platform="blog",
        prompt_text='Write a blog post: "A practical framework for deciding when an agent should own a workflow."',
        max_tokens=1400,
        constraints=(
            "Target 800-1200 words.",
            "Include a strong headline.",
            "Use clear section headings.",
            "Keep the tone practical, not generic.",
            "Avoid padded introductions.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="blog",
        prompt_id="blog_startup_vs_enterprise",
        platform="blog",
        prompt_text='Write a blog post: "Lessons from shipping AI systems in startup and enterprise environments."',
        max_tokens=1400,
        constraints=(
            "Target 800-1200 words.",
            "Include a strong headline.",
            "Use clear section headings.",
            "Keep the tone practical, not generic.",
            "Avoid padded introductions.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="various_topic",
        prompt_id="var_learning_in_public",
        platform="linkedin",
        prompt_text="Write a LinkedIn-style post about learning in public while building a technical product.",
        max_tokens=350,
        constraints=(
            "Target 150-300 words.",
            "Keep it public-facing.",
            "Make it useful, not motivational fluff.",
            "Do not copy links or handles.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="various_topic",
        prompt_id="var_hiring_ai_first",
        platform="linkedin",
        prompt_text="Write a LinkedIn-style post about hiring strong engineers in an AI-first environment.",
        max_tokens=350,
        constraints=(
            "Target 150-300 words.",
            "Keep it public-facing.",
            "Make it useful, not motivational fluff.",
            "Do not copy links or handles.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="various_topic",
        prompt_id="var_speed_vs_reliability",
        platform="linkedin",
        prompt_text="Write a LinkedIn-style post about balancing speed and reliability on a small team.",
        max_tokens=350,
        constraints=(
            "Target 150-300 words.",
            "Keep it public-facing.",
            "Make it useful, not motivational fluff.",
            "Do not copy links or handles.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="various_topic",
        prompt_id="var_remote_decisions",
        platform="public_post",
        prompt_text="Write a short public post about how remote work changes product decision-making.",
        max_tokens=350,
        constraints=(
            "Target 150-300 words.",
            "Keep it public-facing.",
            "Make it useful, not motivational fluff.",
            "Do not copy links or handles.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="various_topic",
        prompt_id="var_taste_vs_execution",
        platform="public_post",
        prompt_text="Write a short public post about the difference between taste and execution in software products.",
        max_tokens=350,
        constraints=(
            "Target 150-300 words.",
            "Keep it public-facing.",
            "Make it useful, not motivational fluff.",
            "Do not copy links or handles.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="response",
        prompt_id="resp_agents_toys",
        platform="x",
        prompt_text='Reply on X to: "AI agents are mostly toys until they can own a real workflow end to end."',
        max_tokens=220,
        constraints=(
            "Use platform-appropriate length.",
            "Be direct and respectful.",
            "Be opinionated without sounding combative.",
            "Do not copy handles or links.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="response",
        prompt_id="resp_copilot_headcount",
        platform="x",
        prompt_text='Reply on X to: "Copilot made every engineer 2x faster, so cutting headcount is rational."',
        max_tokens=220,
        constraints=(
            "Use platform-appropriate length.",
            "Be direct and respectful.",
            "Be opinionated without sounding combative.",
            "Do not copy handles or links.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="response",
        prompt_id="resp_enterprise_audit",
        platform="linkedin",
        prompt_text='Reply to a LinkedIn comment: "Enterprise AI is all smoke and mirrors until it can pass an audit."',
        max_tokens=220,
        constraints=(
            "Use platform-appropriate length.",
            "Be direct and respectful.",
            "Be opinionated without sounding combative.",
            "Do not copy handles or links.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="response",
        prompt_id="resp_validate_before_launch",
        platform="linkedin",
        prompt_text='Reply to a founder asking: "How do you know if an agentic product is actually useful before launch?"',
        max_tokens=220,
        constraints=(
            "Use platform-appropriate length.",
            "Be direct and respectful.",
            "Be opinionated without sounding combative.",
            "Do not copy handles or links.",
        ),
    ),
    PromptSpec(
        suite_name=RUN1_SUITE_NAME,
        bucket="response",
        prompt_id="resp_debugging_work",
        platform="linkedin",
        prompt_text='Reply to a skeptical engineer saying: "LLMs create more debugging work than they save."',
        max_tokens=220,
        constraints=(
            "Use platform-appropriate length.",
            "Be direct and respectful.",
            "Be opinionated without sounding combative.",
            "Do not copy handles or links.",
        ),
    ),
)
