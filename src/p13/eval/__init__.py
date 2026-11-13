"""Evaluation: metrics, pairwise LLM judge, human check, final report. Owner: Nioosha.

Everything here consumes ``GeneratedStory`` / ``StoryRecord`` JSONL only – it never imports
model code, so it works the same for any model or adapter.
"""
