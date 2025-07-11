"""Text classification module for disinformation relevance detection."""

import re
from typing import Any


class ClassifierImpl:
    """Mock classifier implementation using keyword-based heuristics."""

    def __init__(self) -> None:
        """Initialize the classifier with keyword patterns."""
        # Keywords that suggest content is relevant to disinformation analysis
        self.relevant_keywords = {
            "politics",
            "political",
            "government",
            "election",
            "voting",
            "democracy",
            "policy",
            "politician",
            "parliament",
            "congress",
            "senate",
            "minister",
            "campaign",
            "candidate",
            "ballot",
            "referendum",
            "party",
            "coalition",
            "legislation",
            "law",
            "regulation",
            "public",
            "citizen",
            "civic",
            "administration",
            "authority",
            "official",
            "state",
            "federal",
            "local",
            "disinformation",
            "misinformation",
            "fake news",
            "propaganda",
            "bias",
            "conspiracy",
            "rumor",
            "false",
            "misleading",
            "fact-check",
            "verify",
            "social media",
            "facebook",
            "twitter",
            "instagram",
            "tiktok",
            "youtube",
            "news",
            "media",
            "journalism",
            "reporter",
            "broadcast",
            "press",
        }

        # Compile regex pattern for efficient matching
        pattern = r"\b(?:" + "|".join(re.escape(keyword) for keyword in self.relevant_keywords) + r")\b"
        self.keyword_pattern = re.compile(pattern, re.IGNORECASE)

    def classify_batch(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Classify a batch of texts and return results with classifications."""
        if not data:
            return []

        results = []
        for item in data:
            # Create a copy of the original item
            result_item = item.copy()

            # Extract text for classification
            text = item.get("text", "")

            # Perform keyword-based classification
            classification, confidence = self._classify_text(text)

            # Add classification results
            result_item["classification"] = classification
            result_item["confidence"] = confidence

            results.append(result_item)

        return results

    def _classify_text(self, text: str) -> tuple[str, float]:
        """Classify a single text using keyword heuristics."""
        if not text:
            # No text to analyze
            return "not_relevant", 0.1

        # Find keyword matches
        matches = self.keyword_pattern.findall(text.lower())
        match_count = len(matches)

        # Calculate confidence based on keyword density
        word_count = len(text.split())
        keyword_density = 0.0 if word_count == 0 else match_count / word_count

        # Classification thresholds
        high_match_threshold = 3
        high_density_threshold = 0.1

        # Classify based on keyword presence and density
        if match_count == 0:
            # No relevant keywords found
            return "not_relevant", 0.2
        if match_count >= high_match_threshold or keyword_density >= high_density_threshold:
            # High confidence: multiple keywords or high density
            confidence = min(0.9, 0.6 + keyword_density * 2)
            return "relevant", confidence
        # Low to medium confidence: some keywords present
        confidence = min(0.8, 0.4 + keyword_density * 3)
        return "relevant", confidence
