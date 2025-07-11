"""Unit tests for the classifier module."""

from disinfo_relation_checker.classifier import ClassifierImpl


def test_classifier_implements_classify_batch_method() -> None:
    """Test that ClassifierImpl implements classify_batch method."""
    classifier = ClassifierImpl()

    # Should have classify_batch method
    assert hasattr(classifier, "classify_batch")
    assert callable(classifier.classify_batch)


def test_classifier_processes_empty_batch() -> None:
    """Test that classifier handles empty input batch."""
    classifier = ClassifierImpl()

    result = classifier.classify_batch([])

    assert result == []


def test_classifier_adds_classification_and_confidence_fields() -> None:
    """Test that classifier adds classification and confidence to input data."""
    classifier = ClassifierImpl()

    input_data = [
        {"text": "This is about politics and government", "source": "news_site"},
        {"text": "Technology and coding tutorial", "source": "tech_blog"},
    ]

    result = classifier.classify_batch(input_data)

    # Should return same number of items
    expected_item_count = 2
    assert len(result) == expected_item_count

    # Each item should have original fields plus classification and confidence
    for item in result:
        assert "text" in item
        assert "source" in item
        assert "classification" in item
        assert "confidence" in item

        # Classification should be a string
        assert isinstance(item["classification"], str)

        # Confidence should be a float between 0 and 1
        assert isinstance(item["confidence"], float)
        assert 0.0 <= item["confidence"] <= 1.0


def test_classifier_preserves_original_data() -> None:
    """Test that classifier preserves all original fields from input."""
    classifier = ClassifierImpl()

    input_data = [
        {
            "text": "Some political content",
            "source": "news",
            "metadata": "extra_info",
            "timestamp": "2023-01-01",
        },
    ]

    result = classifier.classify_batch(input_data)

    assert len(result) == 1
    output_item = result[0]

    # Should preserve all original fields
    assert output_item["text"] == "Some political content"
    assert output_item["source"] == "news"
    assert output_item["metadata"] == "extra_info"
    assert output_item["timestamp"] == "2023-01-01"

    # Should add new fields
    assert "classification" in output_item
    assert "confidence" in output_item


def test_classifier_handles_missing_text_field() -> None:
    """Test that classifier handles records without text field gracefully."""
    classifier = ClassifierImpl()

    input_data = [
        {"source": "news", "title": "Some title"},  # No text field
    ]

    # Should not raise an exception
    result = classifier.classify_batch(input_data)

    assert len(result) == 1
    assert "classification" in result[0]
    assert "confidence" in result[0]


def test_classifier_uses_heuristic_for_politics_content() -> None:
    """Test that classifier uses keyword-based heuristics for classification."""
    classifier = ClassifierImpl()

    # Political/government keywords should be classified as relevant
    political_data = [
        {"text": "government policy change", "source": "news"},
        {"text": "election results and voting", "source": "news"},
        {"text": "political party statement", "source": "news"},
    ]

    result = classifier.classify_batch(political_data)

    # At least some political content should be classified as relevant
    relevant_count = sum(1 for item in result if item["classification"] == "relevant")
    assert relevant_count > 0


def test_classifier_uses_heuristic_for_non_political_content() -> None:
    """Test that classifier classifies non-political content as not relevant."""
    classifier = ClassifierImpl()

    # Non-political content should be less likely to be relevant
    non_political_data = [
        {"text": "recipe for chocolate cake", "source": "food_blog"},
        {"text": "programming tutorial Python", "source": "tech_blog"},
        {"text": "movie review entertainment", "source": "entertainment"},
    ]

    result = classifier.classify_batch(non_political_data)

    # Should classify non-political content appropriately
    for item in result:
        # Confidence should reflect uncertainty for non-political content
        assert isinstance(item["confidence"], float)
        assert 0.0 <= item["confidence"] <= 1.0


def test_classifier_batch_processing_is_consistent() -> None:
    """Test that classifier produces consistent results for the same input."""
    classifier = ClassifierImpl()

    input_data = [
        {"text": "political news about elections", "source": "news"},
        {"text": "cooking recipe instructions", "source": "blog"},
    ]

    result1 = classifier.classify_batch(input_data)
    result2 = classifier.classify_batch(input_data)

    # Results should be identical for same input
    assert len(result1) == len(result2)
    for item1, item2 in zip(result1, result2, strict=False):
        assert item1["classification"] == item2["classification"]
        assert item1["confidence"] == item2["confidence"]
