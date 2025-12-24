"""
Integration tests for Mistral V3/Tekken tokenizer.

Mistral V3 (Tekken) is NOT YET IMPLEMENTED.
This file only tests that appropriate errors are raised.

V3/Tekken characteristics (for future implementation):
- Vocab size: ~131,126 (131,072 base + 54 agent tokens)
- Uses Tiktoken-style encoding (NOT SentencePiece)
- Different from V1/V2 in vocabulary structure
- Used by: Mistral NeMo, Mistral Large 2, Pixtral
"""

import pytest
from splintr import Tokenizer


class TestMistralV3NotImplemented:
    """Test that Mistral V3/Tekken raises appropriate errors."""

    def test_mistral_v3_not_implemented(self):
        """mistral_v3 name should raise 'not yet implemented' error."""
        with pytest.raises(ValueError, match="not yet implemented"):
            Tokenizer.from_pretrained("mistral_v3")

    def test_hyphenated_v3_names_unknown(self):
        """Hyphenated V3 names should be unknown (not registered)."""
        unknown_names = [
            "mistral-tekken",
            "mistral-nemo",
            "mistral-large",
        ]
        for name in unknown_names:
            with pytest.raises(ValueError, match="Unknown pretrained model"):
                Tokenizer.from_pretrained(name)
