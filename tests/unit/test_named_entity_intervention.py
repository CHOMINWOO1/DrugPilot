"""Generic lexical tests; positive syntax does not imply positive biology."""
import pytest

from drugpilot.agents.ma1 import _direct_target_intervention_pattern as consumer
from drugpilot.tools.public_biology import (
    _direct_target_intervention_pattern as producer, _observed_direct_intervention_present,
    _publication_content_from_sections, _primary_direction_content_rejection_codes,
)


@pytest.mark.parametrize("text",[
    "inhibition of the oncogenic kinase BCR::ABL1",
    "inhibited the fusion protein BCR::ABL1",
    "suppression of the tyrosine kinase BCR::ABL1",
    "inhibition of BCR::ABL1",
])
def test_closed_entity_descriptor_is_recognized_by_both_layers(text):
    assert producer("BCR::ABL1","inhibit").search(text)
    assert consumer("BCR::ABL1","inhibit").search(text)
    assert _observed_direct_intervention_present(text,"ABL1","inhibit")


@pytest.mark.parametrize("text",[
    "inhibition of the pathway downstream of BCR::ABL1",
    "inhibition of the oncogenic kinase BCR::ABL1 signaling",
    "inhibition of the oncogenic kinase BCR::ABL1 pathway",
    "inhibition of another kinase upstream of BCR::ABL1",
    "inhibition of BCR::ABL1-mediated signaling",
])
def test_descriptor_does_not_bridge_arbitrary_text_or_pathways(text):
    assert not producer("BCR::ABL1","inhibit").search(text)
    assert not consumer("BCR::ABL1","inhibit").search(text)
    assert not _observed_direct_intervention_present(text,"ABL1","inhibit")


def test_action_recognition_does_not_supply_missing_favorable_endpoint():
    text="We demonstrate that inhibition of the oncogenic kinase ETV6::FLT3 in acute myeloid leukemia cells induces a stress response."
    content=_publication_content_from_sections([("ABSTRACT",text)],target_symbol="FLT3",disease="acute myeloid leukemia",intervention_direction="inhibit")
    reasons=_primary_direction_content_rejection_codes(content,title="Synthetic primary report",publication_types=["Journal Article"],
        target_symbol="FLT3",disease="acute myeloid leukemia",intervention_direction="inhibit")
    assert "direct_target_intervention_missing" not in reasons
    assert reasons and "favorable_endpoint_polarity_missing" in reasons
