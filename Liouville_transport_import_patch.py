# Required import patch for Liouville_transport.py
#
# Add this immediately below the NumPy/SciPy/Matplotlib imports.

from phase_matched_response import (
    STANDARD_PATHWAYS,
    G_freq,
    initial_state,
    interaction_super,
    liouvillian,
    omega1_sign_for_sector,
    pathway_label,
)

def get_sector_pathways(sector):
    """Return the canonical two-level pathways for a phase-matched sector."""
    sector = sector.upper()
    if sector not in STANDARD_PATHWAYS:
        raise ValueError(
            f"Unknown sector {sector!r}; expected one of "
            f"{tuple(STANDARD_PATHWAYS)}"
        )
    return STANDARD_PATHWAYS[sector]
