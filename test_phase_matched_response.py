#!/usr/bin/env python3
"""Basic validation of the extracted response module."""

import numpy as np

from phase_matched_response import (
    check_comm_vs_all_rwa,
    get_sector_pathways,
    liouvillian,
    mat,
    steady_state,
)


def main() -> None:
    for theta in (0.0, np.pi / 4, np.pi / 2):
        L = liouvillian(theta=theta)
        rho_ss = steady_state(L)
        residual = np.linalg.norm(L @ rho_ss)
        print(f"theta={theta:.6f}")
        print(mat(rho_ss))
        print(f"steady-state residual={residual:.6e}\n")

    mismatch, _, _ = check_comm_vs_all_rwa(
        theta=np.pi / 4,
        omega=1.0,
        initial="fixed_steady0",
        verbose=True,
    )

    print("\nPathway counts")
    for sector in ("RP", "NRP", "DQC", "ALL_RWA"):
        print(f"{sector}: {len(get_sector_pathways(sector))}")

    if mismatch > 1e-10:
        raise SystemExit(
            f"COMM_PLUS/ALL_RWA mismatch is too large: {mismatch:.6e}"
        )


if __name__ == "__main__":
    main()
