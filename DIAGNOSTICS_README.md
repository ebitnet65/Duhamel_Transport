# Basis and stage diagnostics

Keep these scripts in the same directory as:

- `phase_matched_response.py`
- `Liouville_transport.py`

Run the basis diagnostic:

```bash
python diagnose_reference_basis.py
```

Run the stage comparison:

```bash
python diagnose_transport_stages.py
```

Or run both:

```bash
bash run_all_diagnostics.sh
```

Key interpretation:

- A condition number near `1e17` means the nominal pathway basis is
  effectively singular in double precision.
- If RP and NRP are individually well conditioned but RP+NRP is not,
  compute separate sector transport matrices.
- If a sector becomes singular only after `J3` or `G3`, then distinct
  canonical pathways have collapsed into the same emitted coherence.
  The transport basis should then be defined earlier, preferably after
  `J2` or after `U2`.
