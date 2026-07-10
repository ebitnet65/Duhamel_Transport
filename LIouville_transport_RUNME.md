# Liouville-space transport calculation

Place these files in the same directory:

- `phase_matched_response.py`
- `Liouville_transport.py`
- `run_liouville_transport.py`

The current `Liouville_transport.py` imports the required helpers from
`phase_matched_response.py` directly. The older
`Liouville_transport_import_patch.py` file is retained only as historical
context.

Run with the defaults:

```bash
python run_liouville_transport.py
```

A denser lambda sweep:

```bash
python run_liouville_transport.py --n-lambda 101
```

Use the instantaneous steady state instead of the fixed reference state:

```bash
python run_liouville_transport.py --initial steady
```

Choose another output directory:

```bash
python run_liouville_transport.py --output-dir results_transport
```

The primary diagnostics are:

- `||T(0)-I||`: should be near numerical precision.
- `rank`: should equal the number of canonical pathways retained.
- `condition number`: large values indicate a nearly dependent pathway basis.
- `projection residuals`: quantify how much of each finite-lambda pathway lies
  outside the canonical lambda=0 pathway subspace.
