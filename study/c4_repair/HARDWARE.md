# Hardware identification

A permitted host read on 2026-10-02 returned **Apple M5** from
`sysctl -n machdep.cpu.brand_string`. The running experiment records arm64,
ten logical CPUs, macOS 27.0, Python 3.12.13, OR-Tools 9.15.6755 and PyTorch 2.13.0.
The separate measurement is retained in `HARDWARE.json`.

The ordinary sandboxed runner cannot query the model string directly and records
`verified_cpu_model: false`. Its already-sealed v2 metadata is preserved unchanged;
the separate measurement identifies that same host without modifying the experiment.

This measurement identifies the chip, not core affinity, fixed clock frequency,
thermal behavior or exclusion of unrelated desktop activity. Controlled evaluation
must still enforce one experiment process/one solver worker and balance method order.
Resolve how this permitted hardware measurement enters the primary contract before
freezing it; the current freeze deliberately rejects an unidentified CPU model.
