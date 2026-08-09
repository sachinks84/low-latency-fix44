# Low-latency FIX 4.4 encoder

A C++20, allocation-free FIX 4.4 `NewOrderSingle` limit-buy encoder.

It serializes a complete FIX message directly into the caller-provided packet
buffer using `tag=value<SOH>` wire format. The current message includes the FIX
header, session fields, limit-buy order fields, `BodyLength`, and `CheckSum`.

The project provides two implementations of the same message encoder:

| Variant | Number conversion | Checksum | Purpose |
|---|---|---|---|
| `FixEncoderLowLatency.cpp` | Custom direct writers | AVX2 | Low-latency implementation |
| `FixEncoder_Normal.cpp` | `std::to_chars` | Scalar loop | Zero-copy comparison baseline |

Both implementations write directly into the final packet buffer and perform no
dynamic allocation during the benchmark.

## Why the low-latency encoder is faster

The low-latency variant removes work from the hot path:

- Fixed FIX tags such as `35=`, `49=`, and `44=` are emitted as compile-time
  character stores rather than converted at runtime.
- Integer fields use direct decimal writers and a `00..99` digit table.
- The fixed-scale limit price is written directly to the output buffer.
- `BodyLength` is backfilled without a temporary message buffer or `memmove`.
- The checksum uses AVX2 byte summation.
- No intermediate strings or dynamic allocations are created.

## Benchmark result

The benchmark encodes one million complete FIX limit-buy messages. It updates
the limit price and sequence number for every message.

| Variant | Average latency | Throughput |
|---|---:|---:|
| Low-latency | **59.853 ns/message** | **16.708M messages/second** |
| Normal (`std::to_chars` + scalar checksum) | ~220 ns/message | ~4.55M messages/second |

The low-latency encoder is approximately **3.68× faster**, saving about
**160 ns per message** and reducing latency by approximately **72.8%**.

## Hardware counter comparison (`perf stat`)

Per-message hardware counters from `perf stat` on an Intel Xeon Gold 5118:

| Counter | Low-latency | Normal | Reduction |
|---|---:|---:|---:|
| instructions | 548 | 1,944 | **3.55× fewer** |
| cycles | 295 | 599 | **2.03× fewer** |
| branches | 77 | 339 | **4.4× fewer** |
| branch-misses | 1.0 | 1.0 | same |

The custom direct writers and compile-time tag prefixes eliminate 3.55× the
instructions per message compared to `std::to_chars`. The branch count drops
4.4× because the lookup-table approach avoids the internal loops and
conditionals of general-purpose number formatting.

## Build and run

```bash
make
make run          # low-latency encoder
make run-normal   # normal baseline
make compare      # run both
```

`|` in the printed output represents the FIX SOH delimiter (`0x01`).

## Performance methodology

The detailed benchmark methodology, CPU considerations, `taskset` usage, and
`perf stat` commands are documented in
[PERF_BENCHMARK.txt](PERF_BENCHMARK.txt).

Use the same CPU affinity, frequency policy, and workload when comparing
results. Nanosecond timings vary with host contention and CPU scheduling.
