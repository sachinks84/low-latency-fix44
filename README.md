# Low-latency FIX 4.4 encoder

This directory contains two standalone C++20 encoders for a FIX 4.4
`NewOrderSingle` limit-buy message:

- `FixEncoderLowLatency.cpp` — optimized AVX2-oriented encoder.
- `FixEncoder_Normal.cpp` — zero-copy baseline using standard conversions and
  a scalar checksum.

Both programs create the same wire message, write directly into the caller's
packet buffer, and run a one-million-message benchmark. `|` in printed output
represents the FIX SOH delimiter (`0x01`).

The repository is named `low-latency-fix44` so a FIX decoder can be added
alongside the encoder later.

## Build and run

```bash
make
make run          # optimized encoder
make run-normal   # normal baseline
make compare      # run both
```

The optimized implementation requires an x86 CPU with AVX2. The normal
baseline does not use AVX2 and deliberately disables checksum loop
vectorization.

## Message layout

The encoder emits this FIX 4.4 limit-buy order shape:

```text
8=FIX.4.4|9=143|35=D|49=HFT_CLIENT|56=BROKER|34=1|
52=20260809-12:34:56.789|11=HFT-000001|21=1|55=AAPL|54=1|
60=20260809-12:34:56.789|38=100|40=2|44=1.2345|59=0|10=221|
```

The caller supplies the exact packet offset at which `8=FIX.4.4` begins. The
demo uses `alignas( 32 ) char buf[ 1024 ]`, so the message start is 32-byte
aligned.

`BodyLength` is constrained to the canonical three-digit range `100..999`.
That keeps `35=` at the fixed offset `messageStart + 16`:

```text
offset 0 : 8=FIX.4.4<SOH>  (10 bytes)
offset 10: 9=ddd<SOH>     ( 6 bytes)
offset 16: 35=D<SOH>...
```

The actual body is written once, its length is calculated from the pointer
range `[ bodyStart, bodyEnd )`, and the three `BodyLength` digits are then
written into the header. No temporary body buffer, `memmove`, or final-message
copy is used.

## Optimizations in `FixEncoderLowLatency.cpp`

- Direct buffer writers return the next output pointer after each `tag=value`
  field.
- Integer conversion uses a digit-count estimate and a `00..99` table. It
  writes decimal characters right-to-left in two-digit groups without
  `std::to_string` or allocation.
- Unsigned `uint32_t` fields, including MsgSeqNum and OrderQty, use a direct
  unsigned writer; they are never narrowed through `int`.
- The incoming limit price remains a `double`, but the venue price scale is
  fixed at four decimal places. The specialized formatter writes `44=` directly
  to the packet buffer and handles decimal rounding carry.
- All body tags are known two-digit constants. Template-specialized writers
  emit prefixes such as `35=`, `49=`, and `44=` directly instead of converting
  tag integers on every order.
- The `8=`/`9=` header is written directly at the caller-specified message
  start. It preserves a 32-byte-aligned FIX start without padding the
  `BodyLength` field with leading zeroes.
- Checksum uses AVX2 byte-sum instructions. For an unaligned supplied start,
  it sums the scalar prefix up to the next 32-byte boundary, uses aligned AVX2
  loads for the body, then finishes the scalar tail.
- Checksum tag formatting writes the final two digits with the `00..99` table,
  avoiding separate tens and ones arithmetic.
- Global scalar `new` and `new[]` counters verify that the timed benchmark has
  no C++ dynamic allocations.

## Normal baseline

`FixEncoder_Normal.cpp` is intentionally simpler while retaining zero-copy
output:

- `std::to_chars` writes integer and fixed-point values directly into the final
  buffer.
- The checksum is a conventional scalar byte-at-a-time loop.
- It uses the same message, fixed price scale, mutable input object, allocation
  check, and benchmark workload as the optimized version.

This makes `make compare` useful for observing the effect of the custom number
writers and AVX2 checksum without comparing against an allocating encoder.

## Benchmark workload

Each benchmark performs 1,000,000 encodes into the same aligned packet buffer.
The same `NewOrderSingleInput` is reused: `price` starts at `1.2345` and adds
`1.0` per iteration; `seq_num` starts at `1` and increments once per message.
The compiler-only `doNotOptimize` barrier prevents removal of the generated
wire stores without adding a hardware memory load.

For reproducible timing methodology, CPU pinning, `perf stat` commands, and
controlled median results, see [PERF_BENCHMARK.txt](PERF_BENCHMARK.txt).

## Controlled benchmark improvement

Nine alternating one-million-message runs were pinned to an unused CPU. The
median result was:

| Encoder | Total time / 1M encodes | Average time | Throughput |
|---|---:|---:|---:|
| Optimized | 138.321 ms | 138.321 ns/msg | 7.230M msg/s |
| Normal | 281.351 ms | 281.351 ns/msg | 3.554M msg/s |

The optimized encoder reduces the one-million-message run by **143.030 ms**
and the average message time by **143.030 ns**: a **50.8% reduction** in time,
or a **2.03x speedup**. Both benchmark variants were allocation-free and
produced identical FIX wire messages.

## Benchmark caveat

Nanosecond timings depend on CPU frequency, host scheduling, interrupts,
cache pressure, and whether the benchmark core is isolated. Treat the supplied
results as a comparison method, not a production latency guarantee. For HFT
deployment measurements, isolate a physical core, control frequency scaling,
and move NIC interrupt work away from the benchmark core.
