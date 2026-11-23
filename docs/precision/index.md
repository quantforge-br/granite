# Precision

Each model has one page for what it computes in and what each rounding costs. The method is shared.

- [Hy4-flash](hy4-flash.md): weights, stages, the attention caches, drafting, quality against its BF16 reference and the lossy
  routing flags.
- [Inkling 2 Mini](inkling2.md): weights, stages, the attention caches and quality against its reference (the release with
  its experts dequantized to BF16, not a true BF16 model).
- [How quality is measured](quality.md): the gate, the BF16 references, the lossy routing flags, greedy identity and reproduction
  commands.
