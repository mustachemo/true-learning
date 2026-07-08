# true-learning

Learning algorithms the way a kid learns anything: by asking **"why?"** until there's nothing left to ask.

Not tutorials. Not "here's the API, go use it." Every topic here rebuilds an algorithm from the ground up — the math that motivates each piece, the code that implements it, the intermediate outputs that prove (or disprove) it, and crucially, **the versions that don't work and why they don't**. An algorithm only really makes sense once you've seen the simpler thing it's patching.

## Why this exists

Most explanations show you the final algorithm and ask you to trust it. That's memorization, not understanding. Real understanding comes from the failure chain:

> "Why do we do X?" → "Because Y failed." → "Why did Y fail?" → "Because Z was too naive." → "Why was Z naive?" → ...

Each topic in this repo is written as that chain, made explicit and reproducible:

```
naive approach          → fails in a specific, visible way
  + fix 1                → fixes that, introduces a new problem
  + fix 2                → fixes that
  ...                    → = the algorithm everyone actually uses
```

By the end, the "final" algorithm isn't a black box — it's just the sum of every problem you watched it solve, in order.

## How each topic is structured

Every topic lives in its own folder and generally includes:

- **A written walkthrough** (`*_from_first_principles.md` / `.html`) — math derivations, code snippets, and the generated figures inline, following the failure → fix → failure → fix narrative.
- **A single script** that regenerates every figure and every intermediate output from scratch, so nothing is hand-wavy or unreproducible.
- **A `figs/` folder** of the actual intermediate outputs (histograms, images, plots, LUTs, etc.) referenced by the walkthrough.
- **A validation step against a trusted reference implementation** (e.g. matching a standard library function bit-for-bit), so "this is what really happens" isn't just asserted — it's proven.

Rules of thumb followed throughout:

1. **Nothing is introduced without a reason.** If a technique appears, it's because the previous section just showed you why it's needed.
2. **Broken/naive versions are kept, not deleted.** Seeing *why* something fails is often more instructive than seeing the fix.
3. **Code and math stay next to each other.** Every equation has a corresponding snippet; every snippet is small enough to read in one glance.
4. **Claims are checked, not asserted.** Where possible, from-scratch implementations are validated against a well-known library (exact or near-exact match).

## Topics

| Topic | What it covers |
|---|---|
| [`CLAHE/`](CLAHE/CLAHE_from_first_principles.md) | Contrast Limited Adaptive Histogram Equalization, built up from: histogram/CDF → global Histogram Equalization (and why one global mapping fails) → tiling into local regions (Adaptive HE, and why *that* fails from noise + block seams) → contrast limiting (the "CL") → bilinear interpolation of neighboring tile mappings → final from-scratch implementation validated bit-exact against OpenCV's `cv2.createCLAHE`. |

More topics will be added over time, following the same structure.

## Running the code

Each topic's script is self-contained. For example:

```bash
cd CLAHE
python clahe_walkthrough.py
```

This regenerates every figure in that topic's `figs/` folder from scratch — nothing in a walkthrough is a pre-baked image you have to take on faith.
