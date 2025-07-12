# Modelling Diffusive Signals for the Germination of _Aspergillus_ Conidia

This repository contains the code, data and documentation used in my Master's Thesis (MSc Computational Science, UvA/VU, Amsterdam, 2024/2025).

## Abstract

The germination rate of _Aspergillus_ conidia is reportedly influenced by the inducing carbon source in the medium and by an auto-inhibitor produced by the spores. This thesis assesses the plausibility of diffusion-driven mechanisms in timing the action of these signals until germination is enabled. To this end, computational models of spores releasing inhibitor molecules are constructed on multiple scales, first simulating the depletion of inhibitor from a single spore, then exploring the effect of increasing spore culture densities, and eventually inspecting the diffusive outflow in a dense spore cluster. This leads to several observations:
- the commonly considered inhibitor 1-octen-3-ol would be depleted too fast, unless a strong cell wall adsorption or continuous synthesis slow down its decrease;
- increasing spore densities flatten the permeation-driving gradient through an ambient inhibitor saturation;
- dense spore packings do not lead to substantial inhibitor retention, unless their contact area is large.

Finally, germination probability models incorporating induction and inhibition are proposed, representing heterogeneities in the spores through random variables. Parameter estimation through global and local optimisation highlights a promising model that fits experimental data under biologically sensible parameters. In this model, an inhibitor falls below a critical value, and an inhibitor-dependent inducing signal rises above an inhibitor-dependent threshold to trigger germination. In an attempt to explain data with both endogenously and exogenously driven 1-octen-3-ol inhibition, no appropriate parameter combination is found, leading to the supposition that in vivo inhibition is more complex than merely saturating the medium with the compound.

## Code

The experiments for this thesis are programmed in Julia and are presented in the Jupyter notebooks in the `Notebooks` folder. To make sure you have all dependencies installed, you can run the following code in the Julia REPL:

```
using Pkg
Pkg.add(url="https://github.com/mboyan/msc-thesis/")
```

This is my first serious undertaking with Julia, and there are possibly quite a few redundancies or sub-optimal pieces of code. If you have any useful suggestions on how to improve things, forks and pull requests are very welcome. I will likely not maintain this repository regularly, but I appreciate opportunities to learn new things.
