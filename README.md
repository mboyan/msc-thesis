# Notes

## Extension of Dantigny model
Since some initial concentrations may already overlap with threshold distributions, it is possible that the germination curve starts from a nonzero value.
This implies that the Dantigny equation needs to be extended with a baseline time shift $\delta$, yielding a total of 4 Dantigny summaries:

$$
p=p_{\textrm{max}}\left(1-\frac{1}{1+\left(\frac{t+\delta}{\tau}\right)^\nu}\right).
$$

## Important for BMA
- Check if experimental _triplets_, sample sizes etc. (see Ijadpanahsaravi et al. 2023) should be considered in BMA process.

## Update of Assumptions
- Closed-form solution models - solved using GH, else solved by MC;
- Half-saturation constant of inducer is the same in affecting the inhibition threshold and in producing a germination signal - this is because **the inhibition threshold is considered shifted as a by-product of the inducing trigger** (in the case of 2-factor germination);
- Cell wall porosity - insignificant variance (?);
- Induction threshold may be shifted differently depending on carbon source (because inducing signal pathways are inhibited in different proportions), but also inhibition threshold may be shifted differently depending on carbon source (because inducing signal pathways take effect in different proportions) - this means both the multiplication factors and the effective half-saturation constants are inducer-specific;
- Michaelis-Menten kinetics for inducer (because receptor proteins are known to play a role) but simple linear relationship for inhibitor (because effect less known); HOWEVER: the (indirect) inhibitory effect on the induction threshold may be non-linear - therefore, Michaelis-Menten kinetics is used;
- The inducing effect may occur via different parallel signalling pathways (RasA, GPRC to cAMP-PKA pathway...). This is clumped into a single Michaelis-Menten-like relationship with effective half-saturation constants, but the constituent signalling pathways may differ for varying types of carbon source molecules. Also: the constituent components of the inductive effect may contribute to permeability differently (in the case of inducer-modulated cell wall permeability). These assumptions inform which parameters are inducer-specific.
- While there may also be different inhibitor types within a single spore, they can be clumped into a single inhibitor effect (we assume they do not interact) that is specific for a spore colony - therefore, purely inhibitor-dependent parameters do not vary under changing carbon source.

## Deliverables
- Sensitivity analysis - global vs. local, algorithm, Julia package?
- Consider experimental data variances in model fitting
- Mid-resolution (agent-based) lattice simulations vs. selected models - to verify heterogeneity assumptions / impact of spatial distribution
- Use Damköhler number, Biot number or Thiele modulus to support or further analyse adsorption-diffusion relation
- Explore more model variants with 2-way signal influence (revise inducer-dependent inhibition and explore feedback loops)
- Revise notation for publication

## Week 37 highlights
- LaTeX set up, but workaround needs to be used to insert chapter text into Han's Word template of preference (using `python-docx`);
- At first glance, no relevant information on Damköhler/Biot/Thiele analysis in the context of 1-octen-3-ol/Aspergillus; but rough calculations could support existing results;
- Look into Bayesian model averaging;
- Models should be considered _law-driven_.

# Modelling Diffusive Signals for the Germination of _Aspergillus_ Conidia

This repository contains the code, data and documentation used in my Master's Thesis (MSc Computational Science, UvA/VU, Amsterdam, 2024/2025).

## Abstract

The germination rate of _Aspergillus_ conidia is reportedly influenced by the inducing carbon source in the medium and by an auto-inhibitor produced by the spores. This thesis assesses the plausibility of diffusion-driven mechanisms in timing the action of these signals until germination is enabled. To this end, computational models of spores releasing inhibitor molecules are constructed on multiple scales, first simulating the depletion of inhibitor from a single spore, then exploring the effect of increasing spore culture densities, and eventually inspecting the diffusive outflow in a dense spore cluster. This leads to several observations:
- the commonly considered inhibitor 1-octen-3-ol would be depleted too fast, unless a strong cell wall adsorption or continuous synthesis slow down its decrease;
- increasing spore densities flatten the permeation-driving gradient through an ambient inhibitor saturation;
- dense spore packings do not lead to substantial inhibitor retention, unless their contact area is large.

Finally, germination probability models incorporating induction and inhibition are proposed, representing heterogeneities in the spores through random variables. Parameter estimation through global and local optimisation highlights a promising model that fits experimental data under biologically sensible parameters. In this model, an inhibitor falls below a critical value, and an inhibitor-dependent inducing signal rises above an inhibitor-dependent threshold to trigger germination. In an attempt to explain data with both endogenously and exogenously driven 1-octen-3-ol inhibition, no appropriate parameter combination is found, leading to the supposition that in vivo inhibition is more complex than merely saturating the medium with the compound.

## Code

The experiments for this thesis are programmed in Julia and are presented in the Jupyter notebooks in the `Notebooks` folder. The code is structured in designated modules, which can be found in the `src` folder. To make sure you have all dependencies installed, you can run the following code in the Julia REPL:

```
using Pkg
Pkg.add(url="https://github.com/mboyan/msc-thesis/")
```

This is my first serious undertaking with Julia, and there are possibly quite a few redundancies or sub-optimal pieces of code. If you have any useful suggestions on how to improve things, forks and pull requests are very welcome. I will likely not maintain this repository regularly, but I appreciate opportunities to learn new things.
