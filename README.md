# Master Thesis
Includes materials for Master's Thesis Project

## Notes for presentation on first experiments (from meeting on 15.11.2024)
- Present all assumptions as bullet points
- Diagrams of experiment setups
- Present different threshold percentages in graphs
- Compare to real(istic) values
- Compare analytical vs numerical results
- Compute volume of cell wall
- Include references for everything
- Future experiments

## Links on permeability and diffusion
- https://bio.libretexts.org/Bookshelves/Biochemistry/Fundamentals_of_Biochemistry_(Jakubowski_and_Flatt)/01:_Unit_I-_Structure_and_Catalysis/11:_Biological_Membranes_and_Transport/11.02:_Diffusion_Across_a_Membrane_-_Passive_and_Facilitated_Diffusion
- https://legacy.nimbios.org/~gross/bioed/webmodules/diffusion.htm
- https://bio.libretexts.org/Bookshelves/Biochemistry/Fundamentals_of_Biochemistry_(Jakubowski_and_Flatt)/01%3A_Unit_I-_Structure_and_Catalysis/10%3A_Lipids/10.03%3A_Membrane_Bilayer_and_Monolayer_Assemblies_-_Structures_and_Dynamics
- https://bio.libretexts.org/Bookshelves/Cell_and_Molecular_Biology/Book:_Cells_-_Molecules_and_Mechanisms_(Wong)/04:_Membranes_-_Structure_Properties_and_Function/4.02:_Membrane_Permeability
- https://book.bionumbers.org/what-are-the-rates-of-membrane-transporters/
- https://book.bionumbers.org/what-is-the-permeability-of-the-cell-membrane/
- https://books.gw-project.org/flux-equations-for-gas-diffusion-in-porous-media/chapter/effective-molecular-diffusion-coefficient/
- https://perminc.com/resources/fundamentals-of-fluid-flow-in-porous-media/chapter-3-molecular-diffusion/diffusion-coefficient/effective-diffusion-coefficient/

## Questions for meeting with Han 01.11.2024
- How would physical signaling between spores work? Is there further evidence for this?
- Physical experiment - spores won't be washed?
- If the inhibitor is still present even after being washed off, does it mean it is continuously produced? Or just that it is present in very large quantities?
- How long would a laboratory experiment take?
- Inhibitor is released through membrane. Disregard role of permeability of membrane in diffusion?

## Notes from meeting with Han 25.10.2024
- Hypothesis: there is a diffusible signalling molecule between conidia.
- When conidia are closer signal is high. More spores -> more concentration of inhibitor.
- There is a threshold of inhibitor concentration needed for germination (e.g. $10\times$ of own concentration of spore).
- Can start with known diffusion coefficient of glucose in water.
- How fast is the inhibitor distributed homogeneously in the well plate?
- Tests in lab can be performed to confirm/reject the model.
- Improved model: add external source of inhibitor, find diffusion coefficient of unknown unhibitor (e.g. 1-octen-3-ol)?
- Spore - has a certain volume, can only fit a certain volume of inhibitor, produces limited quantity of inhibitor _in itself_.
- Start with 1 spore with inhibitor -> then 10 spores (checkerboard pattern?) -> then more complex distributions.
- Example: in mushrooms 1-octen-3-ol is too high for germination, only when the spores are dispersed can they germinate.
- Knowns:
  - dimensions of well;
  - volume of medium;
  - dimensions of spore;
- Setups:
  - threshold of $10\times$, $100\times$, $1000\times$ concentration of inhibitor in spore;
  - instant vs gradual release?
- Variables:
  - concentration of inhibitor in spore;
  - rate of release in time.
- cAMP can be looked at later;
- OK with working towards publication.

## Questions for Maryam 04.10.2024
- Which of the oCelloscope measurements were saved in .csv data?
- Approx. volume of data per experiment? Does it fit on a normal computer?
- Are the discarded overlapping objects not present in any of the .csv's?
- Is the data after the 15th hour still present in the .csv's?
- Was spore agglomeration observed in the experiments? Were agglomerations discarded due to overlap?

## Notes of meeting with Jaap 02.10.2024
- Better to focus on species that has been most extensively sequenced / researched, rather than based on its relevance.
- Handling data can take a lot of time (bottleneck) - be very careful with re-visiting microscopic imagery, can be a topic on its own;
- Narrow down to single species;
- Pathways in carbon sequencing can be a topic on its own;
- 3D data - cool and relevant, can be used, but can be computationally intense!
- Sigmoid curves come up all the time in biology;
- PDEs better than ABM;
- less asumptions in 3D modelling;
- reduce, reduce, reduce - MVP should be simple and explainable;
- discuss publication with Han.

## Alternative sources of data
- "Synchrotron radiation-based microcomputed tomography for three-dimensional growth analysis of Aspergillus niger pellets" (data available here: https://mediatum.ub.tum.de/1700656)

## oCelloscope measurements

- ScanArea - probably well ID?
- AcquireTimeLocal - date and time of capture;
- BoundingBox - [x0, y0, w, h]?;
- FocusedObjectID - ID of the identified object (may not be the same across frames);
- Area - area of object;
- Branch points - number of branchings;
- Circularity - how close is the shape to a circle;
- Contrast - difference between bright and dark pixels;
- Elongation - ?
- Granularity - ?
- ThinnedLength - length beyond circular region (so length of hypha);
- XPosition - centroid X coordinate;
- YPosition - centroid Y coordinate.

## Notes after lab visit - 20.09.24

### Data

- The .csv data retrieved from the oCelloscope is relatively lightweight and straightforward to interpret. It seems like it should be sufficient to obtain statistical analysis of the germination, which a model could be fitted to or validated with.
- The image data offers much more information that can be extracted if appropriate image processing techniques are applied. These, however, can constitute an entire project on their own, and the large volume of the data poses some infrastructural problems, since it is only stored locally.
- The data predominantly spans the germination stage, but there are also measurements from the initial hyphal growth stage (cell elongation, germ tube length etc.) which could be extrapolated to later stage developments. This means that it would make sense to model the stages from single spores up until the beginning of dense mycelium formation.
- Overlapping objects are usually cleaned from the data, but perhaps revisiting these could still be useful in determining hyphal root densities towards the end of the experiments. 

### Fungus

- Both A. niger and A. pullulans are fungi with great relevance for biotechnology. Since the former is a very well-studied species (including the experiments performed by Maryam at the lab) and less complex than the latter, it makes sense to start with that one and potentially see if some principles are transferable to A. pullulans.
- In modelling these fungal species, it might help to keep in mind what the most relevant properties for their industrial application are - e.g. the relationship between germination/mycelium growth (in space and time) and enzyme production in A. niger or the formation of melanin-rich chlamydospores in A. pullulans, among other cell types.

### Modelling

- The Pmax and tau parameters in the A. niger germination studies have been obtained by fitting a germination model to the data (the asymmetric model by Dantigny et al. (2011)). It is worth analysing the assumptions of this model and interpreting the underlying biological phenomena (e.g. in terms of mass-action events) in order to (1) deconstruct its dependencies and view them in the light of the experiments performed (the variation of nutrients and spore populations) and (2) find the general principles which can apply to modelling other fungi such as A. pullulans.
- I will use the rich body of information in Steinberg et al. (2017) on the cell biology of hyphal growth to sort out some of the most important and relevant cellular/molecular factors that can be represented mathematically.
- Another paper by Baltussen et al. (2020) focuses specifically on the molecular mechanisms of conidial germination in Aspergillus, so I will study that and its links to the aforementioned paper.
- I have been gathering papers on existing models of fungal growth but I still have to sort out the ones most relevant for our current case, see which features and techniques can be adopted.

## Notes for coordination meeting - 13.09.24

**Questions**
- Scope: 42 EC; Is there a relevant question for Han's research group which would benefit from dedicated research into Computational Modelling of mycelium?
- Focus: inoculation of substrate by spores/mycelium; relevance for biofabrication, microecology, ...
- Definition of objectives: modelling an observed phenomenon under assumptions of mathematical abstractions; consolidating knowledge; bridging micro- and macroscopic scales; making predictions on hypothetical scenarios without having to grow samples physically; looking for quantifiable complex system phenomena (pattern formations, density distributions, steady-states/equilibria or other types of dynamics);
- Mentioned examples: mutual inhibition and synergistic behavior between spores of different species; spore clustering under different adhesive conditions;
- Verification through data - what sort of arrangements can be made on data use; only focus on publically available data? NDA? What are typical requirements from the programme? 
- Availability for irl meeting/visit to lab in Utrecht.
