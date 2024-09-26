# Master Thesis
Includes materials for Master's Thesis Project

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
