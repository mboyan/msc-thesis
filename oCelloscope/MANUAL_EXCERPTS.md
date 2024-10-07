# 2015 version
...

GKA analysis on data acquired with Acquire Type set to "Reduced data" or "Minimal data" is more sensitive to minor disturbances than analysis where Acquire Type has been set to "Full data". This may for negative controls (scan areas containing only medium) in some situations result in negative growth curves. This can safely be ignored. In any case, the image data may be used to verify that there has been no growth.

## 6.1.1 Background Corrected Absorption

The Background Corrected Absorption (BCA) algorithm is designed to detect microbial growth with high sensitivity even at very low or high cell concentrations.

Based on the first image, the BCA algorithm corrects background intensities to obtain images with an even light distribution before calculating a threshold pixel value which divide pixels into 'background pixels' and 'object pixels'. The BCA algorithm generates growth curves based on changes in 'object pixels'. In this way, BCA is able to determine microbial growth with high sensitivity as the effect of background intensities are reduced significantly compared to the TA algorithm.

### Limitation:

BCA may lead to inaccurate determination of growth curves when for example condensation obscures light transmission resulting in darker images and consequently in false 'object pixels'.

The BCA algorithm has two steps:

- Correction of illumination profile
- Calculation of absorption by summarising pixel histogram content above a threshold
