This repository will contain the source code to reproduce the results of the paper

"Content representation for Neural Style Transfer Algorithms based on Structural Similarity"

written by [Philip Meier](https://www.th-owl.de/init/en/das-init/team/c/meier-5.html) and [Volker Lohweg](https://www.th-owl.de/init/en/das-init/team/c/lohweg-1.html). It will be available after the paper is presented at the [29. Workshop "Computational Intelligence"](http://www.rst.e-technik.tu-dortmund.de/cms/de/Veranstaltungen/GMA-Fachausschuss/index.html) on the 28th and 29th of November 2019 in Dortmund, Germany.

# Accepted Abstract

Within the field of non-photorealistic rendering (NPR) the term style transfer describes a process, which applies an abstract style to an image without changing the underlying content. The emergence of neural style transfer (NST) techniques, which were pioneered by Gatys, Ecker, and Bethge in 2016 [GEB16], marks a paradigm shift within this field. While traditional NPR methods operate within the pixel space [EF01], NST algorithms utilise the feature space of convolutional neural networks (CNNs) trained on object classification tasks. This enables a general style transfer from a single example image of the intended style. The quality of the resulting image is sometimes high enough to even fool art critics [San+18].

NST techniques treat the style of an image as texture. Thus, its representation involves various forms of global [GEB16; RWB17] and local statistics [LW16]. Within the original formulation, the content of an image is directly represented by the encodings from a deep layer of the CNN [GEB16]. These encodings are subsequently compared with their mean squared error (MSE). To the best knowledge of the authors currently no publication deals with alternative representations of the content. This contribution will change this by introducing a content representation based on the structural similarity (SSIM) index. The SSIM index was introduced by Wang et al. as a measure for image quality [Wan+04]. It was developed in order to compare two images with an objective measure that is aligned with the human perception opposed to conventional methods such as the MSE or the peak signal-to-noise-ratio. The SSIM index is incorporated as content representation into NST algorithms by utilising it as comparison between encodings of a CNN.

The proposed approach will be evaluated in two stages. An objective comparison between different NST algorithms is not possible within the current state of the art, since the quality of the stylisation is highly subjective. Thus, this contribution will focus on content reconstruction in a first step. Images reconstructed by different algorithms can be objectively compared to the original, for example by the SSIM index or the number of matching descriptors of the speeded up robust features (SURF) algorithm [BTv06]. In the second step the proposed content representation is utilised within an NST algorithm and qualitatively compared to the original formulation.

## References

Symbol | Reference
--- | ---
BTv06 | Bay, Herbert; Tuytelaars, Tinne; van Gool, Luc: ‘SURF: Speeded Up Robust Features’. In: Proceedings of the 9th European Conference on Computer Vision (ECCV). 2006.
EF01 | Efros, Alexei A.; Freeman, William T.: ‘Image Quilting for Texture Synthesis and Transfer’. In: Proceedings of the 28th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH). 2001. DOI: [10.1145/383259.383296](https://dl.acm.org/citation.cfm?doid=383259.383296).
GEB16 | Gatys, Leon A.; Ecker, Alexander S.; Bethge, Matthias: ‘Image Style Transfer Using Convolutional Neural Networks’. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016. DOI: [10.1109/CVPR.2016.265](https://ieeexplore.ieee.org/document/7780634).
LW16 | Li, Chuan; Wand, Michael: ‘Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis’. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016. DOI: [10.1109/CVPR.2016.272](https://ieeexplore.ieee.org/document/7780641).
RWB17 | Risser, Eric; Wilmot, Pierre; Barnes, Connelly: [‘Stable and Controllable Neural Texture Synthesis and Style Transfer Using Histogram Losses’](https://arxiv.org/abs/1701.08893). In: Computing Research Repository (CoRR) 1701 (2017).
San+18 | Sanakoyeu, Artsiom et al.: [‘A Style-Aware Content Loss for Real-time HD Style Transfer’](https://arxiv.org/abs/1807.10201). In: Computing Research Repository (CoRR) 1807 (2018).
Wan+04 | Wang, Zhou et al.: ‘Image quality assessment: from error visibility to structural similarity’. In: IEEE Transactions on Image Processing 13.4 (2004). DOI: [10.1109/TIP.2003.819861](https://ieeexplore.ieee.org/document/1284395).
