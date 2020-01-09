This repository contains the source code to reproduce the results of the paper

"Content representation for Neural Style Transfer Algorithms based on Structural Similarity"

written by [Philip Meier](https://www.th-owl.de/init/en/das-init/team/c/meier-5.html) and [Volker Lohweg](https://www.th-owl.de/init/en/das-init/team/c/lohweg-1.html). It was  presented at the [29. Workshop "Computational Intelligence"](http://www.rst.e-technik.tu-dortmund.de/cms/de/Veranstaltungen/GMA-Fachausschuss/index.html) on the 28th and 29th of November 2019 in Dortmund, Germany.

If you use this work within a scientific publication, please cite it as

```
@InProceedings{ML2019,
  author    = {Meier, Philip and Lohweg, Volker},
  title     = {Content Representation for Neural Style Transfer Algorithms based on Structural Similarity},
  booktitle = {Proceedings of the 28\textsuperscript{th} Workshop Computational Intelligence},
  year      = {2019},
  url       = {https://github.com/pmeier/GMA_CI_2019_ssim_content_loss},
}
```

The paper is part the conference proceedings, which are [openly accessible](https://dx.doi.org/10.5445/KSP/1000098736).

# Installation

Clone this repository

`git clone https://github.com/pmeier/GMA_CI_2019_ssim_content_loss`

and install the required packages

```
cd GMA_CI_2019_ssim_content_loss
pip install -r requirements
```

If you experience problems while installing `torch` or `torchvision`, please follow the [official installation instructions](https://pytorch.org/get-started/locally/) for your setup.

# Replication

All results are contained in the `results` folder. If you want to replicate the results yourself you need to

1. download the source images by running `images.py`,
2. perform the experiments by running `experiments.py`, and
3. finally run `process.py` to process the raw experiment results.