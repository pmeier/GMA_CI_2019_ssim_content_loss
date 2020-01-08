from torch import optim
from torchimagefilter import GaussFilter
from pystiche.encoding import vgg19_encoder
from pystiche.nst import (
    DirectEncodingComparisonOperator,
    DiagnosisOperator,
    MultiOperatorEncoder,
    CaffePreprocessingImageOptimizer,
    ImageOptimizerPyramid,
)
from pystiche_replication import GatysEtAl2017StyleLoss
from operators import SSIMEncodingComparisonOperator, SSIMScoreDiagnosisOperator

__all__ = [
    "MeierLohweg2019NCR",
    "MeierLohweg2019NST",
    "MeierLohweg2019NCRPyramid",
    "MeierLohweg2019NSTPyramid",
]


def get_encoder():
    return MultiOperatorEncoder(vgg19_encoder(weights="caffe", preprocessing=False))


def get_content_operator(
    encoder, ssim_loss=True, image_filter=None, ssim_component_weight_ratio=9.0
):
    layers = ("relu_4_2",)

    if ssim_loss:
        name = "Content loss (SSIM)"
        score_weight = 1e3
        if image_filter is None:
            image_filter = GaussFilter(radius=1, padding_mode="replicate")
        return SSIMEncodingComparisonOperator(
            encoder,
            layers,
            name=name,
            score_weight=score_weight,
            component_weight_ratio=ssim_component_weight_ratio,
            image_filter=image_filter,
        )
    else:
        name = "Content loss (SE)"
        score_weight = 1e-3
        return DirectEncodingComparisonOperator(
            encoder, layers, name=name, score_weight=score_weight
        )


def get_style_operator(encoder):
    operator = GatysEtAl2017StyleLoss(encoder)
    operator.score_weight = 1e0
    operator.name = "Style loss"
    return operator


def optimizer_getter(input_image):
    return optim.LBFGS([input_image], lr=1.0, max_iter=1)


class MeierLohweg2019ImageOptimizerBase(CaffePreprocessingImageOptimizer):
    def __init__(self, *operators, diagnose_ssim_score=False):
        if diagnose_ssim_score:
            operator = SSIMScoreDiagnosisOperator(name="SSIM score")
            operators = list(operators) + [operator]

        super().__init__(*operators, optimizer_getter=optimizer_getter)


class MeierLohweg2019NCR(MeierLohweg2019ImageOptimizerBase):
    def __init__(
        self,
        ssim_loss=True,
        image_filter=None,
        ssim_component_weight_ratio=9.0,
        **kwargs
    ):
        encoder = get_encoder()
        content_operator = get_content_operator(
            encoder, ssim_loss, image_filter, ssim_component_weight_ratio
        )
        super().__init__(content_operator, **kwargs)
        self.content_operator = content_operator


class MeierLohweg2019NST(MeierLohweg2019ImageOptimizerBase):
    def __init__(
        self,
        ssim_loss=True,
        image_filter=None,
        ssim_component_weight_ratio=9.0,
        **kwargs
    ):
        encoder = get_encoder()
        content_operator = get_content_operator(
            encoder, ssim_loss, image_filter, ssim_component_weight_ratio
        )
        style_operator = get_style_operator(encoder)
        super().__init__(content_operator, style_operator, **kwargs)
        self.content_operator = content_operator
        self.style_operator = style_operator


class MeierLohweg2019ImageOptimizerPyramidBase(ImageOptimizerPyramid):
    def build_levels(self, level_steps=None):
        level_edge_sizes = (500, 1024)
        if level_steps is None:
            level_steps = (500, 200)
        edges = ("short", "long")
        super().build_levels(level_edge_sizes, level_steps, edges=edges)

    def __call__(self, *args, **kwargs):
        target_image = self.image_optimizer.content_operator.target_image
        if target_image is not None:
            for operator in self.image_optimizer.operators(DiagnosisOperator):
                operator.set_target(target_image)
        return super().__call__(*args, **kwargs)


class MeierLohweg2019NCRPyramid(MeierLohweg2019ImageOptimizerPyramidBase):
    def __init__(self, *args, **kwargs):
        super().__init__(MeierLohweg2019NCR(*args, **kwargs))
        self.ncr = self.image_optimizer


class MeierLohweg2019NSTPyramid(MeierLohweg2019ImageOptimizerPyramidBase):
    def __init__(self, *args, **kwargs):
        super().__init__(MeierLohweg2019NST(*args, **kwargs))
        self.nst = self.image_optimizer
