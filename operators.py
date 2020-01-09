import torch
from torchssim import (
    get_default_ssim_image_filter,
    calculate_ssim_repr as _calculate_ssim_repr,
    calculate_non_structural,
    calculate_structural,
    calculate_simplified_ssim,
)
import pystiche
from pystiche.image.transforms.functional import rgb_to_grayscale
from pystiche.nst import (
    EncodingComparisonOperator,
    PixelComparisonOperator,
    DiagnosisOperator,
)

__all__ = ["SSIMEncodingComparisonOperator", "SSIMScoreDiagnosisOperator"]

SSIMReprenstation = pystiche.namedtuple(
    "ssim_reprensentation", ("raw", "mean", "mean_sq", "var")
)

SimplifiedSSIMContext = pystiche.namedtuple(
    "simplified_ssim_context", ("non_structural_eps", "structural_eps")
)


def calculate_ssim_repr(image, image_filter):
    ssim_repr = _calculate_ssim_repr(image, image_filter)
    return SSIMReprenstation(*ssim_repr)


def calculate_dynamic_range(x):
    dim = 2
    x = torch.abs(torch.flatten(x, dim))
    dynamic_range = torch.max(x, dim).values
    return dynamic_range.view(*dynamic_range.size(), 1, 1)


def calculate_ssim_eps(const, dynamic_range, eps=1e-8):
    return torch.clamp((const * dynamic_range) ** 2.0, eps)


def calculate_simplified_ssim_ctx(x, non_structural_const=1e-2, structural_const=3e-2):
    dynamic_range = calculate_dynamic_range(x)
    non_structural_eps = calculate_ssim_eps(non_structural_const, dynamic_range)
    structural_eps = calculate_ssim_eps(structural_const, dynamic_range)
    return SimplifiedSSIMContext(non_structural_eps, structural_eps)


class SSIMEncodingComparisonOperator(EncodingComparisonOperator):
    def __init__(
        self,
        encoder,
        layers,
        name="SSIM loss",
        component_weight_ratio=1.0,
        image_filter=None,
        non_structural_const=1e-2,
        structural_const=3e-2,
        **kwargs
    ):
        super().__init__(encoder, layers, name, **kwargs)
        self.component_weight_ratio = component_weight_ratio
        self.non_structural_weight = 1.0 / (1.0 + component_weight_ratio)
        self.structural_weight = 1.0 - self.non_structural_weight
        if image_filter is None:
            image_filter = get_default_ssim_image_filter()
        self.image_filter = image_filter
        self.non_structural_const = non_structural_const
        self.structural_const = structural_const

    def _enc_to_repr(self, enc):
        return calculate_ssim_repr(enc, self.image_filter)

    def _input_enc_to_repr(self, enc, ctx):
        return self._enc_to_repr(enc)

    def _target_enc_to_repr(self, enc):
        repr = self._enc_to_repr(enc)
        ctx = calculate_simplified_ssim_ctx(
            enc, non_structural_const=self.non_structural_const
        )
        return repr, ctx

    def _calculate_score(self, input_repr, target_repr, ctx):
        input_mean_sq, target_mean_sq = input_repr.mean_sq, target_repr.mean_sq
        input_var, target_var = input_repr.var, target_repr.var
        mean_prod = input_repr.mean * target_repr.mean
        covar = self.image_filter(input_repr.raw * target_repr.raw) - mean_prod

        non_structural = calculate_non_structural(
            input_mean_sq, target_mean_sq, mean_prod, ctx.non_structural_eps
        )
        structural = calculate_structural(
            input_var, target_var, covar, ctx.structural_eps
        )

        non_structural_score = self.non_structural_weight * torch.mean(
            1.0 - non_structural
        )
        structural_score = self.structural_weight * torch.mean(1.0 - structural)

        return non_structural_score + structural_score


class SSIMScoreDiagnosisOperator(DiagnosisOperator, PixelComparisonOperator):
    def __init__(
        self,
        name="SSIM score",
        image_filter=None,
        non_structural_const=1e-2,
        structural_const=3e-2,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        if image_filter is None:
            image_filter = get_default_ssim_image_filter()
        self.image_filter = image_filter
        self.non_structural_const = non_structural_const
        self.structural_const = structural_const

    def _image_to_repr(self, image):
        return calculate_ssim_repr(rgb_to_grayscale(image), self.image_filter)

    def _input_image_to_repr(self, image, ctx):
        return self._image_to_repr(image)

    def _target_image_to_repr(self, image):
        repr = self._image_to_repr(image)
        ctx = calculate_simplified_ssim_ctx(
            image, non_structural_const=self.non_structural_const
        )
        return repr, ctx

    def _calculate_score(self, input_repr, target_repr, ctx):
        simplified_ssim = calculate_simplified_ssim(
            input_repr, target_repr, ctx, self.image_filter
        )
        return torch.mean(simplified_ssim, (1, 2, 3))
