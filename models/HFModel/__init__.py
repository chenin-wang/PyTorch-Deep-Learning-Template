from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)


_import_structure = {
    "configuration_hfmodel": [
        "hfmodelConfig",
        "hfmodelOnnxConfig",
        "hfmodelTextConfig",
        "hfmodelVisionConfig",
    ],
    "processing_hfmodel": ["hfmodelProcessor"],
    "tokenization_hfmodel": ["hfmodelTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_hfmodel_fast"] = ["hfmodelTokenizerFast"]

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_hfmodel"] = ["hfmodelFeatureExtractor"]
    _import_structure["image_processing_hfmodel"] = ["hfmodelImageProcessor"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_hfmodel"] = [
        "hfmodelModel",
        "hfmodelPreTrainedModel",
        "hfmodelTextModel",
        "hfmodelTextModelWithProjection",
        "hfmodelVisionModel",
        "hfmodelVisionModelWithProjection",
        "hfmodelForImageClassification",
    ]


if TYPE_CHECKING:
    from .configuration_hfmodel import (
        hfmodelConfig,
        hfmodelOnnxConfig,
        hfmodelTextConfig,
        hfmodelVisionConfig,
    )
    from .processing_hfmodel import hfmodelProcessor
    from .tokenization_hfmodel import hfmodelTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_hfmodel_fast import hfmodelTokenizerFast

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_hfmodel import hfmodelImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_hfmodel import (
            hfmodelForImageClassification,
            hfmodelModel,
            hfmodelPreTrainedModel,
            hfmodelTextModel,
            hfmodelTextModelWithProjection,
            hfmodelVisionModel,
            hfmodelVisionModelWithProjection,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
