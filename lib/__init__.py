# lib/__init__.py
# Core Model
from .soct_sam_model import SOCT_SAM
from .rrsis_sam3_model import RRSIS_SAM3
from .rs_adapters import inject_lora_adapters
from .ot_feature_alignment import OTFeatureAligner
from .ot_loss import OTSegmentationLoss
from .dynamic_lora import DynamicLoRALinear, inject_dynamic_lora_adapters
from .contrastive_loss import ContrastiveLoss
from .multiscale_ot_alignment import MultiScaleOTAligner
from .ohem_loss import EnhancedOHEMLoss, OHEMLoss, FocalDiceLoss, BoundaryAwareLoss

# SOCT-SAM Modules
from .soct_alignment import SOCTAligner, DualConditionedSinkhorn, GeometricAggregationOperator
from .soct_loss import SOCTSupervisionLoss
from .language_parser import GLOBAL_PARSER, ReferringExpressionParser
from .prompt_generator import GroundingAwarePromptGenerator
