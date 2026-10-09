import timm
import torch
from timm.layers import freeze_batch_norm_2d
from transformers import (
    AutoImageProcessor,
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
    TimmBackboneConfig,
)

BACKBONE_NAME = "tf_efficientnet_b7"
# Three native backbone levels (strides 8/16/32) -> Deformable DETR generates the fourth
BACKBONE_OUT_INDICES = (2, 3, 4)

NUM_CLASSES = 11

id2label = {
    0: "Explosive",
    1: "Battery",
    2: "Modified laptop",
    3: "Modified parts",
    4: "Modified Mobile phone",
    5: "Modified Pager",
    6: "Modified Walkie Talkie",
    7: "Laptop",
    8: "Pager",
    9: "Mobile Phone",
    10: "Walkie-Talkie",
}

label2id = {v: k for k, v in id2label.items()}

processor = AutoImageProcessor.from_pretrained(
    "SenseTime/deformable-detr"
)

processor.size={
    "height":640,
    "width":640
}
processor.do_resize = True

# In transformers v5 the backbone is described ONLY by `backbone_config`.
# Setting `backbone` / `use_timm_backbone` / `backbone_kwargs` after the config
# is created does nothing, so pass an explicit TimmBackboneConfig instead.
backbone_config = TimmBackboneConfig(backbone=BACKBONE_NAME)
backbone_config.out_indices = list(BACKBONE_OUT_INDICES)

config = DeformableDetrConfig.from_pretrained(
    "SenseTime/deformable-detr",
    backbone_config=backbone_config,
    num_feature_levels=4,
    id2label=id2label,
    label2id=label2id,
)

model = DeformableDetrForObjectDetection(config)

# transformers v5 always builds timm backbones with pretrained=False, and its
# BatchNorm freezer turns EfficientNet's BatchNormAct2d into a plain BN,
# silently dropping the SiLU activations. Swap in a properly built backbone:
# ImageNet-pretrained B7 with activation-preserving frozen BN.
pretrained_backbone = timm.create_model(
    BACKBONE_NAME,
    pretrained=True,
    features_only=True,
    out_indices=BACKBONE_OUT_INDICES,
)
pretrained_backbone = freeze_batch_norm_2d(pretrained_backbone)

assert (
    pretrained_backbone.feature_info.channels()
    == model.model.backbone.intermediate_channel_sizes
), "Backbone channel mismatch with input projections"

model.model.backbone.model = pretrained_backbone

source_model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")

source_state = source_model.state_dict()
target_state = model.state_dict()

loaded = []
skipped_backbone = []
skipped_shape = []
skipped_missing = []

for name, param in source_state.items():

    # Do NOT copy ResNet-50 into EfficientNet-B7
    if name.startswith("model.backbone"):
        skipped_backbone.append(name)
        continue

    # Key doesn't exist in our target architecture
    if name not in target_state:
        skipped_missing.append(name)
        continue

    # Same key but incompatible tensor shape
    if param.shape != target_state[name].shape:
        skipped_shape.append(
            (name, tuple(param.shape), tuple(target_state[name].shape))
        )
        continue

    target_state[name] = param
    loaded.append(name)


model.load_state_dict(target_state)

del source_model
del source_state
del target_state

import gc
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"Loaded:           {len(loaded)}")
print(f"Skipped backbone: {len(skipped_backbone)}")
print(f"Skipped missing:  {len(skipped_missing)}")
print(f"Skipped shape:    {len(skipped_shape)}")

print("\nShape mismatches:")
for item in skipped_shape:
    print(item)