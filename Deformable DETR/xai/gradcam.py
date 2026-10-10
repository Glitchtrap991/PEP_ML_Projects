import torch
import torch.nn.functional as F


class DeformableDETRGradCAM:
    """Query- and class-specific Grad-CAM for EfficientNet-B7 Deformable DETR.

    The heatmap describes sensitivity of one query/class logit to a chosen
    spatial feature tensor. It is not a correctness or causality guarantee.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._capture)
        # Kept for compatibility with callers that inspect this attribute.
        self.backward_handle = None

    def _capture(self, module, inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[-1]
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise RuntimeError(
                f"Grad-CAM target layer must output [B,C,H,W]; got {type(output)} "
                f"with shape {getattr(output, 'shape', None)}"
            )
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient

    def generate(self, pixel_values, query_index, class_id, output_size=None):
        """Return (normalized CPU CAM [H,W], model outputs, raw target logit)."""
        if pixel_values.ndim != 4 or pixel_values.shape[0] != 1:
            raise ValueError("pixel_values must have shape [1,3,H,W]")

        self.model.eval()
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)

        # Ensures a gradient path through a frozen EfficientNet backbone.
        # Does not update weights or modify the caller's tensor.
        # Caller may be using inference_mode/no_grad for normal evaluation.
        # Clone outside inference mode to make an ordinary autograd tensor.
        with torch.inference_mode(False), torch.enable_grad():
            cam_input = pixel_values.detach().clone().requires_grad_(True)
            outputs = self.model(pixel_values=cam_input)
            logits = outputs.logits
            if logits.ndim != 3:
                raise RuntimeError(f"Expected [B,Q,C] logits, got {tuple(logits.shape)}")
            if not 0 <= query_index < logits.shape[1]:
                raise IndexError(f"query_index {query_index} outside [0,{logits.shape[1]})")
            if not 0 <= class_id < logits.shape[2]:
                raise IndexError(f"class_id {class_id} outside [0,{logits.shape[2]})")

            target_score = logits[0, query_index, class_id]
            target_score.backward()

            if self.activations is None or self.gradients is None:
                raise RuntimeError(
                    "Grad-CAM could not capture activations/gradients. "
                    "Verify that target_layer is used in the forward pass."
                )

            activations = self.activations
            gradients = self.gradients
            if activations.shape != gradients.shape:
                raise RuntimeError(
                    f"Activation/gradient shape mismatch: "
                    f"{tuple(activations.shape)} vs {tuple(gradients.shape)}"
                )
            if not torch.isfinite(gradients).all():
                raise RuntimeError("Non-finite Grad-CAM gradients")

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
            if output_size is None:
                output_size = pixel_values.shape[-2:]
            cam = F.interpolate(
                cam, size=tuple(output_size), mode="bilinear", align_corners=False
            )[0, 0]

            cam = cam - cam.min()
            cam = cam / cam.max().clamp_min(1e-8)
            result = cam.detach().cpu()
            score = float(target_score.detach().cpu())

        # Avoid holding the autograd graph between CAMs.
        self.activations = None
        self.gradients = None
        return result, outputs, score

    def generate_top_detection(self, pixel_values, output_size=None):
        """Quick diagnostic: explain the highest sigmoid query/class score.

        For evaluated detections, prefer generate() with the exact query and
        class selected by the inference postprocessor.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            probabilities = outputs.logits.sigmoid()
            flat_index = int(probabilities[0].argmax().item())
            num_classes = probabilities.shape[-1]
            query_index, class_id = divmod(flat_index, num_classes)
            confidence = float(probabilities[0, query_index, class_id].item())

        cam, outputs, target_logit = self.generate(
            pixel_values=pixel_values,
            query_index=query_index,
            class_id=class_id,
            output_size=output_size,
        )
        return {
            "cam": cam,
            "outputs": outputs,
            "query_index": query_index,
            "class_id": class_id,
            "confidence": confidence,
            "target_logit": target_logit,
        }

    def remove_hooks(self):
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None
        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None
        self.activations = None
        self.gradients = None

    def __del__(self):
        if hasattr(self, "forward_handle"):
            self.remove_hooks()