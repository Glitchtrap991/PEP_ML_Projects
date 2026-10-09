import torch
import torch.nn.functional as F


class DeformableDETRGradCAM:
    """
    Detection-specific Grad-CAM for our:

        EfficientNet-B7
            ->
        Deformable DETR
            ->
        11 IEDXRay classes

    A CAM is generated for a specific:
        - object query
        - predicted class

    This does NOT change the detector prediction.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = None
        self.backward_handle = None

        self._register_hooks()

    # ---------------------------------------------------------
    # Hooks
    # ---------------------------------------------------------

    def _register_hooks(self):

        def forward_hook(module, inputs, output):
            """
            Save spatial feature activations produced by the
            selected EfficientNet layer.
            """

            # Some modules may return tuples/lists.
            if isinstance(output, (tuple, list)):
                output = output[-1]

            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            """
            Save gradients of the selected detector score
            with respect to the selected feature map.
            """

            gradient = grad_output[0]

            if isinstance(gradient, (tuple, list)):
                gradient = gradient[-1]

            self.gradients = gradient

        self.forward_handle = self.target_layer.register_forward_hook(
            forward_hook
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                backward_hook
            )
        )

    # ---------------------------------------------------------
    # Generate Grad-CAM
    # ---------------------------------------------------------

    def generate(
        self,
        pixel_values,
        query_index,
        class_id,
        output_size=None,
    ):
        """
        Generate Grad-CAM for one Deformable-DETR detection.

        Parameters
        ----------
        pixel_values:
            Tensor [1, 3, H, W].

        query_index:
            DETR object-query index, e.g. 137.

        class_id:
            IEDXRay class index, e.g. 4 for
            "Modified Mobile phone".

        output_size:
            Optional (height, width) for final CAM.
            Defaults to pixel_values spatial size.

        Returns
        -------
        cam:
            Tensor [H, W] on CPU, normalized to [0, 1].

        outputs:
            Original Deformable-DETR outputs.

        target_score:
            Raw class logit used as the Grad-CAM target.
        """

        self.model.eval()

        self.activations = None
        self.gradients = None

        # Remove gradients left by previous CAMs.
        self.model.zero_grad(set_to_none=True)

        # -----------------------------------------------------
        # Forward pass
        # -----------------------------------------------------

        outputs = self.model(
            pixel_values=pixel_values
        )

        logits = outputs.logits

        if logits.ndim != 3:
            raise RuntimeError(
                f"Expected logits [B, Q, C], got {logits.shape}"
            )

        num_queries = logits.shape[1]
        num_classes = logits.shape[2]

        if not 0 <= query_index < num_queries:
            raise IndexError(
                f"query_index={query_index}, "
                f"but model has {num_queries} queries."
            )

        if not 0 <= class_id < num_classes:
            raise IndexError(
                f"class_id={class_id}, "
                f"but model has {num_classes} classes."
            )

        # -----------------------------------------------------
        # Detection-specific target
        # -----------------------------------------------------

        target_score = logits[
            0,
            query_index,
            class_id,
        ]

        # -----------------------------------------------------
        # Backward pass
        # -----------------------------------------------------

        target_score.backward()

        if self.activations is None:
            raise RuntimeError(
                "No activations captured. "
                "Check the selected target layer."
            )

        if self.gradients is None:
            raise RuntimeError(
                "No gradients captured. "
                "The selected layer may not participate in "
                "the computation of this detection score."
            )

        activations = self.activations
        gradients = self.gradients

        if activations.ndim != 4:
            raise RuntimeError(
                "Grad-CAM requires a spatial feature tensor "
                f"[B,C,H,W], got {activations.shape}"
            )

        if gradients.ndim != 4:
            raise RuntimeError(
                "Expected gradients [B,C,H,W], "
                f"got {gradients.shape}"
            )

        # -----------------------------------------------------
        # Grad-CAM
        #
        # alpha_k = mean_{i,j} d(score)/d(A_kij)
        # CAM = ReLU(sum_k alpha_k * A_k)
        # -----------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(cam)

        # -----------------------------------------------------
        # Resize
        # -----------------------------------------------------

        if output_size is None:
            output_size = pixel_values.shape[-2:]

        cam = F.interpolate(
            cam,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]

        # -----------------------------------------------------
        # Normalize to [0, 1]
        # -----------------------------------------------------

        cam_min = cam.min()
        cam_max = cam.max()

        if (cam_max - cam_min).item() > 1e-8:
            cam = (
                (cam - cam_min)
                / (cam_max - cam_min)
            )
        else:
            cam = torch.zeros_like(cam)

        return (
            cam.detach().cpu(),
            outputs,
            target_score.detach().cpu().item(),
        )

    # ---------------------------------------------------------
    # Automatically choose highest-confidence query
    # ---------------------------------------------------------

    def generate_top_detection(
        self,
        pixel_values,
        output_size=None,
    ):
        """
        Find the highest-scoring query/class pair and
        generate its Grad-CAM.

        Useful for quick testing.
        """

        self.model.eval()

        with torch.no_grad():

            outputs = self.model(
                pixel_values=pixel_values
            )

            probabilities = outputs.logits.sigmoid()

            # [1, Q, C]
            flat_index = probabilities[0].argmax()

            num_classes = probabilities.shape[-1]

            query_index = (
                flat_index // num_classes
            ).item()

            class_id = (
                flat_index % num_classes
            ).item()

            confidence = probabilities[
                0,
                query_index,
                class_id,
            ].item()

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

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def remove_hooks(self):

        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None

        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    def __del__(self):
        self.remove_hooks()