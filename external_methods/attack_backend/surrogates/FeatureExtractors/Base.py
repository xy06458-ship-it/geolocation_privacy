"""Base classes for feature extraction and ensemble loss."""

import torch
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Any, Dict


class BaseFeatureExtractor(nn.Module):
    """Abstract base class for feature extractors."""

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def _unwrap_features(self, outputs: Any) -> Tensor:
        """Convert model outputs to a 2D feature tensor [B, D]."""
        if torch.is_tensor(outputs):
            feats = outputs
        elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            feats = outputs.image_embeds
        elif hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            feats = outputs.text_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            feats = outputs.last_hidden_state[:, 0, :]
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
            feats = outputs[0]
        else:
            raise TypeError(f"Unsupported model output type for features: {type(outputs)}")

        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        return feats

    @staticmethod
    def _l2_normalize(feats: Tensor) -> Tensor:
        return feats / (feats.norm(dim=1, keepdim=True) + 1e-12)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Extract features from input tensor."""
        pass


class EnsembleFeatureExtractor(BaseFeatureExtractor):
    """Ensemble of multiple feature extractors."""

    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Dict[int, Tensor]:
        """Extract features using all extractors.

        Args:
            x: Input image tensor.

        Returns:
            Dictionary mapping extractor index to features.
        """
        features = {}
        for i, model in enumerate(self.extractors):
            features[i] = model(x).squeeze()
        return features


class EnsembleFeatureLoss(nn.Module):
    """Loss function for ensemble feature matching with geo-semantic awareness."""

    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.text_truth = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        """Set ground truth features from target image."""
        self.ground_truth.clear()
        for model in self.extractors:
            self.ground_truth.append(model(x).to(x.device))

    def __call__(self, feature_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        """Compute feature similarity loss.

        Args:
            feature_dict: Dictionary of features from ensemble extractor.
            y: Unused, for compatibility.

        Returns:
            Average cosine similarity across all models.
        """
        loss = 0
        for index, model in enumerate(self.extractors):
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            loss += torch.mean(torch.sum(feature * gt, dim=1))
        return loss / len(self.extractors)

    @torch.no_grad()
    def set_geotext_truth(self, text: str, device: str):
        """Set text embedding for geo-semantic loss."""
        self.text_truth.clear()
        for model in self.extractors:
            self.text_truth.append(model.extract_text_features(text, device))

    def geo_loss(self, feature_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        """Compute geo-semantic aware loss.

        This loss encourages features to be orthogonal to geo-related text
        embeddings while preserving similarity to non-geo content.

        Args:
            feature_dict: Dictionary of features from ensemble extractor.
            y: Unused, for compatibility.

        Returns:
            Geo-semantic loss value.
        """
        loss = 0
        for index, model in enumerate(self.extractors):
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            text = self.text_truth[index]
            I = torch.eye(gt.shape[1], device=gt.device)

            # Compute geo-direction vector
            u_geo = gt - text
            u_geo = u_geo / (torch.norm(u_geo, p=2, dim=1, keepdim=True) + 1e-9)

            # Negative similarity in geo direction
            neg = torch.mean(torch.sum(feature * u_geo, dim=1))

            # Orthogonal projection to preserve non-geo content
            ortho_proj_matrix = I - u_geo.t() @ u_geo
            s1 = ortho_proj_matrix @ feature.t()
            s2 = ortho_proj_matrix @ gt.t()
            pos = torch.mean(torch.sum(s1 * s2, dim=0))

            loss += neg - pos

        return loss / len(self.extractors)
