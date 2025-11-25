import torch
from torchmetrics import Metric
import numpy as np

class CAFAGenericMetric(Metric):
    """
    A generic metric for CAFA-style evaluation. It calculates the maximum F-score (F-max)
    over a range of thresholds. It can compute both standard F-max and a weighted F-max
    if Information Content (IA) values are provided.

    This metric is designed to work with multi-label classification problems where
    predictions are continuous scores between 0 and 1.
    """
    def __init__(self, ia: torch.Tensor = None, dist_sync_on_step=False):
        """
        Args:
            ia (torch.Tensor, optional): A 1D tensor containing the Information Content (IA)
                value for each label. If provided, the metric will compute a weighted F-max.
                The IA values are typically derived from an ontology file (e.g., .obo).
                Defaults to None, which results in an unweighted F-max calculation.
            dist_sync_on_step (bool): Synchronize metric state across processes at each step.
                Defaults to False.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # Register state for predictions and targets.
        # 'dist_reduce_fx="cat"' ensures that when using distributed training,
        # predictions and targets from all processes are concatenated.
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        
        self.ia = ia

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets from a single batch.

        Args:
            preds: A tensor of predictions (scores), e.g., shape (batch_size, num_labels).
            target: A tensor of ground truth labels (binary), e.g., shape (batch_size, num_labels).
        """
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self):
        """
        Compute the F-max score over all accumulated predictions and targets.
        It iterates through thresholds from 0.01 to 0.99 to find the threshold
        that yields the highest F1-score. If IA values were provided, the calculation
        is weighted.

        Returns:
            A tuple containing:
            - fmax (float): The maximum F1-score (weighted or unweighted).
            - precision (float): The precision at the threshold that gives F-max.
            - recall (float): The recall at the threshold that gives F-max.
        """
        preds = torch.cat(self.preds, dim=0).cpu().numpy()
        targets = torch.cat(self.targets, dim=0).cpu().numpy()
        
        ia = self.ia.cpu().numpy() if self.ia is not None else None
        if ia is not None:
            # Ensure IA vector matches the number of labels
            if ia.shape[0] != preds.shape[1]:
                raise ValueError("The length of the IA vector must match the number of labels.")

        fmax = 0.0
        prec = 0.0
        rec = 0.0

        # Iterate over thresholds
        for t in range(1, 100):
            tau = t / 100.0
            
            # Binarize predictions based on the current threshold
            p = preds >= tau

            if ia is None:
                # --- Unweighted calculation ---
                tp = np.sum(np.logical_and(p, targets))
                fp = np.sum(np.logical_and(p, np.logical_not(targets)))
                fn = np.sum(np.logical_and(np.logical_not(p), targets))
            else:
                # --- Weighted calculation ---
                tp = np.sum(np.logical_and(p, targets) * ia)
                fp = np.sum(np.logical_and(p, np.logical_not(targets)) * ia)
                fn = np.sum(np.logical_and(np.logical_not(p), targets) * ia)

            # Calculate precision
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0

            # Calculate recall
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0

            # Calculate F1-score
            if precision + recall > 0:
                f = 2 * (precision * recall) / (precision + recall)
                # Update F-max if a better F1-score is found
                if f > fmax:
                    fmax = f
                    prec = precision
                    rec = recall
        
        return fmax, prec, rec

class CAFAMetrics(CAFAGenericMetric):
    """
    CAFA Metric calculator using torchmetrics.

    This class inherits from CAFAGenericMetric and is intended to be used
    within a PyTorch Lightning module or a standard PyTorch training loop.
    It can compute both a standard F-max and a weighted F-max using
    Information Content (IA) values.

    Example (Unweighted):
    ---------------------
    >>> # Mocks for demonstration
    >>> num_batches = 10
    >>> num_classes = 1500
    >>>
    >>> # 1. Initialize the metric for unweighted F-max
    >>> cafa_metric = CAFAMetrics()
    >>>
    >>> # 2. Simulate a validation loop
    >>> for _ in range(num_batches):
    ...     preds = torch.rand(32, num_classes)
    ...     targets = torch.randint(0, 2, (32, num_classes))
    ...     cafa_metric.update(preds, targets)
    >>>
    >>> # 3. Compute the final metric
    >>> fmax, precision, recall = cafa_metric.compute()
    >>> print(f"Unweighted F-max: {fmax:.4f}")

    Example (Weighted):
    -------------------
    >>> # 1. Initialize the metric with IA values for weighted F-max
    >>> # In a real scenario, IA values are derived from ontology files.
    >>> ia_values = torch.rand(num_classes)
    >>> cafa_metric_weighted = CAFAMetrics(ia=ia_values)
    >>>
    >>> # 2. Simulate a validation loop
    >>> for _ in range(num_batches):
    ...     preds = torch.rand(32, num_classes)
    ...     targets = torch.randint(0, 2, (32, num_classes))
    ...     cafa_metric_weighted.update(preds, targets)
    >>>
    >>> # 3. Compute the final metric
    >>> fmax_w, _, _ = cafa_metric_weighted.compute()
    >>> print(f"Weighted F-max: {fmax_w:.4f}")
    """
    def __init__(self, ia: torch.Tensor = None, dist_sync_on_step=False):
        super().__init__(ia=ia, dist_sync_on_step=dist_sync_on_step)
