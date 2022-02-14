import torch
from sklearn.model_selection import StratifiedKFold

class StratifiedSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)

def make_imbalance(
    X, y, imbalance_ratio=0.2, num_classes=10, verbose=False, **kwargs
    ):
    """
    DownSample features and Labels to create Imbalanced dataset with specific imbalance ratio.


    Parameters
    ----------
    X : torch.tensor
        The Features
    y : torch.tensor
        The labels
    imbalance_ratio : float
        The number of imbalance ration beetween the majority classes and the smallest minority class.
    num_classes : int
        Reduce the number of classes or use None to use all.
        Using the num_classes most populated classes.
    verbose : bool
    kwargs

    Returns
    -------
    X_resampled : torch.tensor
        The Downsampled features
    y_resampled : torch.tensor
        The Downsampled labels
    """
    #TODO generalize and refine function.
    target_stats = torch.eye(int(y.max() + 1), int(y.max() + 1))[y].sum(axis=0)
    samples_weights = torch.zeros(X.shape[0])
    if num_classes==None:
        num_classes = target_stats.shape[0]
    if verbose:
        print(f"The original target distribution in the dataset is: {target_stats}")

    n_occ, indices = torch.sort(target_stats, descending=True)[:num_classes]
    min_samples = n_occ[-1]
    max_samples = n_occ[0]
    if min_samples/max_samples > imbalance_ratio:
        raise ValueError("The imbalance ratio is too small, not enough samples in majority class.")
    desired_max = min_samples/imbalance_ratio
    new_indices = list()
    for i in range(num_classes):
        if min_samples/n_occ[i] <= imbalance_ratio:
            new_indices.append(torch.where(y==indices[i]))
        else:
            indices.append(torch.randperm(torch.where(y==indices[i])[0])[:desired_max])

    new_indices = torch.cat(new_indices)
    X_resampled = X[new_indices]
    y_resampled = y[new_indices]
    if verbose:
        print(f"Make the dataset imbalanced: {torch.eye(int(y_resampled.max() + 1), int(y_resampled.max() + 1))[y_resampled].sum(axis=0)}")

    return X_resampled, y_resampled