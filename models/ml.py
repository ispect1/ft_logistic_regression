"""ML Part"""
import random
import numpy as np  # type: ignore
from config import ETA, MAX_ITER

from utils.ft_math import mean, std
from utils.ft_typing import NotFittedError


class Saver:
    """
    Sae attributes
    """
    def __init__(self, **kwargs):
        """

        Args:
            **kwargs:
        """
        self.attrs = list(kwargs)

    def save(self):
        """

        Returns:

        """
        json_data = {}
        for key in self.attrs:
            curr_data = getattr(self, key)
            if isinstance(curr_data, np.ndarray):
                curr_data = curr_data.tolist()
            json_data[key] = curr_data
        return json_data

    def download(self, json_data):
        """

        Args:
            json_data:

        Returns:

        """
        for key in self.attrs:
            curr_data = json_data[key]
            if isinstance(curr_data, list):
                curr_data = np.array(curr_data)
            setattr(self, key, curr_data)
        return self


class MyLogisticRegression(Saver):
    """
    Log reg
    """
    def __init__(self):
        self._eta = 1e-2
        self._max_iter = 1e4
        self._seed = 21
        self.w = None  # pylint: disable=C0103
        self.classes_ = None
        self._is_trained = False
        super().__init__(w=self.w, _is_trained=self._is_trained,
                         classes_=self.classes_, _seed=self._seed,
                         _eta=self._eta, _max_iter=self._max_iter)

    @property
    def coef_(self):
        """

        Returns:

        """
        return self.w

    @staticmethod
    def _sigmoid(margin):
        return 1 / (1 + np.exp(-margin))

    def stochastic_gradient_step(self, matrix, y_true, weight, train_ind, eta=0.01):  # pylint: disable=too-many-arguments
        """

        Args:
            matrix:
            y_true:
            weight:
            train_ind:
            eta:

        Returns:

        """
        one_x = matrix[train_ind]
        one_y = y_true[train_ind]
        return weight + eta * one_y * one_x * (
                1 - self._sigmoid(one_y * np.dot(one_x, weight)))

    def full_gradient_step(self, matrix, y_true, weight, eta=0.01):
        """

        Args:
            matrix:
            y_true:
            weight:
            eta:

        Returns:

        """
        return weight + eta * np.dot(
            matrix.T, y_true - self._sigmoid(matrix.dot(weight)))

    @staticmethod
    def _binarization_y(y_true, positive_class_name):
        return np.where(y_true == positive_class_name, 1, -1)

    def fit(self, matrix, y_true, eta=ETA, max_iter=MAX_ITER,  # pylint: disable=too-many-arguments
            max_weight_dist=None, seed=None, mode='stochastic'):
        """

        Args:
            matrix:
            y_true:
            eta:
            max_iter:
            max_weight_dist:
            seed:
            mode:

        Returns:

        """
        assert mode in ('full', 'stochastic')
        self._eta = eta or self._eta
        self._max_iter = max_iter or self._max_iter
        max_weight_dist = max_weight_dist or 1e-6
        self._seed = seed or self._seed
        random.seed(self._seed)
        scaler_matrix = np.insert(matrix, 0, 1, axis=1)
        self.w = []
        self.classes_ = []
        for y_name in np.unique(y_true):
            self.classes_.append(y_name)
            bin_y = self._binarization_y(y_true, y_name)
            weight_dist = float('inf')
            iter_num = 0
            weight = np.ones(scaler_matrix.shape[1])
            while weight_dist > max_weight_dist and iter_num < max_iter:
                # порождаем псевдослучайный
                # индекс объекта обучающей выборки
                iter_num += 1
                if mode == 'stochastic':
                    random_ind = random.randint(0, len(scaler_matrix) - 1)
                    weight = self.stochastic_gradient_step(scaler_matrix, bin_y, weight,
                                                           random_ind, eta=eta)
                else:
                    weight = self.full_gradient_step(scaler_matrix, bin_y, weight, eta=eta)

            self.w.append(weight)
        self.w = np.array(self.w)
        self.classes_ = np.array(self.classes_)
        self._is_trained = True
        return self

    def _predict_one(self, obj):
        """

        Args:
            obj:

        Returns:

        """
        return self.classes_[np.argmax([obj.dot(w) for w in self.w])]

    def _predict_proba_one(self, obj):
        """

        Args:
            obj:

        Returns:

        """
        return np.array([self._sigmoid(obj.dot(w)) for w in self.w])

    def predict(self, matrix):
        """

        Args:
            :

        Returns:

        """
        if not self._is_trained:
            raise NotFittedError()
        return np.array([self._predict_one(x) for x in np.insert(matrix, 0, 1, axis=1)])

    def predict_proba(self, matrix):
        """

        Args:
            matrix:

        Returns:

        """
        if not self._is_trained:
            raise NotFittedError()
        return np.array([self._predict_proba_one(x) for x in np.insert(matrix, 0, 1, axis=1)])


class Scaler(Saver):
    """
    Scaling data
    """
    def __init__(self):
        self.std = None
        self.mean = None
        self.count_feature = None
        self._is_trained = False
        super().__init__(_is_trained=self._is_trained, mean=self.mean, std=self.std,
                         count_feature=self.count_feature)

    def fit(self, matrix):
        """

        Args:
            matrix:

        Returns:

        """
        self._is_trained = True

        self.std = []
        self.mean = []
        self.count_feature = matrix.shape[1]
        for feature_idx in range(self.count_feature):
            if isinstance(matrix, np.ndarray):
                data = matrix[:, feature_idx]
            else:
                data = matrix[list(matrix.columns)[feature_idx]]
            curr_mean = mean(data)
            curr_std = std(data)
            self.mean.append(curr_mean)
            self.std.append(curr_std)
        self.mean = np.array(self.mean).reshape(1, -1)
        self.std = np.array(self.std).reshape(1, -1)

        return self

    def transform(self, matrix):
        """

        Args:
            matrix:

        Returns:

        """
        if not self._is_trained:
            return matrix
        assert matrix.shape[1] == self.count_feature, f'X has {matrix.shape[1]} features,' \
                                                      f'but this scaler is expecting' \
                                                      f' {self.count_feature} features as input.'
        return (matrix - self.mean) / self.std

    def fit_transform(self, matrix):
        """

        Args:
            matrix:

        Returns:

        """
        return self.fit(matrix).transform(matrix)
