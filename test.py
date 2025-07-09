from scipy.stats import mode

from sklearn.base import RegressorMixin

class MeanRegressor(RegressorMixin):
    # Predicts the mean of y_train
    def fit(self, X=None, y=None):
        self.is_fitted_=True
        return self

    def predict(self, X=None):
        return np.full(shape=X.shape[0], fill_value=self.param)
