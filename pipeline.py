from sklearn.preprocessing import OneHotEncoder #, StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer


# class TargetClassImputer(BaseEstimator, TransformerMixin):
class TargetClassImputer(object):

    '''Fills in missing values in 'target_class' column with 'NO-TARGET' '''
    def __init__(self, *args):
        self.args = args

    def fit(self, X, y=None):
        return self

    def transform(self, X, *args):
        # return X.fillna(value='NO-TARGET')
        dat = X['target_classes'].fillna(value='')
        return dat


class TargetActivityAvg(object):
    '''Returns average of num_activities & num_targets.'''

    def __init__(self, *args):
        self.args = args

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dat = (X['num_activities'] + X['num_targets']) / 2
        return dat


class AssayRatio(object):
    '''Returns ratio of num_assays / avg(num_targets & num_activities).'''

    def __init__(self, *args):
        self.args = args

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        avg = (X['num_activities'] + X['num_targets']) / 2
        ratio =  X['num_assays'] / avg
        ratio = ratio.fillna(0)
        return ratio


class PassThrough(object):
    '''Takes average of num_activities & num_assays.
    Returns avg_activity_assay column'''

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self


pass_through = lambda x: x

def densify(X):
    return X.toarray()

def reshaper(X):
    return X.values.reshape(-1,1)


assay_transformer = CountVectorizer(lowercase=False, token_pattern=r'\[*(\w{1}),*\]*', )
target_transformer = CountVectorizer(token_pattern=r'([\w*-]{1,}),*')
