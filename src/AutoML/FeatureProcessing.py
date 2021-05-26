import os
import json
import copy
import ppscore
import inspect
import functools
import numpy as np
import pandas as pd
from boruta import BorutaPy
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import MiniBatchKMeans


class FeatureProcessing:
    # todo implement time budget

    def __init__(self,
                 max_lags=10,
                 max_diff=2,
                 information_threshold=0.99,
                 extract_features=True,
                 folder='',
                 mode=None,
                 version=''):
        self.X = None
        self.originalInput = None
        self.Y = None
        self.model = None
        self.mode = mode
        self.threshold = None
        # Register
        self.baseScore = {}
        self.colinearFeatures = None
        self.crossFeatures = None
        self.kMeansFeatures = None
        self.diffFeatures = None
        self.laggedFeatures = None
        # Parameters
        self.maxLags = max_lags
        self.maxDiff = max_diff
        self.informationThreshold = information_threshold
        self.extractFeatures = extract_features
        self.folder = folder if folder == '' or folder[-1] == '/' else folder + '/'
        self.version = version
        # Tests
        assert 0 <= max_lags < 50, 'Max lags needs to be within [0, 50]'
        assert 0 <= max_diff < 3, 'Max diff needs to be within [0, 3]'
        assert 0 < information_threshold < 1, 'Information threshold needs to be within [0, 1]'
        assert mode is not None, 'Mode needs to be specified (regression or classification'

    def extract(self, input_frame, output_frame):
        self._clean_set(input_frame, output_frame)
        if self.extractFeatures:
            # Manipulate features
            # Order of these doesn't matter, all take originalInput features
            self._remove_co_linearity()
            self._calc_baseline()
            self._add_cross_features()
            self._add_k_means_features()
            self._add_trigonometry_features()
            self._add_diff_features()
            self._add_lagged_features()
        return self.X

    def select(self, input_frame, output_frame):
        # Check if not exists
        if os.path.exists(self.folder + 'Sets_v%i.json' % self.version):
            return json.load(open(self.folder + 'Sets_v%i.json' % self.version, 'r'))

        # Execute otherwise
        else:
            # Clean
            self._clean_set(input_frame, output_frame)

            # Different Feature Sets
            result = {'PPS': self._predictive_power_score()}
            result['RFT'], result['RFI'] = self._random_forest_importance()
            # result['BP'] = self._borutapy()

            # Store & Return
            json.dump(result, open(self.folder + 'Sets_v%i.json' % self.version, 'w'))
            return result

    def transform(self, data, features, **args):
        # Split Features
        cross_features = [k for k in features if '__x__' in k or '__d__' in k]
        linear_features = [k for k in features if '__sub__' in k or '__add__' in k]
        trigonometric_features = [k for k in features if 'sin__' in k or 'cos__' in k]
        k_means_features = [k for k in features if 'dist__' in k]
        diff_features = [k for k in features if '__diff__' in k]
        lag_features = [k for k in features if '__lag__' in k]
        original_features = [k for k in features if '__' not in k]

        # Make sure centers are provided if kMeansFeatures are nonzero
        if len(k_means_features) != 0:
            if 'k_means' not in args:
                raise ValueError('For K-Means features, the Centers need to be provided.')
            k_means = args['k_means']

        # Fill missing features for normalization
        required = copy.copy(original_features)
        required += [s.split('__')[::2] for s in cross_features]
        required += [s.split('__')[::2] for s in linear_features]
        required += [k.split('__')[1] for k in trigonometric_features]
        required += [s.split('__diff__')[0] for s in diff_features]
        required += [s.split('__lag__')[0] for s in lag_features]
        if len(k_means_features) != 0:
            required += [k for k in k_means.keys()]
        for k in [k for k in required if k not in data.keys()]:
            data.loc[:, k] = np.zeros(len(data))

        # Select
        X = data[original_features]

        # Multiplicative features
        for key in cross_features:
            if '__x__' in key:
                key_a, key_b = k.split('__x__')
                feature = X[key_a] * X[key_b]
                X.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)
            else:
                key_a, key_b = k.split('__d__')
                feature = X[key_a] / X[key_b]
                X.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)

        # Linear features
        for key in linear_features:
            if '__sub__' in key:
                key_a, key_b = key.split('__sub__')
                feature = X[key_a] - X[key_b]
                X.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)
            else:
                key_a, key_b = key.split('__add__')
                feature = X[key_a] + X[key_b]
                X.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)

        # Differentiated features
        for k in diff_features:
            key, diff = k.split('__diff__')
            feature = data[key]
            for i in range(1, diff):
                feature = feature.diff().fillna(0)
            X.loc[:, k] = feature

        # K-Means features
        if len(k_means_features) != 0:
            # Organise data
            temp = copy.copy(data)
            centers = k_means.iloc[:-2]
            means = k_means.iloc[-2]
            stds = k_means.iloc[-1]
            # Normalize
            temp -= means
            temp /= stds
            # Calculate centers
            for key in k_means_features:
                ind = int(key[key.find('dist__') + 6: key.rfind('_')])
                X.loc[:, key] = np.sqrt(np.square(temp.loc[:, centers.keys()] - centers.iloc[ind]).sum(axis=1))

        # Lagged features
        for k in lag_features:
            key, lag = k.split('__lag__')
            X.loc[:, key] = data[key].shift(-int(lag), fill_value=0)

        # Trigonometric features
        for k in trigonometric_features:
            func, key = k.split('__')
            self.X[k] = getattr(np, func)(self.X[key])

        return X

    def export_function(self):
        code = inspect.getsource(self.transform)
        return """

        ############
        # Features #
        ############""" + code[code.find('\n'): code.rfind('\n', 0, code.rfind('\n'))]

    def _clean_set(self, input_frame, output_frame):
        assert isinstance(input_frame, pd.DataFrame), 'Input supports only Pandas DataFrame'
        assert isinstance(output_frame, pd.Series), 'Output supports only Pandas Series'
        if self.mode == 'classification':
            self.model = DecisionTreeClassifier(max_depth=3)
            if output_frame.nunique() > 2:
                self.mode = 'multiclass'
        elif self.mode == 'regression':
            self.model = DecisionTreeRegressor(max_depth=3)
        # Bit of necessary data cleaning (shouldn't change anything)
        input_frame = input_frame.astype('float32').replace(np.inf, 1e12).replace(-np.inf, -1e12).fillna(0)\
            .reset_index(drop=True)
        self.X = copy.copy(input_frame)
        self.originalInput = copy.copy(input_frame)
        self.Y = output_frame.replace([np.inf, -np.inf], 0).fillna(0).reset_index(drop=True)

    def _calc_baseline(self):
        baseline = {}
        for key in self.originalInput.keys():
            baseline[key] = self._analyse_feature(self.X[key])
        self.baseline = pd.DataFrame(baseline).max(axis=1)

    def _analyse_feature(self, feature):
        feature = feature.clip(lower=1e-12, upper=1e12).fillna(0).values.reshape((-1, 1))
        m = copy.copy(self.model)
        m.fit(feature, self.Y)
        if self.mode == 'multiclass':
            return [m.score(feature, self.Y, self.Y == i) for i in self.Y.unique()]
        else:
            return [m.score(feature, self.Y)]

    def _accept_feature(self, candidate):
        if (candidate > self.baseline.values).any():
            return True
        else:
            return False

    @ staticmethod
    def _select_features(scores):
        # Convert dict to dataframe
        scores = pd.DataFrame(scores)

        # Select inds
        features_per_bin = 50
        inds = []
        for score in range(len(scores)):
            inds += [i for i, k in enumerate(scores.keys())
                     if k in scores.loc[score].sort_values(ascending=False).keys()[:features_per_bin]]

        # Return Keys
        return list(scores.keys()[np.unique(inds)])

    def _remove_co_linearity(self):
        """
        Calculates the Pearson Correlation Coefficient for all input features.
        Those higher than the information threshold are linearly codependent (i.e., describable by y = a x + b)
        These features add little to no information and are therefore removed.
        """
        # Check if not already executed
        if os.path.exists(self.folder + 'Colinear_v%i.json' % self.version):
            print('[Features] Loading Colinear features')
            self.colinearFeatures = json.load(open(self.folder + 'Colinear_v%i.json' % self.version, 'r'))

        # Else run
        else:
            print('[Features] Analysing colinearity')
            nk = len(self.X.keys())
            norm = (self.X - self.X.mean(skipna=True, numeric_only=True)).to_numpy()
            ss = np.sqrt(np.sum(norm ** 2, axis=0))
            corr_mat = np.zeros((nk, nk))
            for i in range(nk):
                for j in range(nk):
                    if i == j:
                        continue
                    if corr_mat[i, j] == 0:
                        c = abs(np.sum(norm[:, i] * norm[:, j]) / (ss[i] * ss[j]))
                        corr_mat[i, j] = c
                        corr_mat[j, i] = c
            upper = np.triu(corr_mat)
            self.colinearFeatures = self.X.keys()[np.sum(upper > self.informationThreshold, axis=0) > 0].to_list()
            json.dump(self.colinearFeatures, open(self.folder + 'Colinear_v%i.json' % self.version, 'w'))

        self.originalInput = self.originalInput.drop(self.colinearFeatures, axis=1)
        print('[Features] Removed %i Co-Linear features (%.3f %% threshold)' % (
            len(self.colinearFeatures), self.informationThreshold))

    @staticmethod
    def _static_multiply(model, data, labels, features):
        feature = data[features[0]] * data[features[1]]
        feature = feature.clip(lower=1e-12, upper=1e12).fillna(0).values.reshape((-1, 1))
        model.fit(feature, labels)
        return (features[0] + '__x__' + features[1],
                max([model.score(feature, labels, labels == i) for i in labels.unique()]))

    def _add_multi_features_mp(self):
        print('[Features] Listing Multiplication Features')
        features = []
        for i, key_a in enumerate(self.originalInput.keys()):
            for j, key_b in enumerate(self.originalInput.keys()):
                if i >= j:
                    continue
                features.append((key_a, key_b))
        print('[Features] Analysing {} Multiplication Features'.format(len(features)))
        scores = dict(process_map(functools.partial(self._static_multiply, self.model, self.X, self.Y),
                                  features, max_workers=8, chunksize=min(100, int(len(features) / 8 / 8))))
        self.multiFeatures = self._select_features(scores)

    def _add_cross_features(self):
        """
        Calculates cross-feature features with m and multiplication.
        Should be limited to say ~500.000 features (runs about 100-150 features / second)
        """
        # Check if not already executed
        if os.path.exists(self.folder + 'crossFeatures_v%i.json' % self.version) and False:
            self.crossFeatures = json.load(open(self.folder + 'crossFeatures_v%i.json' % self.version, 'r'))
            print('[Features] Loaded %i cross features' % len(self.crossFeatures))

        # Else, execute
        else:
            print('[Features] Analysing cross features')
            scores = {}
            n_keys = len(self.originalInput.keys())

            # Analyse Cross Features
            for i, key_a in enumerate(tqdm(self.originalInput.keys())):
                accepted_for_key_a = 0
                for j, key_b in enumerate(self.originalInput.keys()[np.random.permutation(n_keys)]):
                    # Skip if they're the same
                    if key_a == key_b:
                        continue
                    # Skip rest if key_a is not useful in first min(50, 30%) (uniform)
                    if accepted_for_key_a == 0 and j > min(50, int(n_keys / 3)):
                        continue

                    # Analyse Division
                    feature = self.X[key_a] / self.X[key_b]
                    score = self._analyse_feature(feature)
                    # Accept or not
                    if self._accept_feature(score):
                        scores[key_a + '__d__' + key_b] = score
                        accepted_for_key_a += 1

                    # Multiplication i * j == j * i, so skip if j >= i
                    if j >= i:
                        continue

                    # Analyse Multiplication
                    feature = self.X[key_a] * self.X[key_b]
                    score = self._analyse_feature(feature)
                    # Accept or not
                    if self._accept_feature(score):
                        scores[key_a + '__x__' + key_b] = score
                        accepted_for_key_a += 1

            # Select valuable features
            self.crossFeatures = self._select_features(scores)

        # Add features
        for k in self.crossFeatures:
            if '__x__' in k:
                key_a, key_b = k.split('__x__')
                feature = self.X[key_a] * self.X[key_b]
                self.X[k] = feature.clip(lower=1e-12, upper=1e12).fillna(0)
            else:
                key_a, key_b = k.split('__d__')
                feature = self.X[key_a] / self.X[key_b]
                self.X[k] = feature.clip(lower=1e-12, upper=1e12).fillna(0)

        # Store
        json.dump(self.crossFeatures, open(self.folder + 'crossFeatures_v%i.json' % self.version, 'w'))
        print('[Features] Added %i cross features' % len(self.crossFeatures))

    def _add_trigonometry_features(self):
        """
        Calculates trigonometry features with sinus, cosines
        """
        # Check if not already executed
        if os.path.exists(self.folder + 'trigoFeatures_v%i.json' % self.version):
            self.trigoFeatures = json.load(open(self.folder + 'trigoFeatures_v%i.json' % self.version, 'r'))
            print('[Features] Loaded %i trigonometric features' % len(self.trigoFeatures))

        # Else, execute
        else:
            print('[Features] Analysing Trigonometric features')

            scores = {}
            for key in tqdm(self.originalInput.keys()):
                # Sinus feature
                sin_feature = np.sin(self.X[key])
                score = self._analyse_feature(sin_feature)
                if self._accept_feature(score):
                    score['sin__' + key] = score

                # Cosinus feature
                cos_feature = np.cos(self.X[key])
                score = self._analyse_feature(cos_feature)
                if self._accept_feature(score):
                    scores['cos__' + key] = score

            # Select valuable features
            self.trigoFeatures = self._select_features(scores)

        # Add features
        for k in self.trigoFeatures:
            func, key = k.split('__')
            self.X[k] = getattr(np, func)(self.X[key])

        # Store
        json.dump(self.trigoFeatures, open(self.folder + 'trigoFeatures_v%i.json' % self.version, 'w'))
        print('[Features] Added %i trigonometric features' % len(self.trigoFeatures))

    def _add_additive_features(self):
        """
        Calculates simple additive and subtractive features
        """
        # Load if available
        if os.path.exists(self.folder + 'addFeatures_v%i.json' % self.version):
            self.addFeatures = json.load(open(self.folder + 'addFeatures_v%i.json' % self.version, 'r'))
            print('[Features] Loaded %i additive features' % len(self.addFeatures))

        # Else, execute
        else:
            scores = {}
            n_keys = len(self.originalInput.keys())
            for i, key_a in enumerate(self.originalInput.keys()):
                accepted_for_key_a = 0
                for j, key_b in enumerate(self.originalInput.keys()[np.random.permutation(n_keys)]):
                    # Skip if they're the same
                    if key_a == key_b:
                        continue
                    # Skip rest if key_a is not useful in first min(50, 30%) (uniform)
                    if accepted_for_key_a == 0 and j > min(50, int(n_keys / 3)):
                        continue

                    # Subtracting feature
                    feature = self.X[key_a] - self.X[key_b]
                    score = self._analyse_feature(feature)
                    if self._accept_feature(score):
                        scores[key_a + '__sub__' + key_b] = score
                        accepted_for_key_a += 1

                    # A + B == B + A, so skip if i > j
                    if j > i:
                        continue

                    # Additive feature
                    feature = self.X[key_a] + self.X[key_b]
                    score = self._analyse_feature(feature)
                    if self._accept_feature(score):
                        scores[key_a + '__add__' + key_b] = score
                        accepted_for_key_a += 1

            # Select valuable Features
            self.addFeatures = self._select_features(scores)

        # Add features
        for key in self.addFeatures:
            if '__sub__' in key:
                key_a, key_b = key.split('__sub__')
                feature = self.X[key_a] - self.X[key_b]
                self.X.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)
            else:
                key_a, key_b = key.split('__add__')
                feature = self.X[key_a] + self.X[key_b]
                self.X.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)

        # store
        json.dump(self.addFeatures, open(self.folder + 'addFeatures_v%i.json' % self.version, 'w'))
        print('[Features] Added %i additive features' % len(self.addFeatures))

    def _add_k_means_features(self):
        """
        Analyses the correlation of k-means features.
        k-means is a clustering algorithm which clusters the data.
        The distance to each cluster is then analysed.
        """
        # Check if not exist
        if os.path.exists(self.folder + 'K-MeansFeatures_v%i.json' % self.version):
            # Load features and cluster size
            self.kMeansFeatures = json.load(open(self.folder + 'K-MeansFeatures_v%i.json' % self.version, 'r'))
            k_means_data = pd.read_csv(self.folder + 'KMeans_v%i.csv' % self.version)
            print('[Features] Loaded %i K-Means features' % len(self.kMeansFeatures))

            # Prepare data
            data = copy.copy(self.originalInput)
            centers = k_means_data.iloc[:-2]
            means = k_means_data.iloc[-2]
            stds = k_means_data.iloc[-1]
            data -= means
            data /= stds

            # Add them
            for key in self.kMeansFeatures:
                ind = int(key[key.find('dist__') + 6: key.rfind('_')])
                self.X[key] = np.sqrt(np.square(data - centers.iloc[ind]).sum(axis=1))

        # If not executed, analyse all
        else:
            print('[Features] Calculating and Analysing K-Means features')
            # Prepare data
            data = copy.copy(self.originalInput)
            means = data.mean()
            stds = data.std()
            stds[stds == 0] = 1
            data -= means
            data /= stds

            # Determine clusters
            clusters = min(max(int(np.log10(len(self.originalInput)) * 8), 8), len(self.originalInput.keys()))
            kmeans = MiniBatchKMeans(n_clusters=clusters)
            column_names = ['dist__%i_%i' % (i, clusters) for i in range(clusters)]
            distances = pd.DataFrame(columns=column_names, data=kmeans.fit_transform(data))
            distances = distances.clip(lower=1e-12, upper=1e12).fillna(0)

            # Analyse correlation
            scores = {}
            for key in tqdm(distances.keys()):
                score = self._analyse_feature(distances[key])
                if self._accept_feature(score):
                    scores[key] = score

            # Add the valuable features
            self.kMeansFeatures = self._select_features(scores)
            for k in self.kMeansFeatures:
                self.X[k] = distances[k]

            # Create output
            centers = pd.DataFrame(columns=self.originalInput.keys(), data=kmeans.cluster_centers_)
            centers = centers.append(means, ignore_index=True)
            centers = centers.append(stds, ignore_index=True)
            centers.to_csv(self.folder + 'KMeans_v%i.csv' % self.version, index=False)
            json.dump(self.kMeansFeatures, open(self.folder + 'K-MeansFeatures_v%i.json' % self.version, 'w'))
            print('[Features] Added %i K-Means features (%i clusters)' % (len(self.kMeansFeatures), clusters))

    def _add_diff_features(self):
        """
        Analyses whether the differenced signal of the data should be included.
        """

        # Check if we're allowed
        if self.maxDiff == 0:
            print('[Features] Differenced features skipped, max diff = 0')
            return

        # Check if exist
        if os.path.exists(self.folder + 'diffFeatures_v%i.json' % self.version):
            self.diffFeatures = json.load(open(self.folder + 'diffFeatures_v%i.json' % self.version, 'r'))
            print('[Features] Loaded %i differenced features' % len(self.diffFeatures))

        # If not exist, execute
        else:
            print('[Features] Analysing differenced features')
            # Copy data so we can diff without altering original data
            keys = self.originalInput.keys()
            diff_input = copy.copy(self.originalInput)

            # Calculate scores
            scores = {}
            for diff in tqdm(range(1, self.maxDiff + 1)):
                diff_input = diff_input.diff().fillna(0)
                for key in keys:
                    score = self._analyse_feature(diff_input[key])
                    if self._accept_feature(score):
                        scores[key + '__diff__%i' % diff] = score

            # Select the valuable features
            self.diffFeatures = self._select_features(scores)
            print('[Features] Added %i differenced features' % len(self.diffFeatures))

        # Add Differenced Features
        for k in self.diffFeatures:
            key, diff = k.split('__diff__')
            feature = self.X[key]
            for i in range(1, diff):
                feature = feature.diff().clip(lower=1e-12, upper=1e12).fillna(0)
            self.X[k] = feature
        json.dump(self.diffFeatures, open(self.folder + 'diffFeatures_v%i.json' % self.version, 'w'))

    def _add_lagged_features(self):
        """
        Analyses the correlation of lagged features (value of sensor_x at t-1 to predict target at t)
        """
        # Check if allowed
        if self.maxLags == 0:
            print('[Features] Lagged features skipped, max lags = 0')
            return

        # Check if exists
        if os.path.exists(self.folder + 'laggedFeatures_v%i.json' % self.version):
            self.laggedFeatures = json.load(open(self.folder + 'laggedFeatures_v%i.json' % self.version, 'r'))
            print('[Features] Loaded %i lagged features' % len(self.laggedFeatures))

        # Else execute
        else:
            print('[Features] Analysing lagged features')
            keys = self.originalInput.keys()
            scores = {}
            for lag in tqdm(range(1, self.maxLags)):
                for key in keys:
                    score = self._analyse_feature(self.X[key].shift(-1))
                    if self._accept_feature(score):
                        scores[key + '__lag__%i' % lag] = score

            # Select
            self.laggedFeatures = self._select_features(scores)
            print('[Features] Added %i lagged features' % len(self.laggedFeatures))

        # Add selected
        for k in self.laggedFeatures:
            key, lag = k.split('__lag__')
            self.X[k] = self.originalInput[key].shift(-int(lag), fill_value=0)

        # Store
        json.dump(self.laggedFeatures, open(self.folder + 'laggedFeatures_v%i.json' % self.version, 'w'))

    def _predictive_power_score(self):
        """
        Calculates the Predictive Power Score (https://github.com/8080labs/ppscore)
        Assymmetric correlation based on single decision trees trained on 5.000 samples with 4-Fold validation.
        """
        print('[Features] Determining features with PPS')
        data = self.X.copy()
        data['target'] = self.Y.copy()
        pp_score = ppscore.predictors(data, "target")
        pp_cols = pp_score['x'][pp_score['ppscore'] != 0].to_list()
        print('[Features] Selected %i features with Predictive Power Score' % len(pp_cols))
        return pp_cols

    def _random_forest_importance(self):
        """
        Calculates Feature Importance with Random Forest, aka Mean Decrease in Gini Impurity
        Symmetric correlation based on multiple features and multiple trees ensemble
        """
        print('[Features] Determining features with RF')
        if self.mode == 'regression':
            rf = RandomForestRegressor().fit(self.X, self.Y)
        elif self.mode == 'classification' or self.mode == 'multiclass':
            rf = RandomForestClassifier().fit(self.X, self.Y)
        else:
            raise ValueError('Method not implemented')
        fi = rf.feature_importances_
        sfi = fi.sum()
        ind = np.flip(np.argsort(fi))
        # Info Threshold
        ind_keep = [ind[i] for i in range(len(ind)) if fi[ind[:i]].sum() <= self.informationThreshold * sfi]
        threshold = self.X.keys()[ind_keep].to_list()
        ind_keep = [ind[i] for i in range(len(ind)) if fi[i] > sfi / 100]
        increment = self.X.keys()[ind_keep].to_list()
        print('[Features] Selected %i features with RF thresholded' % len(threshold))
        print('[Features] Selected %i features with RF increment' % len(increment))
        return threshold, increment

    def _borutapy(self):
        print('[Features] Determining features with Boruta')
        if self.mode == 'regression':
            rf = RandomForestRegressor()
        elif self.mode == 'classification' or self.mode == 'multiclass':
            rf = RandomForestClassifier()
        selector = BorutaPy(rf, n_estimators='auto', verbose=0)
        selector.fit(self.X.to_numpy(), self.Y.to_numpy())
        bp_cols = self.X.keys()[selector.support_].to_list()
        print('[Features] Selected %i features with Boruta' % len(bp_cols))
        return bp_cols
