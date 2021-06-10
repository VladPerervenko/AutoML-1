import os
import time
import json
import copy
import ppscore
import inspect
import warnings
import functools
import itertools
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
    # todo improve thresholds of RFT / RFI

    def __init__(self,
                 max_lags=10,
                 max_diff=2,
                 information_threshold=0.99,
                 extract_features=True,
                 folder='',
                 mode='classification',
                 timeout=900,
                 version=0):
        """
        Automatically extracts and selects features. Removes Co-Linear Features.
        Included Feature Extraction algorithms:
        - Multiplicative Features
        - Dividing Features
        - Additive Features
        - Subtractive Features
        - Trigonometric Features
        - K-Means Features
        - Lagged Features
        - Differencing Features

        Included Feature Selection algorithms:
        - Random Forest Feature Importance (Threshold and Increment)
        - Predictive Power Score
        - Boruta

        Parameters
        ----------
        max_lags int: Maximum lags for lagged features to analyse
        max_diff int: Maximum differencing order for differencing features
        information_threshold float: Information threshold for co-linear features
        extract_features bool: Whether or not to extract features
        folder str: Parent folder for results
        mode str: classification / regression
        timeout int: Feature Extraction can be exhausting for many features, this limits the scope
        version int: To version all stored files
        """
        # Tests
        assert isinstance(max_lags, int)
        assert isinstance(max_diff, int)
        assert isinstance(information_threshold, float)
        assert isinstance(extract_features, bool)
        assert isinstance(folder, str)
        assert isinstance(mode, str)
        assert isinstance(timeout, int)
        assert isinstance(version, int)
        assert 0 <= max_lags < 50, 'Max lags needs to be within [0, 50]'
        assert 0 <= max_diff < 3, 'Max diff needs to be within [0, 3]'
        assert 0 < information_threshold < 1, 'Information threshold needs to be within [0, 1]'
        assert mode.lower() in ['classification', 'regression'], \
            'Mode needs to be specified (regression or classification)'

        # Parameters
        self.x = None
        self.originalInput = None
        self.y = None
        self.model = None
        self.mode = mode
        self.timeout = timeout
        # Register
        self.baseScore = {}
        self.coLinearFeatures = []
        self.addFeatures = []
        self.crossFeatures = []
        self.trigoFeatures = []
        self.kMeansFeatures = []
        self.laggedFeatures = []
        self.diffFeatures = []
        # Parameters
        self.maxLags = max_lags
        self.maxDiff = max_diff
        self.informationThreshold = information_threshold
        self.extractFeatures = extract_features
        self.version = version
        self.folder = folder if folder == '' or folder[-1] == '/' else folder + '/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

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
        return self.x

    def select(self, input_frame, output_frame):
        # Check if not exists
        if os.path.exists(self.folder + 'Sets_v{}.json'.format(self.version)):
            return json.load(open(self.folder + 'Sets_v{}.json'.format(self.version), 'r'))

        # Execute otherwise
        else:
            # Clean
            self._clean_set(input_frame, output_frame)

            # Different Feature Sets
            result = {'PPS': self._predictive_power_score()}
            result['RFT'], result['RFI'] = self._random_forest_importance()
            # result['BP'] = self._borutapy()

            # Store & Return
            json.dump(result, open(self.folder + 'Sets_v{}.json'.format(self.version), 'w'))
            return result

    @staticmethod
    def transform(data, features, **args):
        # Split Features
        cross_features = [k for k in features if '__x__' in k or '__d__' in k]
        linear_features = [k for k in features if '__sub__' in k or '__add__' in k]
        trigonometric_features = [k for k in features if 'sin__' in k or 'cos__' in k]
        k_means_features = [k for k in features if 'dist__' in k]
        diff_features = [k for k in features if '__diff__' in k]
        lag_features = [k for k in features if '__lag__' in k]
        original_features = [k for k in features if '__' not in k]

        # Fill missing features for normalization
        required = copy.copy(original_features)
        required += list(itertools.chain.from_iterable([s.split('__')[::2] for s in cross_features]))
        required += list(itertools.chain.from_iterable([s.split('__')[::2] for s in linear_features]))
        required += list(itertools.chain.from_iterable([k.split('__')[1] for k in trigonometric_features]))
        required += list(itertools.chain.from_iterable([s.split('__diff__')[0] for s in diff_features]))
        required += list(itertools.chain.from_iterable([s.split('__lag__')[0] for s in lag_features]))

        # Make sure centers are provided if kMeansFeatures are nonzero
        k_means = None
        if len(k_means_features) != 0:
            if 'k_means' not in args:
                raise ValueError('For K-Means features, the Centers need to be provided.')
            k_means = args['k_means']
            required += [k for k in k_means.keys()]

        # Remove duplicates from required
        required = list(set(required))

        # Impute missing keys
        missing_keys = [k for k in required if k not in data.keys()]
        if len(missing_keys) > 0:
            warnings.warn('Imputing {} keys'.format(len(missing_keys)))
        for k in missing_keys:
            data.loc[:, k] = np.zeros(len(data))

        # Start Output with selected original features
        x = data[original_features]

        # Multiplicative features
        for key in cross_features:
            if '__x__' in key:
                key_a, key_b = key.split('__x__')
                feature = data[key_a] * data[key_b]
                x.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)
            else:
                key_a, key_b = key.split('__d__')
                feature = data[key_a] / data[key_b]
                x.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)

        # Linear features
        for key in linear_features:
            if '__sub__' in key:
                key_a, key_b = key.split('__sub__')
                feature = data[key_a] - data[key_b]
                x.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)
            else:
                key_a, key_b = key.split('__add__')
                feature = data[key_a] + data[key_b]
                x.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)

        # Differentiated features
        for k in diff_features:
            key, diff = k.split('__diff__')
            feature = data[key]
            for i in range(1, int(diff)):
                feature = feature.diff().fillna(0)
            x.loc[:, k] = feature

        # K-Means features
        if len(k_means_features) != 0:
            # Organise data
            centers = k_means.iloc[:-2]
            means = k_means.iloc[-2]
            stds = k_means.iloc[-1]
            temp = copy.deepcopy(data.loc[:, centers.keys()])
            # Normalize
            temp -= means
            temp /= stds
            # Calculate centers
            for key in k_means_features:
                ind = int(key[key.find('dist__') + 6: key.rfind('_')])
                x.loc[:, key] = np.sqrt(np.square(temp - centers.iloc[ind]).sum(axis=1))

        # Lagged features
        for k in lag_features:
            key, lag = k.split('__lag__')
            x.loc[:, k] = data[key].shift(-int(lag), fill_value=0)

        # Trigonometric features
        for k in trigonometric_features:
            func, key = k.split('__')
            x.loc[:, k] = getattr(np, func)(data[key])

        return x[features]

    def export_function(self):
        code = inspect.getsource(self.transform)
        code = code[code.find('\n')+1:]
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
        self.x = copy.copy(input_frame)
        self.originalInput = copy.copy(input_frame)
        self.y = output_frame.replace([np.inf, -np.inf], 0).fillna(0).reset_index(drop=True)

    def _calc_baseline(self):
        baseline = {}
        for key in self.originalInput.keys():
            baseline[key] = self._analyse_feature(self.x[key])
        self.baseline = pd.DataFrame(baseline).max(axis=1)

    def _analyse_feature(self, feature):
        feature = feature.clip(lower=1e-12, upper=1e12).fillna(0).values.reshape((-1, 1))
        m = copy.copy(self.model)
        m.fit(feature, self.y)
        if self.mode == 'multiclass':
            return [m.score(feature, self.y, self.y == i) for i in self.y.unique()]
        else:
            return [m.score(feature, self.y)]

    def _accept_feature(self, candidate):
        if any(candidate > self.baseline.values):
            return True
        else:
            return False

    @ staticmethod
    def _select_features(scores):
        # If scores is empty, return empty list
        if len(scores) == 0:
            return []

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
        if os.path.exists(self.folder + 'Colinear_v{}.json'.format(self.version)):
            print('[Features] Loading Colinear features')
            self.coLinearFeatures = json.load(open(self.folder + 'Colinear_v{}.json'.format(self.version), 'r'))

        # Else run
        else:
            print('[Features] Analysing colinearity')
            nk = len(self.x.keys())
            norm = (self.x - self.x.mean(skipna=True, numeric_only=True)).to_numpy()
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
            self.coLinearFeatures = self.x.keys()[np.sum(upper > self.informationThreshold, axis=0) > 0].to_list()
            json.dump(self.coLinearFeatures, open(self.folder + 'Colinear_v{}.json'.format(self.version), 'w'))

        self.originalInput = self.originalInput.drop(self.coLinearFeatures, axis=1)
        print('[Features] Removed {} Co-Linear features ({:.3f} %% threshold)'.format(
            len(self.coLinearFeatures), self.informationThreshold))

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
        scores = dict(process_map(functools.partial(self._static_multiply, self.model, self.x, self.y),
                                  features, max_workers=8, chunksize=min(100, int(len(features) / 8 / 8))))
        self.multiFeatures = self._select_features(scores)

    def _add_cross_features(self):
        """
        Calculates cross-feature features with m and multiplication.
        Should be limited to say ~500.000 features (runs about 100-150 features / second)
        """
        # Check if not already executed
        if os.path.exists(self.folder + 'crossFeatures_v{}.json'.format(self.version)) and False:
            self.crossFeatures = json.load(open(self.folder + 'crossFeatures_v{}.json'.format(self.version), 'r'))
            print('[Features] Loaded {} cross features'.format(len(self.crossFeatures)))

        # Else, execute
        else:
            print('[Features] Analysing cross features')
            scores = {}
            n_keys = len(self.originalInput.keys())
            start_time = time.time()

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
                    # Skip if we're out of time
                    if time.time() - start_time > self.timeout:
                        continue

                    # Analyse Division
                    feature = self.x[key_a] / self.x[key_b]
                    score = self._analyse_feature(feature)
                    # Accept or not
                    if self._accept_feature(score):
                        scores[key_a + '__d__' + key_b] = score
                        accepted_for_key_a += 1

                    # Multiplication i * j == j * i, so skip if j >= i
                    if j >= i:
                        continue

                    # Analyse Multiplication
                    feature = self.x[key_a] * self.x[key_b]
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
                feature = self.x[key_a] * self.x[key_b]
                self.x[k] = feature.clip(lower=1e-12, upper=1e12).fillna(0)
            else:
                key_a, key_b = k.split('__d__')
                feature = self.x[key_a] / self.x[key_b]
                self.x[k] = feature.clip(lower=1e-12, upper=1e12).fillna(0)

        # Store
        json.dump(self.crossFeatures, open(self.folder + 'crossFeatures_v{}.json'.format(self.version), 'w'))
        print('[Features] Added {} cross features'.format(len(self.crossFeatures)))

    def _add_trigonometry_features(self):
        """
        Calculates trigonometry features with sinus, cosines
        """
        # Check if not already executed
        if os.path.exists(self.folder + 'trigoFeatures_v{}.json'.format(self.version)):
            self.trigoFeatures = json.load(open(self.folder + 'trigoFeatures_v{}.json'.format(self.version), 'r'))
            print('[Features] Loaded {} trigonometric features'.format(len(self.trigoFeatures)))

        # Else, execute
        else:
            print('[Features] Analysing Trigonometric features')

            scores = {}
            for key in tqdm(self.originalInput.keys()):
                # Sinus feature
                sin_feature = np.sin(self.x[key])
                score = self._analyse_feature(sin_feature)
                if self._accept_feature(score):
                    scores['sin__' + key] = score

                # Co sinus feature
                cos_feature = np.cos(self.x[key])
                score = self._analyse_feature(cos_feature)
                if self._accept_feature(score):
                    scores['cos__' + key] = score

            # Select valuable features
            self.trigoFeatures = self._select_features(scores)

        # Add features
        for k in self.trigoFeatures:
            func, key = k.split('__')
            self.x[k] = getattr(np, func)(self.x[key])

        # Store
        json.dump(self.trigoFeatures, open(self.folder + 'trigoFeatures_v{}.json'.format(self.version), 'w'))
        print('[Features] Added {} trigonometric features'.format(len(self.trigoFeatures)))

    def _add_additive_features(self):
        """
        Calculates simple additive and subtractive features
        """
        # Load if available
        if os.path.exists(self.folder + 'addFeatures_v{}.json'.format(self.version)):
            self.addFeatures = json.load(open(self.folder + 'addFeatures_v{}.json'.format(self.version), 'r'))
            print('[Features] Loaded {} additive features'.format(len(self.addFeatures)))

        # Else, execute
        else:
            scores = {}
            start_time = time.time()
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
                    # Skip if we're out of time
                    if time.time() - start_time > self.timeout:
                        continue

                    # Subtracting feature
                    feature = self.x[key_a] - self.x[key_b]
                    score = self._analyse_feature(feature)
                    if self._accept_feature(score):
                        scores[key_a + '__sub__' + key_b] = score
                        accepted_for_key_a += 1

                    # A + B == B + A, so skip if i > j
                    if j > i:
                        continue

                    # Additive feature
                    feature = self.x[key_a] + self.x[key_b]
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
                feature = self.x[key_a] - self.x[key_b]
                self.x.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)
            else:
                key_a, key_b = key.split('__add__')
                feature = self.x[key_a] + self.x[key_b]
                self.x.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)

        # store
        json.dump(self.addFeatures, open(self.folder + 'addFeatures_v{}.json'.format(self.version), 'w'))
        print('[Features] Added {} additive features'.format(len(self.addFeatures)))

    def _add_k_means_features(self):
        """
        Analyses the correlation of k-means features.
        k-means is a clustering algorithm which clusters the data.
        The distance to each cluster is then analysed.
        """
        # Check if not exist
        if os.path.exists(self.folder + 'K-MeansFeatures_v{}.json'.format(self.version)):
            # Load features and cluster size
            self.kMeansFeatures = json.load(open(self.folder + 'K-MeansFeatures_v{}.json'.format(self.version), 'r'))
            k_means_data = pd.read_csv(self.folder + 'KMeans_v{}.csv'.format(self.version))
            print('[Features] Loaded {} K-Means features'.format(len(self.kMeansFeatures)))

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
                self.x[key] = np.sqrt(np.square(data - centers.iloc[ind]).sum(axis=1))

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
            k_means = MiniBatchKMeans(n_clusters=clusters)
            column_names = ['dist__{}_{}'.format(i, clusters) for i in range(clusters)]
            distances = pd.DataFrame(columns=column_names, data=k_means.fit_transform(data))
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
                self.x[k] = distances[k]

            # Create output
            centers = pd.DataFrame(columns=self.originalInput.keys(), data=k_means.cluster_centers_)
            centers = centers.append(means, ignore_index=True)
            centers = centers.append(stds, ignore_index=True)
            centers.to_csv(self.folder + 'KMeans_v{}.csv'.format(self.version), index=False)
            json.dump(self.kMeansFeatures, open(self.folder + 'K-MeansFeatures_v{}.json'.format(self.version), 'w'))
            print('[Features] Added {} K-Means features ({} clusters)'.format(len(self.kMeansFeatures), clusters))

    def _add_diff_features(self):
        """
        Analyses whether the diff signal of the data should be included.
        """

        # Check if we're allowed
        if self.maxDiff == 0:
            print('[Features] Diff features skipped, max diff = 0')
            return

        # Check if exist
        if os.path.exists(self.folder + 'diffFeatures_v{}.json'.format(self.version)):
            self.diffFeatures = json.load(open(self.folder + 'diffFeatures_v{}.json'.format(self.version), 'r'))
            print('[Features] Loaded {} diff features'.format(len(self.diffFeatures)))

        # If not exist, execute
        else:
            print('[Features] Analysing diff features')
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
                        scores[key + '__diff__{}'.format(diff)] = score

            # Select the valuable features
            self.diffFeatures = self._select_features(scores)
            print('[Features] Added {} differenced features'.format(len(self.diffFeatures)))

        # Add Diff Features
        for k in self.diffFeatures:
            key, diff = k.split('__diff__')
            feature = self.x[key]
            for i in range(1, int(diff)):
                feature = feature.diff().clip(lower=1e-12, upper=1e12).fillna(0)
            self.x[k] = feature
        json.dump(self.diffFeatures, open(self.folder + 'diffFeatures_v{}.json'.format(self.version), 'w'))

    def _add_lagged_features(self):
        """
        Analyses the correlation of lagged features (value of sensor_x at t-1 to predict target at t)
        """
        # Check if allowed
        if self.maxLags == 0:
            print('[Features] Lagged features skipped, max lags = 0')
            return

        # Check if exists
        if os.path.exists(self.folder + 'laggedFeatures_v{}.json'.format(self.version)):
            self.laggedFeatures = json.load(open(self.folder + 'laggedFeatures_v{}.json'.format(self.version), 'r'))
            print('[Features] Loaded {} lagged features'.format(len(self.laggedFeatures)))

        # Else execute
        else:
            print('[Features] Analysing lagged features')
            keys = self.originalInput.keys()
            scores = {}
            for lag in tqdm(range(1, self.maxLags)):
                for key in keys:
                    score = self._analyse_feature(self.x[key].shift(-1))
                    if self._accept_feature(score):
                        scores[key + '__lag__{}'.format(lag)] = score

            # Select
            self.laggedFeatures = self._select_features(scores)
            print('[Features] Added {} lagged features'.format(len(self.laggedFeatures)))

        # Add selected
        for k in self.laggedFeatures:
            key, lag = k.split('__lag__')
            self.x[k] = self.originalInput[key].shift(-int(lag), fill_value=0)

        # Store
        json.dump(self.laggedFeatures, open(self.folder + 'laggedFeatures_v{}.json'.format(self.version), 'w'))

    def _predictive_power_score(self):
        """
        Calculates the Predictive Power Score (https://github.com/8080labs/ppscore)
        Assymmetric correlation based on single decision trees trained on 5.000 samples with 4-Fold validation.
        """
        print('[Features] Determining features with PPS')
        data = self.x.copy()
        data['target'] = self.y.copy()
        pp_score = ppscore.predictors(data, "target")
        pp_cols = pp_score['x'][pp_score['ppscore'] != 0].to_list()
        print('[Features] Selected {} features with Predictive Power Score'.format(len(pp_cols)))
        return pp_cols

    def _random_forest_importance(self):
        """
        Calculates Feature Importance with Random Forest, aka Mean Decrease in Gini Impurity
        Symmetric correlation based on multiple features and multiple trees ensemble
        """
        print('[Features] Determining features with RF')
        if self.mode == 'regression':
            rf = RandomForestRegressor().fit(self.x, self.y)
        elif self.mode == 'classification' or self.mode == 'multiclass':
            rf = RandomForestClassifier().fit(self.x, self.y)
        else:
            raise ValueError('Method not implemented')
        fi = rf.feature_importances_
        sfi = fi.sum()
        ind = np.flip(np.argsort(fi))
        # Info Threshold
        ind_keep = [ind[i] for i in range(len(ind)) if fi[ind[:i]].sum() <= 0.85 * sfi]
        threshold = self.x.keys()[ind_keep].to_list()
        ind_keep = [ind[i] for i in range(len(ind)) if fi[i] > sfi / 200]
        increment = self.x.keys()[ind_keep].to_list()
        print('[Features] Selected {} features with 85% RF threshold'.format(len(threshold)))
        print('[Features] Selected {} features with 0.5% RF increment'.format(len(increment)))
        return threshold, increment

    def _borutapy(self):
        print('[Features] Determining features with Boruta')
        rf = None
        if self.mode == 'regression':
            rf = RandomForestRegressor()
        elif self.mode == 'classification' or self.mode == 'multiclass':
            rf = RandomForestClassifier()
        selector = BorutaPy(rf, n_estimators='auto', verbose=0)
        selector.fit(self.x.to_numpy(), self.y.to_numpy())
        bp_cols = self.x.keys()[selector.support_].to_list()
        print('[Features] Selected {} features with Boruta'.format(len(bp_cols)))
        return bp_cols
