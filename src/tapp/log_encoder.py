import math
import numpy as np
from tqdm import tqdm
from umap import UMAP
from abc import ABC, abstractmethod
from tapp.text_encoder import BERTbaseFineTunedNextActivityTextEncoder
from tapp.text_encoder import BERTbaseFineTunedNextTimeTextEncoder
from tapp.text_encoder import BERTbaseFineTunedNextActivityAndTimeTextEncoder
from tapp.text_encoder import BoWTextEncoder, BoNGTextEncoder, LDATextEncoder


class Encoder(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, docs):
        pass

    @abstractmethod
    def transform(self, docs):
        pass


class LogEncoder(Encoder):

    def __init__(self, text_encoder=None, advanced_time_attributes=True, text_base_for_training='event'):
        self.text_encoder = text_encoder
        self.activities = []
        self.data_attributes = []
        self.text_attribute = None
        self.categorical_attributes = []
        self.categorical_attributes_values = []
        self.numerical_attributes = []
        self.numerical_divisor = []
        self.event_dim = 0
        self.feature_dim = 0
        self.advanced_time_attributes = advanced_time_attributes
        self.time_scaling_divisor = [1, 1, 1]
        self.process_start_time = 0
        self.text_base_for_training = text_base_for_training
        super().__init__()

    def fit(self, log, activities=None, data_attributes=None, text_attribute=None):
        # Fit encoder to log
        self.activities = activities
        self.data_attributes = data_attributes
        self.text_attribute = text_attribute
        self.categorical_attributes = list(filter(lambda attribute: not _is_numerical_attribute(log, attribute), self.data_attributes))
        self.categorical_attributes_values = [_get_event_labels(log, attribute) for attribute in self.categorical_attributes]
        self.numerical_attributes = list(filter(lambda attribute: _is_numerical_attribute(log, attribute), self.data_attributes))
        self.numerical_divisor = [np.max([event[attribute].timestamp() for case in log for event in case]) for attribute in self.numerical_attributes]
        self.process_start_time = np.min([event["time:timestamp"].timestamp() for case in log for event in case])

        # Scaling divisors for time related features to achieve values between 0 and 1
        time_between_events_max = np.max(
            [event["time:timestamp"].timestamp() - case[event_index - 1]["time:timestamp"].timestamp() for case in
             log for event_index, event in enumerate(case) if event_index > 0])
        self.time_scaling_divisor = [time_between_events_max, 86400, 604800]  # 86400 = 24 * 60 * 60, 604800 = 7 * 24 * 60 * 60

        # Event dimension: Maximum number of events in a case
        self.event_dim = _get_max_case_length(log)

        # Feature dimension: Encoding size of an event
        activity_encoding_length = len(self.activities)
        # FIXME: appears to be a bug in the original code
        #time_encoding_length = 6 if self.advanced_time_attributes else 2
        time_encoding_length = 3 if self.advanced_time_attributes else 1
        categorical_attributes_encoding_length = sum([len(values) for values in self.categorical_attributes_values])
        numerical_attributes_encoding_length = len(self.numerical_attributes)
        text_encoding_length = self.text_encoder.encoding_length if self.text_encoder is not None and self.text_attribute is not None else 0
        if isinstance(self.text_encoder, BERTbaseFineTunedNextActivityAndTimeTextEncoder):
            self.feature_dim = self.feature_dim = activity_encoding_length + time_encoding_length + categorical_attributes_encoding_length + numerical_attributes_encoding_length + 2 * text_encoding_length
        else:
            self.feature_dim = self.feature_dim = activity_encoding_length + time_encoding_length + categorical_attributes_encoding_length + numerical_attributes_encoding_length + text_encoding_length
        print("Event encoding length:", self.feature_dim)
        print("feature_dim components: activity_encoding_length =", activity_encoding_length,
              ", time_encoding_length =", time_encoding_length,
              ", categorical_attributes_encoding_length =", categorical_attributes_encoding_length,
              ", numerical_attributes_encoding_length =", numerical_attributes_encoding_length,
              ", text_encoding_length =", text_encoding_length)
        print("categorical_attributes_values:", self.categorical_attributes_values)

        # Train text encoder
        if self.text_encoder is not None and self.text_attribute is not None:
            # collect texts / documents and labels for BERT fine-tuning (labels are indices, API: "should be in [0, ..., config.num_labels - 1]")
            #event_docs = [event[self.text_attribute] for case in log for event in case if self.text_attribute in event]
            event_docs = []
            for case in log:
                text = case.attributes.get(self.text_attribute, None)
                for event in case:
                    if self.text_attribute in event:
                        event_docs.append(event[self.text_attribute])
                    elif text is not None:
                        event_docs.append(text)
            print(f"Number of documents for text encoder training: {len(event_docs)}")
            print(f"Number of unique documents: {len(set(event_docs))}")
            print("Sample documents:", event_docs[:5])
            event_next_activities = []
            event_next_times = []

            prefix_docs = []
            prefix_next_activities = []
            prefix_next_times = []

            for case in log:
                for event_i, event in enumerate(case):
                    if self.text_attribute in event:
                        # next activity
                        next_event_i = event_i + 1
                        if next_event_i == len(case):
                            # case terminated
                            event_next_activities.append(len(self.activities))
                            event_next_times.append(0)
                        else:
                            # case ongoing
                            event_next_activities.append(self.activities.index(case[next_event_i]["concept:name"]))
                            event_next_times.append((case[next_event_i]["time:timestamp"].timestamp() -
                                                     case[event_i]["time:timestamp"].timestamp()) /
                                                    self.time_scaling_divisor[0])

                case_event_idx_docs = [(event_i, event[self.text_attribute]) for event_i, event in enumerate(case) if self.text_attribute in event]

                # skip case with only one document (already in event_docs)
                if len(case_event_idx_docs) <= 1:
                    continue
                else:
                    case_docs = ''
                    case_next_activities = []
                    case_next_times = []

                    # concat events' texts
                    for _, doc in case_event_idx_docs:
                        case_docs += ' ' + str(doc)

                    # prefix next activity
                    next_event_i = case_event_idx_docs[-1][0]
                    if next_event_i == len(case):
                        # case terminated
                        case_next_activities.append(len(self.activities))
                        case_next_times.append(0)
                    else:
                        # case ongoing
                        case_next_activities.append(self.activities.index(case[next_event_i]["concept:name"]))
                        case_next_times.append((case[next_event_i]["time:timestamp"].timestamp() - case[event_i][
                            "time:timestamp"].timestamp()) / self.time_scaling_divisor[0])

                    prefix_docs.append(case_docs)
                    prefix_next_activities.extend(case_next_activities)
                    prefix_next_times.extend(case_next_times)

            if self.text_base_for_training == 'event':
                docs = event_docs
                next_activities = event_next_activities
                next_times = event_next_times
            elif self.text_base_for_training == 'prefix':
                docs = event_docs + prefix_docs
                next_activities = event_next_activities + prefix_next_activities
                next_times = event_next_times + prefix_next_times

            print("self.text_encoder type:", type(self.text_encoder))
            # fine-tune BERT on next activity prediction (Sequence Classification)
            if type(self.text_encoder).__name__ == "BERTbaseFineTunedNextActivityTextEncoder":
                self.text_encoder.fit(docs, np.array(next_activities))

            # fine-tune BERT on next event time prediction (Sequence Regression)
            elif isinstance(self.text_encoder, BERTbaseFineTunedNextTimeTextEncoder):
                self.text_encoder.fit(docs, np.array(next_times))

            elif isinstance(self.text_encoder, BERTbaseFineTunedNextActivityAndTimeTextEncoder):
                self.text_encoder.fit(docs, np.array(next_activities))

            # other text model
            else:
                self.text_encoder.fit(docs)

    #TODO: i adjusted the offset for the 
    def transform(self, log, for_training=True):

        def _maybe_reduce(vecs, target_dim):
            """Reduce embedding dimensionality to target_dim using UMAP if possible.
            If there aren't enough samples, just truncate the vectors to the first target_dim dimensions.
            """
            print("vecs shape:", vecs.shape)
            print("target_dim:", target_dim)
            print("vecs sample:", vecs[:3])

            n_samples, n_features = vecs.shape

            # Only reduce if current dim > target_dim
            if n_features > target_dim:
                if n_samples < target_dim * 2:  # heuristic: not enough samples for UMAP
                    print(f"Not enough samples ({n_samples}) for UMAP; truncating to first {target_dim} dimensions.")
                    return vecs[:, :target_dim]
                else:
                    print(f"Reducing embedding dimension from {n_features} → {target_dim} using UMAP...")
                    reducer = UMAP(n_components=target_dim, random_state=42)
                    return reducer.fit_transform(vecs)

            # No reduction needed
            return vecs

        case_dim = np.sum([len(case) for case in log]) if for_training else len(log)

        # Prepare input and output vectors/matrices
        x = np.zeros((case_dim, self.event_dim, self.feature_dim))
        if for_training:
            y_next_act = np.zeros((case_dim, len(self.activities) + 1))
            y_next_time = np.zeros(case_dim)

        print("Encoding log with", len(log), "cases...")

        # Collect all unique text values for pre-encoding
        all_text_values = set()
        if self.text_encoder and self.text_attribute:
            for case in log:
                for event in case:
                    text_value = str(
                        event.get(self.text_attribute, case.attributes.get(self.text_attribute, ""))
                    ).strip()
                    all_text_values.add(text_value)

        # Encode all texts first (batched)
        text_cache = {}
        if self.text_encoder and self.text_attribute:
            all_text_values = list(all_text_values)

            if isinstance(self.text_encoder, BERTbaseFineTunedNextActivityAndTimeTextEncoder):
                vec_act_all, vec_time_all = self.text_encoder.transform(all_text_values)
            else:
                vec_all = self.text_encoder.transform(all_text_values)

            # Optionally apply dimension reduction
            if isinstance(self.text_encoder, BERTbaseFineTunedNextActivityAndTimeTextEncoder):
                vec_act_all = _maybe_reduce(vec_act_all, self.text_encoder.encoding_length)
                vec_time_all = _maybe_reduce(vec_time_all, self.text_encoder.encoding_length)
            else:
                vec_all = _maybe_reduce(vec_all, self.text_encoder.encoding_length)

            # Populate cache with final embeddings
            for i, text_value in enumerate(all_text_values):
                if isinstance(self.text_encoder, BERTbaseFineTunedNextActivityAndTimeTextEncoder):
                    text_cache[text_value] = (
                        np.expand_dims(vec_act_all[i], axis=0),
                        np.expand_dims(vec_time_all[i], axis=0)
                    )
                else:
                    text_cache[text_value] = (np.expand_dims(vec_all[i], axis=0),)

        # Encode traces and prefix traces
        trace_dim_index = 0
        for case in tqdm(log, desc="Encoding log", unit="case"):
            case_start_time = case[0]["time:timestamp"].timestamp()
            prefix_lengths = range(1, len(case) + 1) if for_training else range(len(case), len(case) + 1)

            for prefix_length in prefix_lengths:
                previous_event_time = case_start_time
                padding = self.event_dim - prefix_length

                for event_index, event in enumerate(case):
                    if event_index <= prefix_length - 1:
                        # --- Encode activity ---
                        if event["concept:name"] in self.activities:
                            x[trace_dim_index][padding + event_index][self.activities.index(event["concept:name"])] = 1
                        offset = len(self.activities)

                        # --- Encode time attributes ---
                        event_time = event["time:timestamp"]
                        x[trace_dim_index][padding + event_index][offset + 0] = (
                            event_time.timestamp() - previous_event_time
                        ) / self.time_scaling_divisor[0]
                        if self.advanced_time_attributes:
                            x[trace_dim_index][padding + event_index][offset + 1] = (
                                event_time.hour * 3600 + event_time.second
                            ) / self.time_scaling_divisor[1]
                            x[trace_dim_index][padding + event_index][offset + 2] = (
                                event_time.weekday() * 86400
                                + event_time.hour * 3600
                                + event_time.second
                            ) / self.time_scaling_divisor[2]
                            offset += 3
                        else:
                            offset += 1

                        previous_event_time = event_time.timestamp()

                        # --- Encode categorical attributes ---
                        for attribute_index, attribute in enumerate(self.categorical_attributes):
                            if event[attribute] in self.categorical_attributes_values[attribute_index]:
                                x[trace_dim_index][padding + event_index][
                                    offset + self.categorical_attributes_values[attribute_index].index(event[attribute])
                                ] = 1
                            offset += len(self.categorical_attributes_values[attribute_index])

                        # --- Encode numerical attributes ---
                        for attribute_index, attribute in enumerate(self.numerical_attributes):
                            x[trace_dim_index][padding + event_index][offset] = float(event[attribute]) / self.numerical_divisor[attribute_index]
                            offset += 1

                        # --- Encode textual attribute (using precomputed cache) ---
                        if self.text_encoder and self.text_attribute:
                            text_value = str(
                                event.get(self.text_attribute, case.attributes.get(self.text_attribute, ""))
                            ).strip()
                            cached = text_cache[text_value]

                            if isinstance(self.text_encoder, BERTbaseFineTunedNextActivityAndTimeTextEncoder):
                                vec_act, vec_time = cached
                                x[trace_dim_index][padding + event_index][offset:offset + len(vec_act[0])] = vec_act[0]
                                x[trace_dim_index][padding + event_index][
                                    offset + len(vec_act[0]):offset + len(vec_act[0]) + len(vec_time[0])
                                ] = vec_time[0]
                                offset += 2 * len(vec_act[0])
                            else:
                                vec = cached[0]
                                x[trace_dim_index][padding + event_index][offset:offset + len(vec[0])] = vec[0]
                                offset += len(vec[0])

                # --- Set target values ---
                if for_training:
                    if prefix_length == len(case):
                        y_next_act[trace_dim_index][len(self.activities)] = 1
                        y_next_time[trace_dim_index] = 0
                    else:
                        y_next_act[trace_dim_index][self.activities.index(case[prefix_length]["concept:name"])] = 1
                        y_next_time[trace_dim_index] = (
                            case[prefix_length]["time:timestamp"].timestamp()
                            - case[prefix_length - 1]["time:timestamp"].timestamp()
                        ) / self.time_scaling_divisor[0]

                trace_dim_index += 1

        if for_training:
            return x, y_next_act, y_next_time
        else:
            return x


    def transform_tree(self, x, k):
        if k > self.event_dim:
            k = self.event_dim

        X_tree = []
        for sample in x:
            X_tree.append(sample[-k:].flatten())
        X_tree = np.array(X_tree)
        return X_tree

    def get_feature_names(self, k=None):
        if k is None or k > self.event_dim:
            k = self.event_dim

        feature_names = []
        for event_i in range(k):
            event_i = event_i - k

            # Activities
            for act in self.activities:
                feature_names.append(f"event_{event_i}_activity_{act}")

            # Time features
            feature_names.append(f"event_{event_i}_time_since_prev_scaled")
            if self.advanced_time_attributes:
                feature_names.append(f"event_{event_i}_time_since_midnight_scaled")
                feature_names.append(f"event_{event_i}_time_since_monday_scaled")

            # Categorical attributes
            for attr, values in zip(self.categorical_attributes, self.categorical_attributes_values):
                for val in values:
                    feature_names.append(f"event_{event_i}_{attr}={val}")

            # Numerical attributes
            for attr in self.numerical_attributes:
                feature_names.append(f"event_{event_i}_{attr}_scaled")

            # Text attributes
            if self.text_encoder is not None and self.text_attribute is not None:
                encoder = self.text_encoder

                # Special case: dual-head BERT
                if isinstance(encoder, BERTbaseFineTunedNextActivityAndTimeTextEncoder):
                    for j in range(encoder.encoding_length):
                        feature_names.append(f"event_{event_i}_{encoder.name}_activity_{j}")
                    for j in range(encoder.encoding_length):
                        feature_names.append(f"event_{event_i}_{encoder.name}_time_{j}")

                # BoW or BoNG → use vocab words
                elif isinstance(encoder, (BoWTextEncoder, BoNGTextEncoder)):
                    if encoder.vectorizer is None:
                        raise ValueError(f"Vectorizer for {encoder.name} has not been fit yet.")
                    vocab = encoder.vectorizer.get_feature_names_out()
                    for word in vocab:
                        feature_names.append(f"event_{event_i}_{encoder.name}_{word}")

                # LDA → topic names
                elif isinstance(encoder, LDATextEncoder):
                    if encoder.model is None:
                        raise ValueError(f"LDA model for {encoder.name} has not been fit yet.")
                    for i in range(encoder.num_topics):
                        feature_names.append(f"event_{event_i}_{encoder.name}_topic_{i}")

                # Generic encoder fallback
                else:
                    for j in range(encoder.encoding_length):
                        feature_names.append(f"event_{event_i}_{encoder.name}_{j}")

        return feature_names



def _get_event_labels(log, attribute_name):
    if not log:
        return []

    seen = set()
    unique_values = []
    for case in log:
        for event in case:
            val = event[attribute_name]
            if val is None or (isinstance(val, float) and math.isnan(val)):
                val = '__NaN__'  # sentinel for missing values
            if val not in seen:
                seen.add(val)
                unique_values.append(val)
    return unique_values
    


def _is_numerical(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def _is_numerical_attribute(log, attribute):
    first_case = log[0]
    first_event = first_case[0]

    if attribute in first_event:
        return _is_numerical(first_event[attribute])
    elif attribute in first_case.attributes:
        return _is_numerical(first_case.attributes[attribute])
    else:
        return False


def _get_max_case_length(log):
    return max([len(case) for case in log]) if log else 0
