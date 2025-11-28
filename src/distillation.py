import os
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tapp.log_encoder import LogEncoder
from tapp.text_encoder import (
    BoWTextEncoder,
    BoNGTextEncoder,
    LDATextEncoder,
)
from tapp.tapp_model import _get_event_labels
import numpy as np
import pickle
import re
import pandas as pd
from sklearn.tree import export_text, _tree
from tqdm.notebook import tqdm


def get_distillation_paths(folder_name, text_encoder, data_attributes, text_attribute, run_id):
    data_attributes = ",".join(sorted(data_attributes)) if data_attributes else "None"
    distillation_folder = os.path.join("data", "distillation", folder_name)
    os.makedirs(distillation_folder, exist_ok=True)
    encoder_name = text_encoder.name if text_encoder else "None"
    encoding_length = text_encoder.encoding_length if text_encoder else "None"
    distillation_file_train = f"y_{encoder_name}_{encoding_length}_{data_attributes}_{text_attribute}_{run_id}_train.npy"
    distillation_file_test = f"y_{encoder_name}_{encoding_length}_{data_attributes}_{text_attribute}_{run_id}_test.npy"
    distillation_path_train = os.path.join(distillation_folder, distillation_file_train)
    distillation_path_test = os.path.join(distillation_folder, distillation_file_test)
    return distillation_path_train, distillation_path_test

def get_evaluation_paths(folder_name, version, model_names_subset, data_attributes, text_attribute, run_id, model_type="dt"):
    distillation_folder = os.path.join("data", "distillation", "evaluation", folder_name)
    data_attributes = ",".join(sorted(data_attributes)) if data_attributes else "None"
    os.makedirs(distillation_folder, exist_ok=True)
    base_name = f"{model_type}_{version}_{'_'.join(model_names_subset)}_{data_attributes}_{text_attribute}_{run_id}"
    tree_model_file = f"model_{base_name}.pkl"
    tree_string_file = f"tree_str_{base_name}.txt"
    features_file = f"features_{base_name}.pkl"
    y_file = f"y_{base_name}.npy"
    return (
        os.path.join(distillation_folder, tree_model_file),
        os.path.join(distillation_folder, tree_string_file),
        os.path.join(distillation_folder, features_file),
        os.path.join(distillation_folder, y_file),
    )

def evaluate_distillation(
    model, X_train, X_test, y_train, y_test, y_test_distilled, description="Evaluation", output=True, y_save_path=None, model_save_path=None, start_alpha=0.0, end_alpha=1.0, skip_pruning=False
):
    best_alpha = None
    # --- 1. Split off 20% validation data ---
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False
    )

    print("skip_pruning", skip_pruning)
    # --- 2. If model is a DecisionTree, perform CCP pruning search ---
    if (isinstance(model, DecisionTreeClassifier) or isinstance(model, DecisionTreeRegressor)) and not skip_pruning:

        # get effective alphas
        path = model.cost_complexity_pruning_path(X_train_sub, y_train_sub)
        print("Ccp alpha paths:", len(path.ccp_alphas))
        ccp_alphas = path.ccp_alphas
        ccp_alphas = ccp_alphas[ccp_alphas >= start_alpha]
        ccp_alphas = ccp_alphas[ccp_alphas <= end_alpha]
        print(f"Ccp alpha paths, after filtering:", len(ccp_alphas))

        best_alpha = None
        best_val_acc = -1

        # Try each alpha, with tqdm progress bar
        for ccp in tqdm(ccp_alphas, desc="CCP pruning search"):
            clf = model.__class__(
                random_state=model.random_state,
                criterion=model.criterion,
                splitter=model.splitter,
                max_depth=model.max_depth,
                ccp_alpha=ccp
            )
            clf.fit(X_train_sub, y_train_sub)
            y_val_pred = clf.predict(X_val)
            if y_val_pred.ndim == 2:
                y_val_pred = y_val_pred.argmax(axis=1)
                y_val_true = y_val.argmax(axis=1)
            else:
                y_val_true = y_val
            val_acc = accuracy_score(y_val_true, y_val_pred)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_alpha = ccp

        # retrain final model with best alpha
        model = model.__class__(
            random_state=model.random_state,
            criterion=model.criterion,
            splitter=model.splitter,
            max_depth=model.max_depth,
            ccp_alpha=best_alpha
        )

    # --- 3. Train final model on full 80% training subset ---
    model.fit(X_train_sub, y_train_sub)

    # --- 4. Evaluate on test data ---
    y_pred = model.predict(X_test)
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    con_acc = accuracy_score(y_test_distilled, y_pred)
    con_f1 = f1_score(y_test_distilled, y_pred, average="weighted")
    num_nodes = model.tree_.node_count
    max_depth = model.tree_.max_depth
    decision_paths = model.decision_path(X_test)  # sparse matrix
    path_lengths = np.array(decision_paths.sum(axis=1)).flatten() - 1
    avg_path_length = path_lengths.mean()

    # --- 5. Output results ---
    if output:
        print(
            f"{description}: acc - {acc:.4f}, f1 - {f1:.4f}, "
            f"con_acc - {con_acc:.4f}, con_f1 - {con_f1:.4f}"
        )
        if hasattr(model, "tree_"):
            print("Selected ccp_alpha:", best_alpha)
            print("Number of nodes:", num_nodes)
            print("Max depth:", max_depth)
        
    if y_save_path:
        np.save(y_save_path, y_pred)
        print(f"Saved output to {model_save_path}")
    if model_save_path:
        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved model to {model_save_path}")

    return {"accuracy": acc, "f1_score": f1, "con_accuracy": con_acc, "con_f1_score": con_f1, "num_nodes": num_nodes, "max_depth": max_depth, "ccp_alpha": best_alpha, "avg_path_length": avg_path_length}

def evaluate_distillation_old(
    model, X_train, X_test, y_train, y_test, y_test_distilled, description="Evaluation", output=True
):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    con_acc = accuracy_score(y_test_distilled, y_pred)
    con_f1 = f1_score(y_test_distilled, y_pred, average="weighted")
    if output:
        print(
            f"{description}: acc - {acc:.4f}, f1 - {f1:.4f}, con_acc - {con_acc:.4f}, con_f1 - {con_f1:.4f}"
        )
        if hasattr(model, "tree_"):
            print("Number of nodes:", model.tree_.node_count)
            print("Max depth:", model.tree_.max_depth)
    return acc, f1, con_acc, con_f1

def save_evaluation_results(output_path="data/distillation/evaluation_results.csv", **kwargs):
    # Handle model_names specially if it's a list
    if "model_names" in kwargs and isinstance(kwargs["model_names"], list):
        kwargs["model_names"] = ",".join(kwargs["model_names"])
    if "data_attributes" in kwargs and isinstance(kwargs["data_attributes"], list):
        kwargs["data_attributes"] = ",".join(kwargs["data_attributes"])

    df = pd.DataFrame([kwargs])

    # Append to CSV if it exists, otherwise create a new one
    mode = 'a' if os.path.exists(output_path) else 'w'
    header = not os.path.exists(output_path)
    df.to_csv(output_path, mode=mode, header=header, index=False)

def analyze_text_splits(tree, feature_names):
    tree_ = tree.tree_
    split_features = tree_.feature  # indices of features used at split nodes

    text_splits = []
    for idx in split_features:
        if idx >= 0:  # -2 means leaf node
            fname = feature_names[idx]
            if "text" in fname:
                text_splits.append(fname)

    count_total = len(text_splits)
    count_per_feature = Counter(text_splits)

    return count_total, dict(count_per_feature)


def prepare_text_feature_datasets(
    text_models,
    log,
    complete_log,
    k=3,
    advanced_time_attributes=True,
    text_base_for_training="event",
    data_attributes=["age"],
    text_attribute="question",
):
    datasets = {}
    activities = _get_event_labels(complete_log, "concept:name")

    # Baseline without text model (keep all features)
    baseline_encoder = LogEncoder(
        text_encoder=None,
        advanced_time_attributes=advanced_time_attributes,
        text_base_for_training=text_base_for_training,
    )
    baseline_encoder.fit(
        complete_log,
        activities=activities,
        data_attributes=data_attributes,
        text_attribute=text_attribute,
    )
    X, y, _ = baseline_encoder.transform(log, for_training=True)
    X = baseline_encoder.transform_tree(X, k=k)
    feature_names_base = baseline_encoder.get_feature_names(k=k)
    datasets["None"] = {"X": X, "features": feature_names_base}
    datasets["y"] = y
    print(
        f"Baseline features: {len(feature_names_base)}, shape: {X.shape}, shape y: {y.shape}"
    )

    # Process each text model
    for text_model in text_models:
        log_encoder = LogEncoder(
            text_encoder=text_model,
            advanced_time_attributes=advanced_time_attributes,
            text_base_for_training=text_base_for_training,
        )
        log_encoder.fit(
            complete_log,
            activities=activities,
            data_attributes=data_attributes,
            text_attribute=text_attribute,
        )
        X, y, _ = log_encoder.transform(log, for_training=True)
        X = log_encoder.transform_tree(X, k=k)
        feature_names = log_encoder.get_feature_names(k=k)

        # Filter only "text" features
        text_indices = [i for i, name in enumerate(feature_names) if "BoW" in name or "BoNG" in name or "LDA" in name]
        X_text = (
            X[:, text_indices] if len(text_indices) > 0 else np.empty((X.shape[0], 0))
        )
        feature_names_text = [
            name
            for i, name in enumerate(feature_names)
            if i in text_indices
        ]

        datasets[text_model.name] = {
            "X": X_text,
            "features": feature_names_text,
        }
        print(
            f"Text model: {text_model.name}, features: {len(feature_names_text)}, shape: {X_text.shape}, feature names: {feature_names_text}"
        )

    return datasets

def concatenate_text_feature_datasets(datasets, model_names):
    # Start with baseline features
    X_base = datasets["None"]["X"]
    feature_names = list(datasets["None"]["features"])
    
    X_list = [X_base]
    
    # Add features from the requested models
    for model_name in model_names:
        if model_name not in datasets:
            raise ValueError(f"Model '{model_name}' not found in datasets")
        
        X_model = datasets[model_name]["X"]
        feats_model = datasets[model_name]["features"]
        
        X_list.append(X_model)
        feature_names.extend(feats_model)
    
    # Concatenate horizontally
    X_concat = np.concatenate(X_list, axis=1)
    
    return X_concat, feature_names


def get_feature_datasets(folder_name, text_models, log, train_log, test_log, k, data_attributes, text_attribute, force_recompute=False):
    # create or load the datasets
    train_dataset_path = os.path.join("data", "distillation", folder_name, f"train_dataset_{k}.pkl")
    test_dataset_path = os.path.join("data", "distillation", folder_name, f"test_dataset_{k}.pkl")
    if force_recompute or (not os.path.exists(train_dataset_path) or not os.path.exists(test_dataset_path)):
        train_dataset = prepare_text_feature_datasets(text_models, train_log, log, k=k, data_attributes=data_attributes, text_attribute=text_attribute)
        test_dataset = prepare_text_feature_datasets(text_models, test_log, log, k=k, data_attributes=data_attributes, text_attribute=text_attribute)
        with open(train_dataset_path, "wb") as f:
            pickle.dump(train_dataset, f)
        with open(test_dataset_path, "wb") as f:
            pickle.dump(test_dataset, f)
    else:
        with open(train_dataset_path, "rb") as f:
            train_dataset = pickle.load(f)
        with open(test_dataset_path, "rb") as f:
            test_dataset = pickle.load(f)
    return train_dataset, test_dataset


def tree_to_str_1(dt, feature_names=None, class_names=None):
    tree_str = export_text(dt, feature_names=feature_names, max_depth=dt.tree_.max_depth)
    if class_names is not None:
        for i, name in enumerate(class_names):
            # Use regex to replace only exact "class: <number>"
            tree_str = re.sub(rf"class:\s*{i}\b", f"class: {name}", tree_str)
    return tree_str


def tree_to_str_2(dt, feature_names=None, class_names=None):
    tree_ = dt.tree_
    lines = []

    def recurse(node_id):
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED:  # Internal node
            feature = feature_names[tree_.feature[node_id]] if feature_names else f"feature_{tree_.feature[node_id]}"
            threshold = tree_.threshold[node_id]
            left = tree_.children_left[node_id]
            right = tree_.children_right[node_id]
            lines.append(f"Node {node_id}: {feature} > {threshold:.4f}? T: Node {left}, F: Node {right}")
            recurse(left)
            recurse(right)
        else:  # Leaf node
            values = tree_.value[node_id][0]
            predicted_class_idx = int(values.argmax())
            predicted_class = class_names[predicted_class_idx] if class_names is not None else str(predicted_class_idx)
            lines.append(f"Node {node_id} (Leaf): Predict={predicted_class}")

    recurse(0)
    return "\n".join(lines)
    

def tree_to_str(dt, feature_names=None, class_names=None):
    if feature_names is not None:
        feature_names = [name.replace("_", " ") for name in feature_names]
    tree_ = dt.tree_
    lines = []

    def recurse(node_id):
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED:  # Internal node
            feature = feature_names[tree_.feature[node_id]] if feature_names else f"feature_{tree_.feature[node_id]}"
            threshold = tree_.threshold[node_id]
            left = tree_.children_left[node_id]
            right = tree_.children_right[node_id]
            lines.append(f"{node_id}: {feature} > {threshold:.4f} T: {left}, F: {right}")
            recurse(left)
            recurse(right)
        else:  # Leaf node
            values = tree_.value[node_id][0]
            predicted_class_idx = int(values.argmax())
            predicted_class = class_names[predicted_class_idx] if class_names is not None else str(predicted_class_idx)
            lines.append(f"{node_id}: predict {predicted_class}")

    recurse(0)
    return "\n".join(lines)


def explain_sample(X_sample, feature_names):
    if feature_names is not None:
        feature_names = [name.replace("_", " ") for name in feature_names]
    explanations = []
    for name, value in zip(feature_names, X_sample):
        if value != 0:
            # integer formatting if possible, otherwise float
            if float(value).is_integer():
                explanations.append(f"{name} = {int(value)}")
            else:
                explanations.append(f"{name} = {value:.4f}")
    return ", ".join(explanations) if explanations else "All features are zero"


def get_decision_path(tree, X, feature_names=None, class_names=None, show_distribution=False):
    tree_ = tree.tree_

    if X.ndim == 1:
        X = X.reshape(1, -1)

    node_indicator = tree.decision_path(X)
    leaf_id = tree.apply(X)

    feature_name = [
        feature_names[i] if feature_names is not None else f"feature_{i}"
        for i in tree_.feature
    ]

    sample_id = 0  # since we only handle one sample
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
    ]

    def format_distribution(values, node_id):
        classes_in_tree = tree.classes_
        weighted = values[0]
        total = tree_.n_node_samples[node_id]
        if weighted.sum() > 0:
            counts = (weighted / weighted.sum()) * total
        else:
            counts = weighted

        counts = counts.astype(int)

        # Use the class indices actually present in the tree
        if class_names is not None:
            return "{" + ", ".join(
                f"{class_names[cls]}: {cnt}" for cls, cnt in zip(classes_in_tree, counts)
            ) + "}"
        else:
            return "{" + ", ".join(
                f"{cls}: {cnt}" for cls, cnt in zip(classes_in_tree, counts)
            ) + "}"



    path = []

    for node_id in node_index:
        if leaf_id[sample_id] == node_id:
            # reached leaf
            value = tree_.value[node_id]
            if class_names is not None and tree_.n_outputs == 1:
                predicted_class = np.argmax(value)
                class_idx = tree.classes_[predicted_class]
                prediction = class_names[class_idx]
            else:
                prediction = value

            msg = f"--> Reached leaf node {node_id} with prediction: {prediction}"
            if show_distribution:
                msg += f" | class distribution = {format_distribution(value, node_id)}"
            path.append(msg)

        else:
            feature = feature_name[node_id]
            threshold = tree_.threshold[node_id]
            feature_value = X[sample_id, tree_.feature[node_id]]

            if feature_value <= threshold:
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            msg = (
                f"Node {node_id}: ({feature} = {feature_value:.3f} {threshold_sign} {threshold:.3f})"
            )

            if show_distribution:
                msg += f" | class distribution = {format_distribution(tree_.value[node_id], node_id)}"

            path.append(msg)

    return "\n".join(path)
