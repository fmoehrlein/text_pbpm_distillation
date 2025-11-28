import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from cnn_model import EnhancedCNNModel
from logger import setup_logger
from data_processing import split_train_val_test_tuple

log = setup_logger()


def convert_to_datetime(df, column):
    df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)
    # Remove timezone if present
    if df[column].dt.tz is not None:
        df[column] = df[column].dt.tz_convert(None)
    # Normalize to midnight
    df[column] = df[column].dt.normalize()


def extract_X_y(df, num_articles, sent_window_size=False, close_column="Close", mode="regression"):
    # Get news embeddings (assumed to be arrays)
    news_cols = [f"News {i+1}" for i in range(num_articles)]
    news_embeddings = df[news_cols].values  # shape: (N, num_articles)

    # Expand into (N, num_articles, embedding_dim)
    news_tensor = np.stack([[np.array(x) for x in row] for row in news_embeddings])

    # Previous 10 close values
    close_cols = [f"Close_{i}" for i in range(10)]
    if sent_window_size:
        close_cols.append("FinBERT score")
    X_attributes = df[close_cols].values.astype(np.float32)

    # Target values
    if mode == "trend":
        current_close = df[close_column].values.astype(np.float32)
        next_close = df[close_column].shift(-1).values.astype(np.float32)
        y = np.array([
            [1, 0] if next > current else [0, 1]
            for current, next in zip(current_close, next_close)
        ])
        y = y[:-1]               # Last row will have NaN in next_close
        X_attributes = X_attributes[:-1]
        news_tensor = news_tensor[:-1]
    else:
        y = df[close_column].values.astype(np.float32)

    # Build input list for model
    X = [np.array([]), X_attributes, news_tensor]

    return X, y


def build_news_csv(
    idx_df: pd.DataFrame, sub_data_dir: str, base_col: str = "News ", date_column="Date"
) -> pd.DataFrame:
    """Create a news dataframe with dynamic News n columns."""
    idx_df[date_column] = pd.to_datetime(idx_df[date_column])

    # will accumulate rows as dicts
    rows = []
    max_cols = 0  # track highest News n seen so far

    for _, row in idx_df.iterrows():
        date = row[date_column]
        inner_file = sub_data_dir / row["filename"]
        titles_df = pd.read_csv(inner_file)  # this file must have a column 'Titles'
        titles = titles_df["Titles"].astype(str).tolist()

        # update max_cols if we encounter more titles than before, but ensure less than 10
        if len(titles) > 10:
            log.debug(f"More than 10 titles found for {date}, truncating to 10.")
            titles = titles[:10]
        max_cols = max(max_cols, len(titles))

        rows.append({date_column: date, "titles": titles})

    # Build final dataframe with flexible News columns
    news_records = []
    for r in rows:
        record = {date_column: r[date_column]}
        # fill up to current max_cols
        for i in range(max_cols):
            record[f"{base_col}{i+1}"] = r["titles"][i] if i < len(r["titles"]) else "0"
        news_records.append(record)

    news_df = pd.DataFrame(news_records).sort_values(date_column).reset_index(drop=True)
    log.debug(
        f"Created news DataFrame with {len(news_df)} rows and {len(news_df.columns)} columns."
    )
    log.debug(f"Columns: {news_df.columns.tolist()}")
    log.debug(f"First few rows:\n{news_df.head()}")
    return news_df


# TODO: the minmax scaling should be done on the training set only, should be fine because min and max appear in the training set
def create_merged_stock_df(
    news_df, stocks_df, date_column="Date", close_column="Close"
):
    # --- Ensure datetime format and sort ---
    convert_to_datetime(news_df, date_column)
    convert_to_datetime(stocks_df, date_column)
    news_df = news_df.sort_values(date_column).reset_index(drop=True)
    stocks_df = stocks_df.sort_values(date_column).reset_index(drop=True)

    # --- Scale 'Close' ---
    scaler = MinMaxScaler()
    stocks_df[close_column] = scaler.fit_transform(stocks_df[[close_column]])

    # --- Create 10 shifted (previous day) scaled close columns ---
    window_size = 10
    for i in range(window_size):
        stocks_df[f"Close_{i}"] = stocks_df[close_column].shift(i + 1)

    # --- Prepare DataFrame for merge_asof ---
    rolling_cols = [f"Close_{i}" for i in range(window_size)]
    stocks_for_merge = stocks_df[[date_column] + rolling_cols]

    # --- Merge past 10 days' scaled closes (asof = most recent before news) ---
    merged_df = pd.merge_asof(
        news_df.sort_values(date_column),
        stocks_for_merge.sort_values(date_column),
        on=date_column,
        direction="backward",
    )

    # --- Drop rows that don't have a full 10-day history ---
    merged_df = merged_df.dropna(subset=rolling_cols).reset_index(drop=True)

    # --- Add same-day unscaled Close price ---
    same_day_close = stocks_df[[date_column, close_column]]
    merged_df = pd.merge(merged_df, same_day_close, on=date_column, how="left")

    # --- Load sentence-transformer model ---
    model_name = "intfloat/e5-large-v2"
    model = SentenceTransformer(model_name)

    # --- Find News_n columns to encode ---
    news_columns = [col for col in merged_df.columns if col.startswith("News")]

    # --- Encode each News_n column ---
    all_texts = []
    valid_masks = []
    for col in news_columns:
        values = merged_df[col].astype(str)
        mask = values.str.strip() != "0"
        valid_masks.append(mask.tolist())
        all_texts.extend(["passage: " + v for v in values[mask]])
    embeddings = model.encode(
        all_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32,
        device="cuda",
    )
    dim = (
        embeddings.shape[1]
        if len(embeddings)
        else model.get_sentence_embedding_dimension()
    )
    zero_vec = np.zeros(dim)
    it = iter(embeddings)
    for col, mask in zip(news_columns, valid_masks):
        merged_df[col] = [next(it) if valid else zero_vec for valid in mask]

    return merged_df


def evaluate_stock_model(
    X,
    y,
    hidden_units=(128, 64, 32),
    hidden_units_news=(128,),
    cnn_layers=[(256, 1), (128, 3)],
    pooling_layers=[1, 2],
    learning_rate=0.001,
    epochs=100,
    patience=5,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    mode="regression",
):
    assert len(X[2]) == len(y), "Mismatch between article matrices and targets"

    (
        X_train_act,
        X_train_attr,
        X_train_art,
        y_train,
        X_val_a,
        X_val_attr,
        X_val_art,
        y_val,
        X_test_a,
        X_test_attr,
        X_test_art,
        y_test,
    ) = split_train_val_test_tuple(
        X,
        y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    activity_dim = X_train_act.shape[1] if len(X_train_act) > 0 else 0
    attr_dim = X_train_attr.shape[1] if len(X_train_attr) > 0 else 0
    num_articles = len(X_train_art[0]) if len(X_train_art) > 0 else 0
    emb_dim = X_train_art[0].shape[1] if num_articles > 0 else 0
    log.debug(
        f"Activity dim: {activity_dim}, Attr dim: {attr_dim}, Num articles: {num_articles}, Embedding dim: {emb_dim}"
    )

    model_name = f"remaining_time_stocks_cnn"
    model = EnhancedCNNModel(
        activity_dim=activity_dim,
        attr_dim=attr_dim,
        num_articles=num_articles,
        emb_dim=emb_dim,
        hidden_units=hidden_units,
        hidden_units_news=hidden_units_news,
        cnn_layers=cnn_layers,
        pooling_layers=pooling_layers,
        learning_rate=learning_rate,
        model_name=model_name,
        mode=mode,
    )

    model.train(
        (X_train_act, X_train_attr, X_train_art),
        y_train,
        (X_val_a, X_val_attr, X_val_art),
        y_val,
        X_test=(X_test_a, X_test_attr, X_test_art),
        y_test=y_test,
        patience=patience,
        batch_size=batch_size,
    )

    mae, mse, r2 = model.evaluate((X_test_a, X_test_attr, X_test_art), y_test)
    log.info(
        f"Evaluation for {model_name}: MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}"
    )

    # model.plot_training_history()
    return mae, mse, r2


def run_stocks_experiments(
    df,
    num_articles_list,
    hidden_units=(128, 64, 32),
    hidden_units_news=(128,),
    cnn_layers=[(256, 1), (128, 3)],
    pooling_layers=[1, 2],
    learning_rate=0.001,
    patience=5,
    batch_size=32,
    epochs=100,
    k=5,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    close_column="Close",
    sent_window_size=False,
    mode="regression",
):
    results = []
    run_id = 0

    for num_articles in num_articles_list:
        print(f"\nEvaluating with num_articles = {num_articles}")
        for i in range(k):
            run_id += 1

            # Prepare data
            X, y = extract_X_y(
                df,
                num_articles=num_articles,
                close_column=close_column,
                sent_window_size=sent_window_size,
                mode=mode,
            )

            # Evaluate model
            mae, mse, r2 = evaluate_stock_model(
                X,
                y,
                hidden_units=hidden_units,
                hidden_units_news=hidden_units_news,
                cnn_layers=cnn_layers,
                pooling_layers=pooling_layers,
                learning_rate=learning_rate,
                patience=patience,
                epochs=epochs,
                batch_size=batch_size,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                mode=mode,
            )

            # Store metrics
            results.append(
                {
                    "id": run_id,
                    "num_articles": num_articles,
                    "mae": mae,
                    "mse": mse,
                    "r2": r2,
                }
            )

    # Create DataFrame
    results_df = pd.DataFrame(results)

    return results_df

from datetime import timedelta

def enrich_stock_with_sentiment(
    stocks_df: pd.DataFrame,
    news_df: pd.DataFrame,
    window_sizes: list[int],
    stock_date_col: str = "date",
    news_date_col: str = "pub_date",
    sentiment_col: str = "sent"
) -> pd.DataFrame:
    # Copy to avoid mutation
    stocks_df = stocks_df.copy()
    news_df = news_df.copy()

    convert_to_datetime(stocks_df, stock_date_col)
    convert_to_datetime(news_df, news_date_col)
    stocks_df[stock_date_col] = stocks_df[stock_date_col].dt.date
    news_df[news_date_col] = news_df[news_date_col].dt.date

    # For performance, group news by date
    news_by_date = news_df.groupby(news_date_col)[sentiment_col].apply(list).to_dict()

    # Precompute all windowed sentiments
    for window in window_sizes:
        sent_scores = []

        for current_date in stocks_df[stock_date_col]:
            # Define window range
            window_start = current_date - timedelta(days=window - 1)
            window_dates = [window_start + timedelta(days=i) for i in range(window)]

            # Collect all sentiments in the window
            sentiments = []
            for d in window_dates:
                sentiments.extend(news_by_date.get(d, []))

            # Compute mean or NaN if no data
            avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0
            sent_scores.append(avg_sent)

        # Assign to new column
        stocks_df[f"sent_{window}"] = sent_scores

    return stocks_df

def prepare_stock_datasets(
    stocks_df: pd.DataFrame,
    window_size: int = 10,
    sent_window_size: int = None,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    target_column: str = "Close",
):
    """
    Splits data, scales features, and builds TF datasets.

    Returns:
        Tuple[train_ds, val_ds, test_ds]
    """
    features = [target_column]
    if sent_window_size is not None:
        sent_col = f"sent_{sent_window_size}"
        if sent_col not in stocks_df.columns:
            raise ValueError(f"Missing sentiment column: {sent_col}")
        features.append(sent_col)

    stocks_df = stocks_df.copy()
    df = stocks_df[features].dropna().astype(np.float32)
    
    # Create target column (next-day close)
    df["target"] = df[target_column].shift(-1)
    df = df.dropna()

    total_len = len(df)
    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    # Fit scaler on train, apply to all
    scaler = MinMaxScaler()
    df_train_scaled = scaler.fit_transform(df_train[features])
    df_val_scaled = scaler.transform(df_val[features])
    df_test_scaled = scaler.transform(df_test[features])

    train_targets = df_train["target"].values
    val_targets = df_val["target"].values
    test_targets = df_test["target"].values

    def make_ds(data, targets):
        min_len = min(len(data), len(targets))
        return tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data[:min_len],
            targets=targets[:min_len],
            sequence_length=window_size,
            sampling_rate=1,
            shuffle=True,
            batch_size=batch_size,
        )

    train_ds = make_ds(df_train_scaled, train_targets)
    val_ds = make_ds(df_val_scaled, val_targets)
    test_ds = make_ds(df_test_scaled, test_targets)

    return train_ds, val_ds, test_ds, scaler

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, List, Tuple


def prepare_stock_datasets_flexible(
    stocks_df: pd.DataFrame,
    news_df: Optional[pd.DataFrame] = None,
    window_size: int = 10,
    sent_window_size: Optional[int] = None,
    text_window_size: Optional[int] = None,
    pca_dim: Optional[int] = None,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    target_column: str = "Close",
):
    """
    Prepares train/val/test datasets for late fusion model with separate stock and text inputs.
    Returns:
        train_ds, val_ds, test_ds, scaler, pca (if used)
    """
    embedding_cols = [c for c in news_df.columns if c.startswith("emb_")]
    use_sentiment = sent_window_size is not None
    use_embeddings = text_window_size is not None

    stocks_df = stocks_df.copy()
    convert_to_datetime(stocks_df, "Date")
    stocks_df = stocks_df.sort_values("Date").reset_index(drop=True)

    if news_df is not None and use_embeddings:
        news_df = news_df.copy()
        news_df["Date"] = news_df["pub_date"]
        convert_to_datetime(news_df, "Date")
        news_df = news_df.sort_values("Date").reset_index(drop=True)

    # Prepare embedding values aligned to each stock date (one embedding vector per stock date)
    if use_embeddings:
        embed_values = []
        embed_dim = len(embedding_cols)

        for date in stocks_df["Date"]:
            start_date = date - pd.Timedelta(days=text_window_size)
            subset = news_df[(news_df["Date"] >= start_date) & (news_df["Date"] <= date)]

            if not subset.empty:
                latest_embed = subset.iloc[-1][embedding_cols].values
            else:
                latest_embed = np.zeros(embed_dim)

            embed_values.append(latest_embed)

        embed_df = pd.DataFrame(embed_values, columns=embedding_cols)
        stocks_df = pd.concat([stocks_df.reset_index(drop=True), embed_df.reset_index(drop=True)], axis=1)

    # Target (next-day close)
    stocks_df["target"] = stocks_df[target_column].shift(-1)
    stocks_df = stocks_df.dropna().reset_index(drop=True)

    # Split indices
    total_len = len(stocks_df)
    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)

    # Split data
    df_train = stocks_df.iloc[:train_end].copy()
    df_val = stocks_df.iloc[train_end:val_end].copy()
    df_test = stocks_df.iloc[val_end:].copy()

    # Columns for stock features (time series)
    stock_feature_cols = [target_column]
    if use_sentiment:
        sent_col = f"sent_{sent_window_size}"
        if sent_col not in stocks_df.columns:
            raise ValueError(f"Missing sentiment column: {sent_col}")
        stock_feature_cols.append(sent_col)

    # PCA on embeddings if requested
    pca = None
    if use_embeddings:
        if pca_dim:
            pca = PCA(n_components=pca_dim)
            pca.fit(df_train[embedding_cols].values)
            for df in [df_train, df_val, df_test]:
                reduced = pca.transform(df[embedding_cols].values)
                for i in range(pca_dim):
                    df[f"pca_{i}"] = reduced[:, i]
            embedding_cols_to_use = [f"pca_{i}" for i in range(pca_dim)]
        else:
            embedding_cols_to_use = embedding_cols
    else:
        embedding_cols_to_use = []

    # Scale stock features
    scaler = MinMaxScaler()
    df_train_stock = scaler.fit_transform(df_train[stock_feature_cols])
    df_val_stock = scaler.transform(df_val[stock_feature_cols])
    df_test_stock = scaler.transform(df_test[stock_feature_cols])

    # Prepare embeddings arrays
    if use_embeddings:
        train_embed = df_train[embedding_cols_to_use].values
        val_embed = df_val[embedding_cols_to_use].values
        test_embed = df_test[embedding_cols_to_use].values
    else:
        train_embed = val_embed = test_embed = None

    train_targets = df_train["target"].values
    val_targets = df_val["target"].values
    test_targets = df_test["target"].values

    # Create time series datasets for stock features (shape: [batch, window_size, features])
    def make_stock_ds(stock_data, targets):
        min_len = min(len(stock_data), len(targets))
        return tf.keras.preprocessing.timeseries_dataset_from_array(
            data=stock_data[:min_len],
            targets=targets[:min_len],
            sequence_length=window_size,
            sampling_rate=1,
            shuffle=False,
            batch_size=batch_size,
        )

    train_stock_ds = make_stock_ds(df_train_stock, train_targets)
    val_stock_ds = make_stock_ds(df_val_stock, val_targets)
    test_stock_ds = make_stock_ds(df_test_stock, test_targets)

    if use_embeddings:
        # For embeddings, we provide the embedding vector aligned to each sequence's *last* timestep
        # So, for each sequence (window), the embedding corresponds to the last date in that window

        def make_embed_ds(embed_data):
            sequences = []
            for i in range(len(embed_data) - window_size + 1):
                # Embedding of last day in window
                sequences.append(embed_data[i + window_size - 1])
            return tf.data.Dataset.from_tensor_slices(np.array(sequences)).batch(batch_size)

        train_embed_ds = make_embed_ds(train_embed)
        val_embed_ds = make_embed_ds(val_embed)
        test_embed_ds = make_embed_ds(test_embed)

        # Zip datasets to provide inputs as a tuple (stock_seq, embed_vec) with targets
        train_ds = tf.data.Dataset.zip((tf.data.Dataset.zip((train_stock_ds.map(lambda x, y: x), train_embed_ds)), train_stock_ds.map(lambda x, y: y)))
        val_ds = tf.data.Dataset.zip((tf.data.Dataset.zip((val_stock_ds.map(lambda x, y: x), val_embed_ds)), val_stock_ds.map(lambda x, y: y)))
        test_ds = tf.data.Dataset.zip((tf.data.Dataset.zip((test_stock_ds.map(lambda x, y: x), test_embed_ds)), test_stock_ds.map(lambda x, y: y)))
    else:
        # Only stock input
        train_ds = train_stock_ds
        val_ds = val_stock_ds
        test_ds = test_stock_ds

    return train_ds, val_ds, test_ds


def prepare_stock_datasets_flexible_old(
    stocks_df: pd.DataFrame,
    news_df: Optional[pd.DataFrame] = None,
    embedding_cols: Optional[List[str]] = None,
    window_size: int = 10,
    sent_window_size: Optional[int] = None,
    text_window_size: Optional[int] = None,
    pca_dim: Optional[int] = None,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    target_column: str = "Close",
    sent_column: str = "sent",
):
    """
    Prepares train/val/test datasets with optional sentiment and text embeddings.
    Returns:
        train_ds, val_ds, test_ds, scaler, pca (if used)
    """
    use_sentiment = sent_window_size is not None
    use_embeddings = text_window_size is not None

    if not use_sentiment and not use_embeddings:
        print("⚠️ Warning: neither sentiment nor embeddings are being used.")

    stocks_df = stocks_df.copy()
    convert_to_datetime(stocks_df, "Date")
    stocks_df = stocks_df.sort_values("Date").reset_index(drop=True)

    if news_df is not None:
        news_df = news_df.copy()
        news_df["Date"] = news_df["pub_date"]
        convert_to_datetime(news_df, "Date")
        news_df = news_df.sort_values("Date").reset_index(drop=True)
        embedding_cols = [c for c in news_df.columns if c.startswith("emb_")]

    if use_embeddings:
        embed_values = []
        embed_dim = len(embedding_cols)

        for date in stocks_df["Date"]:
            start_date = date - pd.Timedelta(days=text_window_size)
            subset = news_df[(news_df["Date"] >= start_date) & (news_df["Date"] <= date)]

            if not subset.empty:
                latest_embed = subset.iloc[-1][embedding_cols].values
            else:
                latest_embed = np.zeros(embed_dim)

            embed_values.append(latest_embed)

        # Convert to DataFrame with proper column names
        embed_df = pd.DataFrame(embed_values, columns=[f"emb_{i}" for i in range(embed_dim)])

        # Concatenate embeddings with stocks_df
        stocks_df = pd.concat([stocks_df.reset_index(drop=True), embed_df], axis=1)


    # Target
    stocks_df["target"] = stocks_df[target_column].shift(-1)
    stocks_df = stocks_df.dropna()

    # Split
    total_len = len(stocks_df)
    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)

    df_train = stocks_df.iloc[:train_end].copy()
    df_val = stocks_df.iloc[train_end:val_end].copy()
    df_test = stocks_df.iloc[val_end:].copy()

    # Prepare feature columns
    feature_cols = [target_column]

    if use_sentiment:
        sent_col = f"sent_{sent_window_size}"
        if sent_col not in stocks_df.columns:
            raise ValueError(f"Missing sentiment column: {sent_col}")
        feature_cols.append(sent_col)

    pca = None
    if use_embeddings:
        if pca_dim:
            pca = PCA(n_components=pca_dim)
            pca.fit(df_train[embedding_cols].values)
            for df in [df_train, df_val, df_test]:
                reduced = pca.transform(df[embedding_cols].values)
                for i in range(pca_dim):
                    df[f"pca_{i}"] = reduced[:, i]
            feature_cols.extend([f"pca_{i}" for i in range(pca_dim)])
        else:
            feature_cols.extend(embedding_cols)

    # Scale
    scaler = MinMaxScaler()
    df_train_scaled = scaler.fit_transform(df_train[feature_cols])
    df_val_scaled = scaler.transform(df_val[feature_cols])
    df_test_scaled = scaler.transform(df_test[feature_cols])

    train_targets = df_train["target"].values
    val_targets = df_val["target"].values
    test_targets = df_test["target"].values

    def make_ds(data, targets):
        min_len = min(len(data), len(targets))
        return tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data[:min_len],
            targets=targets[:min_len],
            sequence_length=window_size,
            sampling_rate=1,
            shuffle=False,
            batch_size=batch_size,
        )

    train_ds = make_ds(df_train_scaled, train_targets)
    val_ds = make_ds(df_val_scaled, val_targets)
    test_ds = make_ds(df_test_scaled, test_targets)

    return train_ds, val_ds, test_ds
