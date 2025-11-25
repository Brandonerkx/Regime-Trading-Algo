def build_regime_predictions(
    features_df,
    regime_spy_df,
    sc,
    X_trend_train,
    trend_selected_features,
    trend_model,
    transition_matrix,
    n_trend_train
):

    regime_series = regime_spy_df['regime_id'].copy()
    regime_series.index = regime_series.index.normalize()
    feat_index = features_df.index
    regime_aligned = regime_series.reindex(feat_index)

    preds_dates = []
    preds_vals = []

    valid_mask = regime_aligned.notna()
    if not valid_mask.any():
        raise ValueError("No overlapping dates between features_df and regime_spy_df.")

    first_valid_idx = valid_mask[valid_mask].index[0]
    first_valid_pos = feat_index.get_loc(first_valid_idx)

    start_pos = max(n_trend_train, first_valid_pos)

    for i in range(start_pos, len(features_df) - 1):

        date_t = feat_index[i]
        date_tp1 = feat_index[i + 1]

        current_regime = regime_aligned.iat[i]
        if pd.isna(current_regime):
            continue
        current_regime = int(current_regime)

        try:
            X_t_raw = features_df.loc[[date_t], X_trend_train.columns]
        except KeyError:
            continue

        X_t_scaled = sc.transform(X_t_raw)
        X_t_scaled_df = pd.DataFrame(
            X_t_scaled,
            index=[date_t],
            columns=X_trend_train.columns
        )

        missing = [c for c in trend_selected_features if c not in X_t_scaled_df.columns]
        if missing:
            continue
        X_t_final = X_t_scaled_df[trend_selected_features]
        P_ml = trend_model.predict_proba(X_t_final)[0]
        P_markov = transition_matrix[current_regime]

        # Penalize Markov to avoid overpowering predictions from our model
        alpha = 0.8
        P_combined = (P_ml ** alpha) * (P_markov ** (1 - alpha))
        P_combined /= P_combined.sum()

        if P_combined.sum() == 0:
            P_combined = P_ml
        P_combined = P_combined / P_combined.sum()

        regime_next = int(np.argmax(P_combined))
        preds_dates.append(date_tp1)
        preds_vals.append(regime_next)

    regime_pred_next = pd.Series(preds_vals, index=preds_dates, name="regime_pred_next")
    return regime_pred_next

def _hurst_exponent(x):
    try:
        H, _, _ = compute_Hc(np.asarray(x), kind="price", simplified=True)
        return H
    except Exception:
        return np.nan

def rolling_hurst(series, window):
    return series.rolling(window=window, min_periods=window).apply(
        lambda a: _hurst_exponent(a), raw=False
    )

def classify_hurst_binary(h):
    if pd.isna(h):
        return np.nan
    if h > H_TRENDING_TH:
        return 0 
    else:
        return 1 
                   
                   
def kama_market_regime(df, col, n, m):
    df_copy = df.copy()
    kama_fast = ta.momentum.KAMAIndicator(df_copy[col], n).kama()
    kama_slow = ta.momentum.KAMAIndicator(df_copy[col], m).kama()
    kama_diff = kama_fast - kama_slow  #
    df_copy["kama_trend"] = 1          #
    df_copy.loc[kama_diff > 0, "kama_trend"] = 0  
    return df_copy
    # 0=Uptrend , 1=Downtrend
