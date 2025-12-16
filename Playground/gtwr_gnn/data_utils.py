import numpy as np
import pandas as pd

def load_panel_xlsx(path_xlsx, lat_col, lon_col, time_col, target_col, feature_cols):
    df = pd.read_excel(path_xlsx)
    df = df[[lat_col, lon_col, time_col, target_col] + list(feature_cols)].dropna().copy()
    return df

def build_panel_arrays(df, time_col, target_col, feature_cols, lat_col, lon_col, times_sorted=None):
    if times_sorted is None:
        times_sorted = np.sort(df[time_col].unique())
    df_sorted = df.sort_values([time_col, lat_col, lon_col]).reset_index(drop=True)

    X_blocks, y_blocks, C_blocks = [], [], []
    Ns = []
    for t in times_sorted:
        dtt = df_sorted[df_sorted[time_col] == t].reset_index(drop=True)
        X_blocks.append(dtt[feature_cols].values.astype(np.float32))
        y_blocks.append(dtt[target_col].values.astype(np.float32))
        C_blocks.append(dtt[[lat_col, lon_col]].values.astype(np.float32))
        Ns.append(len(dtt))
    # balance panel to min N
    if len(set(Ns)) != 1:
        minN = min(Ns)
        X_blocks = [x[:minN] for x in X_blocks]
        y_blocks = [y[:minN] for y in y_blocks]
        C_blocks = [c[:minN] for c in C_blocks]
        N_per_year = minN
    else:
        N_per_year = Ns[0]

    X_all = np.vstack(X_blocks)
    y_all = np.hstack(y_blocks)
    C_all = np.vstack(C_blocks)
    return dict(X_all=X_all, y_all=y_all, coords_all=C_all,
                coords_blocks=C_blocks, times=times_sorted, N_per_year=N_per_year)

def year_rows(times_sorted, N_per_year, target_year):
    idx, cur = [], 0
    for t in times_sorted:
        if t == target_year:
            idx.extend(range(cur, cur+N_per_year))
        cur += N_per_year
    return np.array(idx)

def split_train_val_test(times_sorted, N_per_year, use_val=True):
    test_year = times_sorted[-1]
    val_year = times_sorted[-2] if use_val else None
    train_years = times_sorted[:-2] if use_val else times_sorted[:-1]
    train_rows = np.concatenate([year_rows(times_sorted, N_per_year, y) for y in train_years])
    val_rows   = year_rows(times_sorted, N_per_year, val_year) if use_val else np.array([], dtype=int)
    test_rows  = year_rows(times_sorted, N_per_year, test_year)
    return dict(train_rows=train_rows, val_rows=val_rows, test_rows=test_rows,
                train_years=train_years, val_year=val_year, test_year=test_year)
