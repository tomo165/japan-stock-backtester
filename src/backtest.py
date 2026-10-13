"""
バックテストエンジン
====================

スクリーニングで得たシグナル(Date, Code)を、翌営業日の寄付でエントリーし、
設定した手仕舞いルールで決済してトレード成績を集計する。

■ 先読みバイアス回避
  シグナルは「当日引け」で確定 → エントリーは「翌営業日の寄付(Open)」。

■ 手仕舞い(config.exit.mode)
  - "fixed"  : 固定保有日数 hold_days。entry_idx+hold_days 日目の Close で決済。
  - "tp_sl"  : 利確 tp% / 損切り sl% のどちらか先着で決済。max_hold_days で時間切れ。
               同一日に高値が利確・安値が損切りの両方に触れた場合は「損切り優先」(保守的)。

■ 生存者バイアスの反映
  保有中に上場廃止でデータが尽きた場合は、最後に取得できた Close で強制決済し
  delisted_exit=True を立てる(廃止による損失を成績に含める)。

■ コスト
  往復コスト cost_bps(bps)を各トレードのリターンから控除。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def default_exit_config() -> dict:
    return {
        "mode": "fixed",       # "fixed" or "tp_sl"
        "hold_days": 10,        # fixed 用: 保有営業日数
        "tp_pct": 10.0,         # tp_sl 用: 利確[%]
        "sl_pct": 5.0,          # tp_sl 用: 損切り[%]
        "max_hold_days": 20,    # tp_sl / trailing 用: 最大保有営業日数
        # --- trailing(トレイリングストップ)用 ---
        "trail_pct": 8.0,           # ピーク高値からの下落幅[%]でトレール決済
        "trail_activation_pct": 0.0,  # トレール発動に必要な含み益[%](0で即発動)
        "trail_init_sl_pct": None,    # 初期ハード損切り[%](null で無効)
        "cost_bps": 20.0,       # 往復売買コスト[bps] (0.20%)
        "one_position_per_code": True,  # 同一銘柄の保有中は新規シグナルを無視
    }


class _CodeSeries:
    """1銘柄の時系列を高速アクセスするための配列ラッパー。"""
    __slots__ = ("dates", "open", "high", "low", "close", "pos")

    def __init__(self, sub: pd.DataFrame):
        sub = sub.sort_values("Date")
        self.dates = sub["Date"].to_numpy()
        self.open = sub["Open"].to_numpy(dtype=float)
        self.high = sub["High"].to_numpy(dtype=float)
        self.low = sub["Low"].to_numpy(dtype=float)
        self.close = sub["Close"].to_numpy(dtype=float)
        self.pos = {d: i for i, d in enumerate(self.dates)}


def _simulate_trade(cs: _CodeSeries, entry_idx: int, cfg: dict,
                    side: str = "long") -> dict | None:
    """1トレードのシミュレーション。side='long'(買い) / 'short'(空売り)。

    ショートは全ロジックを鏡像化する:
      - 利確は下落方向、損切りは上昇方向。
      - トレイルは「エントリー以降の最安値(trough)」を基準に、trough から trail% 上昇で買戻し。
      - リターン = (entry - exit)/entry(下がるほど利益)。
    """
    n = len(cs.close)
    entry_price = cs.open[entry_idx]
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    mode = cfg["mode"]
    is_long = (side == "long")
    delisted = False

    if mode == "fixed":
        hold = int(cfg["hold_days"])
        exit_idx = entry_idx + hold
        if exit_idx >= n:
            exit_idx = n - 1
            delisted = True
        exit_price = cs.close[exit_idx]
        reason = "delisted" if delisted else "time"

    elif mode == "tp_sl":
        max_hold = int(cfg["max_hold_days"])
        last = min(entry_idx + max_hold, n - 1)
        exit_idx, exit_price, reason = last, cs.close[last], "time"
        if is_long:
            tp = entry_price * (1 + cfg["tp_pct"] / 100.0)
            sl = entry_price * (1 - cfg["sl_pct"] / 100.0)
            for j in range(entry_idx, last + 1):
                if cs.low[j] <= sl:      # 損切り優先(保守的)
                    exit_idx, exit_price, reason = j, sl, "sl"; break
                if cs.high[j] >= tp:
                    exit_idx, exit_price, reason = j, tp, "tp"; break
        else:  # short
            tp = entry_price * (1 - cfg["tp_pct"] / 100.0)  # 下落で利確
            sl = entry_price * (1 + cfg["sl_pct"] / 100.0)  # 上昇で損切り
            for j in range(entry_idx, last + 1):
                if cs.high[j] >= sl:     # 損切り優先(保守的)
                    exit_idx, exit_price, reason = j, sl, "sl"; break
                if cs.low[j] <= tp:
                    exit_idx, exit_price, reason = j, tp, "tp"; break
        if entry_idx + max_hold >= n and reason == "time" and exit_idx == n - 1:
            delisted = True; reason = "delisted"

    elif mode == "trailing":
        trail = cfg["trail_pct"] / 100.0
        act = cfg.get("trail_activation_pct", 0.0) / 100.0
        init_sl = cfg.get("trail_init_sl_pct", None)
        max_hold = int(cfg["max_hold_days"])
        last = min(entry_idx + max_hold, n - 1)
        activated = (act <= 0.0)
        exit_idx, exit_price, reason = last, cs.close[last], "time"

        if is_long:
            init_sl_price = entry_price * (1 - init_sl / 100.0) if init_sl else None
            peak = entry_price
            for j in range(entry_idx, last + 1):
                stop_level = peak * (1 - trail) if activated else None
                if init_sl_price is not None:
                    stop_level = init_sl_price if stop_level is None else max(stop_level, init_sl_price)
                if stop_level is not None and cs.low[j] <= stop_level:
                    exit_idx, exit_price = j, stop_level
                    reason = "trail" if (activated and (init_sl_price is None or stop_level > init_sl_price)) else "sl"
                    break
                if cs.high[j] > peak:
                    peak = cs.high[j]
                if not activated and peak >= entry_price * (1 + act):
                    activated = True
        else:  # short: trough を基準に上方向のストップ
            init_sl_price = entry_price * (1 + init_sl / 100.0) if init_sl else None
            trough = entry_price
            for j in range(entry_idx, last + 1):
                stop_level = trough * (1 + trail) if activated else None
                if init_sl_price is not None:
                    stop_level = init_sl_price if stop_level is None else min(stop_level, init_sl_price)
                if stop_level is not None and cs.high[j] >= stop_level:
                    exit_idx, exit_price = j, stop_level
                    reason = "trail" if (activated and (init_sl_price is None or stop_level < init_sl_price)) else "sl"
                    break
                if cs.low[j] < trough:
                    trough = cs.low[j]
                if not activated and trough <= entry_price * (1 - act):
                    activated = True
        if entry_idx + max_hold >= n and reason == "time" and exit_idx == n - 1:
            delisted = True; reason = "delisted"

    else:
        raise ValueError(f"未知の exit mode: {mode}")

    if is_long:
        gross_ret = exit_price / entry_price - 1.0
    else:
        gross_ret = (entry_price - exit_price) / entry_price
    net_ret = gross_ret - cfg.get("cost_bps", 0.0) / 1e4
    return {
        "entry_date": cs.dates[entry_idx],
        "exit_date": cs.dates[exit_idx],
        "side": side,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "hold_days": exit_idx - entry_idx,
        "gross_ret": gross_ret,
        "ret": net_ret,
        "reason": reason,
        "delisted_exit": delisted,
    }


def build_series(panel_ind: pd.DataFrame) -> dict:
    """銘柄ごとの価格系列(_CodeSeries)を構築。パラメータ探索で使い回すと高速。"""
    series: dict[str, _CodeSeries] = {}
    for code, sub in panel_ind.groupby("Code"):
        series[code] = _CodeSeries(sub[["Date", "Open", "High", "Low", "Close"]])
    return series


def run_backtest(panel_ind: pd.DataFrame, signals: pd.DataFrame, cfg: dict,
                 series: dict | None = None, side: str = "long") -> pd.DataFrame:
    """
    signals(Date, Code, ...)を1件ずつシミュレートし、トレード明細 DataFrame を返す。

    panel_ind: Open/High/Low/Close を含む全銘柄パネル(指標付与済みでも可)。
    series: build_series() の結果を渡すと再構築を省いて高速化(パラメータ探索用)。
    side: 'long'(買い) / 'short'(空売り)。
    """
    if series is None:
        series = build_series(panel_ind)
    if len(signals) == 0:
        return pd.DataFrame()

    one_per_code = cfg.get("one_position_per_code", True)
    held_until: dict[str, np.datetime64] = {}

    # iterrows は遅いので numpy 配列で回す
    sig = signals.sort_values(["Date", "Code"])
    codes = sig["Code"].to_numpy()
    dates = sig["Date"].to_numpy().astype("datetime64[ns]")

    trades = []
    for i in range(len(codes)):
        code = codes[i]
        sig_date = dates[i]
        cs = series.get(code)
        if cs is None:
            continue
        if one_per_code and code in held_until and sig_date <= held_until[code]:
            continue
        sig_idx = cs.pos.get(sig_date)
        if sig_idx is None:
            continue
        entry_idx = sig_idx + 1
        if entry_idx >= len(cs.close):
            continue
        tr = _simulate_trade(cs, entry_idx, cfg, side=side)
        if tr is None:
            continue
        tr["Code"] = code
        tr["signal_date"] = sig_date
        trades.append(tr)
        if one_per_code:
            held_until[code] = np.datetime64(tr["exit_date"])

    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame) -> dict:
    """トレード明細から主要指標を計算。PF, 勝率, 期待値, 合計損益など。"""
    if trades.empty:
        return {"trades": 0}
    r = trades["ret"]
    wins = r[r > 0]
    losses = r[r < 0]
    gross_profit = wins.sum()
    gross_loss = -losses.sum()
    pf = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "avg_ret": float(r.mean()),
        "median_ret": float(r.median()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(pf),
        "total_ret_sum": float(r.sum()),   # 等額投資の単純合計リターン
        "expectancy": float(r.mean()),
        "delisted_exits": int(trades["delisted_exit"].sum()),
        "avg_hold_days": float(trades["hold_days"].mean()),
    }
