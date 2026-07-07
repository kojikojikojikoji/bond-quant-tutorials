# bondlab API リファレンス

`bondlab` は債券クオンツ・チュートリアルで層ごとに育てる自作ライブラリ。
各層は下位層にだけ依存し（下向き一方向）、上位層は下位層を import してよいが
逆流させない。層間の内部依存は `bond -> daycount`、`analytics -> curve`、
`models -> curve` の3本のみで、閉路を持たない有向非巡回グラフ（DAG）である。

バージョンは `pyproject.toml` の `version`（現在 `0.0.1`）でセマンティック
バージョニング（MAJOR.MINOR.PATCH）に従う。

| 層 | 役割 | 直接依存 | 構築シリーズ |
|---|---|---|---|
| `data` | 市場データ取得・キャッシュ | pandas | S0-2 |
| `rates` | 複利規約・割引係数 | numpy | S1-1 |
| `daycount` | 日数計算規約 | （標準ライブラリ） | S1-2 |
| `bond` | 固定利付債 | `daycount`, scipy | S1-3/1-4 |
| `curve` | イールドカーブ | numpy | S2 |
| `analytics` | リスク指標・PCA | `curve`, numpy | S3-1〜3-4 |
| `risk` | VaR / ES・検定 | scipy | S3-5 |
| `sim` | 確率過程シミュレーション | numpy, scipy | S4 |
| `models` | 短期金利モデル | `curve`, numpy | S5 |
| `pricing` | 金利デリバティブ | scipy | S6 |
| `credit` | クレジット | scipy | S7 |
| `mbs` | 証券化商品 | numpy | S8 |
| `bt` | バックテスト | pandas, numpy | S9 |

---

## `bondlab.data`（S0-2）市場データ取得・キャッシュ

| 公開シンボル | 種別 | 一行説明 |
|---|---|---|
| `fred_series(series_id, *, cache_only, max_retries, timeout)` | 関数 | FRED 時系列を取得し `pandas.Series` で返す。キャッシュ優先・鍵は `.env` から |
| `load_sample(name)` | 関数 | `data/samples/` の合成/再配布可データ（鍵不要）を読む |
| `CACHE_DIR`, `REPO_ROOT` | 定数 | キャッシュ先とリポジトリルートの `Path` |

## `bondlab.rates`（S1-1）金利・割引の基本

| 公開シンボル | 種別 | 一行説明 |
|---|---|---|
| `discount_factor(rate, t, convention)` | 関数 | レートと年数から割引係数（ベクトル化対応） |
| `rate_from_discount(df, t, convention)` | 関数 | 割引係数からレートを逆算（`discount_factor` の逆関数） |
| `convert_rate(rate, t, src, dst)` | 関数 | 割引係数を保ったままレートを規約変換 |
| `to_continuous(rate, convention)` | 関数 | m 回複利レート→連続複利レート（t 非依存） |
| `from_continuous(rate_c, convention)` | 関数 | 連続複利レート→m 回複利レート |

## `bondlab.daycount`（S1-2）日数計算規約

| 公開シンボル | 種別 | 一行説明 |
|---|---|---|
| `year_fraction(start, end, convention)` | 関数 | 2日付間の年数。ACT/360・ACT/365F・30/360・ACT/ACT に対応 |

## `bondlab.bond`（S1-3/1-4）固定利付債

| 公開シンボル | 種別 | 一行説明 |
|---|---|---|
| `FixedRateBond` | dataclass | 固定利付債（額面100基準・満期アンカーのスケジュール） |
| `FixedRateBond.cashflows()` | メソッド | (日付, CF) のリスト（満期に額面償還を加算） |
| `FixedRateBond.dirty_price(ytm, settlement)` | メソッド | street convention の dirty price |
| `FixedRateBond.clean_price(ytm, settlement)` | メソッド | clean price = dirty − 経過利子 |
| `FixedRateBond.accrued(settlement)` | メソッド | 経過利子（クーポン期間の経過割合で按分） |
| `FixedRateBond.yield_from_price(clean, settlement)` | メソッド | clean price から YTM を Brent 法で逆算 |
| `FixedRateBond.period_cashflows(settlement)` | メソッド | (期間指数 n=w+j, CF) 配列。デュレーション計算用 |

## `bondlab.curve`（S2）イールドカーブ

| 公開シンボル | 種別 | 一行説明 |
|---|---|---|
| `DiscountCurve(times, dfs, interp)` | クラス | 時点→割引係数のカーブ（log-linear / linear-zero 補間） |
| `DiscountCurve.discount(t)` | メソッド | 時点 t の割引係数（端は傾き外挿） |
| `DiscountCurve.zero_rate(t)` | メソッド | 連続複利ゼロレート $-\ln DF(t)/t$ |
| `DiscountCurve.forward_rate(t1, t2)` | メソッド | 区間 [t1,t2] の連続複利フォワード |
| `bootstrap_par(tenors, par_rates, frequency, interp)` | 関数 | パー利回りから割引係数を逐次剥ぎ取る（等間隔グリッド前提） |
| `nss(tau, beta0, beta1, beta2, beta3, lam1, lam2)` | 関数 | Nelson-Siegel-Svensson のゼロレート関数 |
| `fit_nss(tenors, yields, lam_grid)` | 関数 | NSS を利回りにフィット（λ 格子探索＋線形最小二乗） |

## `bondlab.analytics`（S3-1〜3-4）リスク指標・PCA

| 公開シンボル | 種別 | 一行説明 |
|---|---|---|
| `duration_convexity(bond, ytm, settlement)` | 関数 | Macaulay/修正デュレーション・コンベクシティ・DV01 を解析計算 |
| `effective_duration(price_fn, y0, bump)` | 関数 | 価格関数の実効デュレーション（中心差分。コーラブル債等） |
| `bump_curve(curve, tenor, size, width)` | 関数 | 特定テナー近傍のゼロレートをバンプした新カーブ（KRD 用） |
| `pca(changes)` | 関数 | カーブ変動行列の主成分分析（固有値・固有ベクトル・寄与率） |

## `bondlab.risk`（S3-5）VaR / ES・検定

| 公開シンボル | 種別 | 一行説明 |
|---|---|---|
| `historical_var(pnl, alpha)` | 関数 | ヒストリカル VaR（損失分位点を正で返す） |
| `historical_es(pnl, alpha)` | 関数 | ヒストリカル Expected Shortfall（順序統計量ベース） |
| `parametric_var(pnl, alpha)` | 関数 | 正規分布仮定のパラメトリック VaR |
| `parametric_es(pnl, alpha)` | 関数 | 正規分布仮定の Expected Shortfall |
| `kupiec_pof(exceptions, n, alpha)` | 関数 | Kupiec POF 検定（被覆率の尤度比検定） |

## `bondlab.sim`（S4）確率過程シミュレーション

| 公開シンボル | 種別 | 一行説明 |
|---|---|---|
| `brownian_paths(n_paths, n_steps, T, seed, antithetic)` | 関数 | 標準ブラウン運動のパス生成（対照変量対応） |
| `simulate_sde(drift, diffusion, x0, T, n_steps, n_paths, scheme, ...)` | 関数 | 1次元 SDE の Euler-Maruyama / Milstein 数値積分 |
| `mc_stats(samples, confidence)` | 関数 | モンテカルロ推定量の平均・標準誤差・信頼区間 |
| `control_variate(target, control, control_mean)` | 関数 | 制御変量法による分散削減 |

## `bondlab.models`（S5）短期金利モデル

| 公開シンボル | 種別 | 一行説明 |
|---|---|---|
| `Vasicek(a, b, sigma, r0)` | クラス | Vasicek モデル。`zcb` / `zero_rate` / `simulate`（厳密） |
| `CIR(a, b, sigma, r0)` | クラス | CIR モデル。`zcb` / `feller` / `simulate_exact`（非中心 χ²） |
| `HullWhite(a, sigma, curve)` | クラス | Hull-White（拡張 Vasicek）。初期カーブを厳密再現する `zcb(t, T)` |

## `bondlab.pricing`（S6）金利デリバティブ

| 公開シンボル | 種別 | 一行説明 |
|---|---|---|
| `swap_annuity(curve, times, start)` | 関数 | 固定レッグのアニュイティ（フォワードスタート対応） |
| `par_swap_rate(disc_curve, proj_curve, times, start)` | 関数 | マルチカーブのパースワップレート |
| `black76(forward, strike, expiry, vol, df, option)` | 関数 | Black'76（対数正規）コール/プット価格 |
| `bachelier(forward, strike, expiry, vol, df, option)` | 関数 | Bachelier（正規）コール/プット価格（負金利対応） |
| `implied_vol_black(price, forward, strike, expiry, df, option)` | 関数 | Black'76 インプライドボラの数値解 |
| `caplet(forward, strike, expiry, vol, tau, df, model)` | 関数 | 1本のキャップレット価格 |
| `swaption_black(forward_swap, strike, expiry, vol, annuity, option, model)` | 関数 | アニュイティ測度下のスワップション価格 |
| `sabr_vol(forward, strike, expiry, alpha, beta, rho, nu)` | 関数 | SABR インプライドボラ（Hagan 近似） |

## `bondlab.credit`（S7）クレジット

| 公開シンボル | 種別 | 一行説明 |
|---|---|---|
| `HazardCurve(times, hazards)` | クラス | 区分定数ハザードによる生存確率カーブ（`survival` / `default_prob`） |
| `cds_legs(hazard, disc_curve, maturity, recovery, freq, n_int)` | 関数 | CDS のリスキーアニュイティとプロテクション PV |
| `cds_par_spread(hazard, disc_curve, maturity, recovery, freq)` | 関数 | CDS パースプレッド |
| `bootstrap_hazard(disc_curve, tenors, spreads, recovery, freq)` | 関数 | CDS スプレッド列からハザードカーブを剥ぎ取る |
| `merton_equity(asset, asset_vol, debt, expiry, r)` | 関数 | Merton 構造モデルの株式価値（企業価値コール） |
| `distance_to_default(...)` / `merton_pd(...)` | 関数 | 距離 to default（d2）とリスク中立デフォルト確率 |
| `solve_asset(equity, equity_vol, debt, expiry, r)` | 関数 | 観測株式値から企業価値・資産ボラを逆算 |

## `bondlab.mbs`（S8）証券化商品

| 公開シンボル | 種別 | 一行説明 |
|---|---|---|
| `psa_cpr(age_months, psa)` | 関数 | PSA モデルの年率 CPR |
| `cpr_to_smm(cpr)` / `smm_to_cpr(smm)` | 関数 | 年率 CPR ↔ 月次 SMM の相互変換 |
| `mbs_cashflows(balance, wac, wam, smm)` | 関数 | パススルー MBS の月次キャッシュフロー生成 |
| `weighted_average_life(cashflows)` | 関数 | 加重平均年限 WAL |

## `bondlab.bt`（S9）バックテスト

| 公開シンボル | 種別 | 一行説明 |
|---|---|---|
| `backtest(signals, returns, cost_bps, lag)` | 関数 | シグナル×リターンの日次損益（1期ラグ約定・回転コスト） |
| `performance(pnl, periods_per_year)` | 関数 | 年率リターン・ボラ・シャープ・最大ドローダウン・勝率 |
| `conversion_factor(coupon, years_to_maturity, notional_coupon, freq)` | 関数 | 国債先物のコンバージョンファクター（簡易版） |
