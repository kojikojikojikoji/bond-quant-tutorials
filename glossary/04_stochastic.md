# 用語集 — 確率過程の数学（S4）

定義の正はこのファイル。各 notebook 末尾の用語表はここへのリンクとして扱う。
項目は S4 の各 notebook（S4-1〜S4-5）の初出順にまとまっている。

### ブラウン運動（Brownian motion）
- 定義: $W_0=0$、独立増分、定常増分 $W_t-W_s\sim\mathcal{N}(0,t-s)$、連続な見本路
  をもつ確率過程。見本路はほとんど確実に至る所微分不可能で非有界変動。
- 数式: $W_t\sim\mathcal{N}(0,t)$、$\operatorname{Cov}(W_s,W_t)=\min(s,t)$
- 関連: 独立増分, 2次変分, スケーリング極限
- 初出: S4-1

### 独立増分（independent increments）
- 定義: 重ならない時間区間にわたる増分が互いに独立という性質。
  $0\le t_0<\dots<t_k$ に対し $W_{t_1}-W_{t_0},\dots,W_{t_k}-W_{t_{k-1}}$ が独立。
- 関連: ブラウン運動
- 初出: S4-1

### 2次変分（quadratic variation）
- 定義: 分割を細かくした極限での増分2乗和。ブラウン運動では確定値 $t$ に収束する
  （$L^2$ かつほとんど確実に）。1次変分が発散するのと対照的で、$(dW)^2=dt$ の
  略記の根拠になる。
- 数式: $[W]_t=\lim\sum_i(\Delta W_i)^2=t$、$\operatorname{Var}[[W]^{(m)}_t]=2t^2/m$
- 関連: ブラウン運動
- 初出: S4-1

### スケーリング極限（scaling limit）
- 定義: ランダムウォークを時間 $1/n$・空間 $1/\sqrt{n}$ で縮小した $n\to\infty$ の
  極限。ドンスカーの定理により増分の分布によらず標準ブラウン運動へ分布収束する。
- 数式: $S^{(n)}_t=M_{\lfloor nt\rfloor}/\sqrt{n}\;\Rightarrow\;W_t$
- 関連: ブラウン運動
- 初出: S4-1

### 伊藤積分（Itô integral）
- 定義: 被積分過程 $H_t$ を各小区間の**左端点**で評価して構成する確率積分。
  左端点評価により $H_{t_i}$ が増分 $W_{t_{i+1}}-W_{t_i}$ と独立になり、積分が
  マルチンゲールになる。有界変動でないブラウン運動が相手なので、評価点を
  中点にずらすと値が変わる（Stratonovich 積分との差は2次変分ぶん $T/2$）。
- 数式: $\int_0^T H_t\,dW_t = \lim_{n\to\infty}\sum_i H_{t_i}(W_{t_{i+1}}-W_{t_i})$
- 関連: 伊藤の公式, 2次変分, Ornstein-Uhlenbeck過程
- 初出: S4-2

### 伊藤の公式（Itô's formula）
- 定義: 確率過程 $X_t$ の関数 $f(t,X_t)$ の微分を与える公式。通常の連鎖律に
  加えて2次変分由来の $\tfrac12\sigma^2 f_{xx}$ 項が付く。GBM・OU の厳密解の
  導出や、SDE で表した価格過程の分布計算の中心的な道具。
- 数式: $df = (f_t + \mu f_x + \tfrac12\sigma^2 f_{xx})\,dt + \sigma f_x\,dW$。
  例 $d(W_t^2)=2W_t\,dW_t+dt$
- 関連: 伊藤積分, ドリフト, 拡散係数
- 初出: S4-2

### ドリフト（drift）
- 定義: SDE $dX=\mu\,dt+\sigma\,dW$ の $dt$ の係数 $\mu(t,x)$。単位時間あたりの
  変化の平均的な向きと大きさを表す。OU 過程では $\kappa(\theta-x)$ が平均回帰を担う。
- 数式: $\mu(t,x)=\lim_{h\to0}\tfrac1h\,\mathbb{E}[X_{t+h}-X_t\mid X_t=x]$
- 関連: 拡散係数, 伊藤の公式
- 初出: S4-2

### 拡散係数（diffusion coefficient）
- 定義: SDE の $dW$ の係数 $\sigma(t,x)$。揺らぎ（ボラティリティ）の大きさを表す。
  二乗が単位時間あたりの分散増加率 $\sigma^2$ を与える（$(dW)^2=dt$ より）。
- 数式: $\sigma^2(t,x)=\lim_{h\to0}\tfrac1h\,\mathrm{Var}[X_{t+h}-X_t\mid X_t=x]$
- 関連: ドリフト, 伊藤積分
- 初出: S4-2

### Ornstein-Uhlenbeck過程（Ornstein-Uhlenbeck process）
- 定義: 水準 $\theta$ へ速さ $\kappa>0$ で引き戻される平均回帰過程。条件付き分布・
  定常分布ともに正規分布になる、扱いやすいガウス過程。S5 の Vasicek 金利モデルの
  素過程。定常分布の平均は $\theta$、分散は $\sigma^2/(2\kappa)$。
- 数式: $dX_t=\kappa(\theta-X_t)\,dt+\sigma\,dW_t$、
  $X_\infty\sim\mathcal{N}(\theta,\ \sigma^2/(2\kappa))$
- 関連: 伊藤積分, ドリフト
- 初出: S4-2

### Euler-Maruyama法（Euler-Maruyama method）
- 定義: SDE $dX=a\,dt+b\,dW$ を区間左端の係数で離散化する最も基本的な数値解法。
  決定的な Euler 法に確率項 $b\,\Delta W_i$ を足した形で、あらゆる SDE に使えるが、
  拡散が状態依存（乗法ノイズ）の場合パス単位の精度（強収束）は $\sqrt{\Delta t}$ で頭打ち。
- 数式: $X_{i+1}=X_i+a(t_i,X_i)\,\Delta t+b(t_i,X_i)\,\Delta W_i$、強収束次数 0.5・弱収束次数 1.0
- 関連: Milstein法, 強収束, 弱収束, ドリフト, 拡散係数
- 初出: S4-3

### Milstein法（Milstein method）
- 定義: Euler-Maruyama に、区間内での拡散係数の揺らぎを1段補正する項を加えた解法。
  伊藤・テイラー展開から現れる $\int\Delta W\,dW=\tfrac12(\Delta W^2-\Delta t)$ に
  $b\,\partial_x b$ を掛けて加える。乗法ノイズでも強収束次数が 1.0 に上がる。拡散が
  定数（加法ノイズ）なら $\partial_x b=0$ で Euler-Maruyama に一致。
- 数式: $X_{i+1}=X_i+a\,\Delta t+b\,\Delta W_i+\tfrac12\,b\,\partial_x b\,(\Delta W_i^2-\Delta t)$
- 関連: Euler-Maruyama法, 強収束, 拡散係数
- 初出: S4-3

### 強収束（strong convergence）
- 定義: 真の解と数値解を**同一のブラウン運動**で走らせたときの、パス単位の終端誤差の
  期待値が $O(\Delta t^\gamma)$ で減る速さ。$\gamma$ を強収束次数と呼ぶ。軌道そのものの
  一致を測るので、マルチレベル MC や経路依存量の厳密解検証で問題になる。
- 数式: $\mathbb{E}\big[\,|X^{\Delta t}_N-X_T|\,\big]\le C\,\Delta t^{\gamma}$（EM: $\gamma=0.5$, Milstein: $\gamma=1.0$）
- 関連: 弱収束, Euler-Maruyama法, Milstein法
- 初出: S4-3

### 弱収束（weak convergence）
- 定義: 滑らかな関数の期待値（分布・モーメント）の誤差が $O(\Delta t^\beta)$ で減る速さ。
  $\beta$ を弱収束次数と呼ぶ。個々のパスの一致は問わない。デリバティブ価格は割引ペイオフの
  期待値なので、価格評価に必要なのは弱収束であり、EM でも $\beta=1.0$ で足りる。
- 数式: $\big|\,\mathbb{E}[g(X^{\Delta t}_N)]-\mathbb{E}[g(X_T)]\,\big|\le C\,\Delta t^{\beta}$（EM・Milstein とも $\beta=1.0$）
- 関連: 強収束, Euler-Maruyama法
- 初出: S4-3

### full truncation（full truncation）
- 定義: CIR など拡散に平方根を含む過程の Euler 離散化で、拡散項に $\sqrt{\max(r,0)}$ を
  使う修正。状態が負に振れても平方根が実数のまま計算でき、NaN 破綻を防ぐ。負値到達率
  そのものは素朴な $\sqrt{r}$ と同じ（負に落ちるまで同一軌道）だが、負領域で拡散が 0 に
  なりドリフトの平均回帰で正へ戻すため、浅い負値に留めて完走させられる。
- 数式: $r_{i+1}=r_i+\kappa(\theta-r_i)\,\Delta t+\sigma\sqrt{\max(r_i,0)}\,\Delta W_i$
- 関連: Euler-Maruyama法, Ornstein-Uhlenbeck過程
- 初出: S4-3

### 標準誤差（standard error）
- 定義: モンテカルロ推定量 $\hat{\theta}_n$ のばらつきの尺度。標本標準偏差 $s$ を
  $\sqrt{n}$ で割った値で、中心極限定理から $\hat{\theta}_n \pm 1.96\,\mathrm{SE}$ が
  近似95%信頼区間になる。パス数を増やしても $1/\sqrt{n}$ でしか縮まない。
- 数式: $\mathrm{SE} = \dfrac{s}{\sqrt{n}}, \quad s^2 = \dfrac{1}{n-1}\sum_i (X_i-\hat{\theta}_n)^2$
- 関連: 対照変量法, 制御変量法
- 初出: S4-4

### 対照変量法（antithetic variates）
- 定義: 正規乱数 $Z$ とその符号反転 $-Z$ を対にし、両者のペイオフの平均を1標本に
  使う分散削減法。ペイオフが単調なら対が負に相関し、独立2標本より分散が下がる。
- 数式: $\operatorname{Var}(\bar{X}) = \dfrac{\sigma^2}{2}(1+\rho), \quad \rho=\operatorname{Corr}(f(Z),f(-Z))$
- 関連: 標準誤差, 制御変量法
- 初出: S4-4

### 制御変量法（control variate）
- 定義: 目標 $X$ と相関し期待値 $\mu_Y$ が既知の変量 $Y$ を使い、$X-c(Y-\mu_Y)$ で
  補正して分散を下げる不偏な手法。最適係数 $c^\*=\operatorname{Cov}(X,Y)/\operatorname{Var}(Y)$
  のとき分散は $1-\rho_{XY}^2$ 倍になる。$c^\*$ は $X$ を $Y$ に回帰した傾きに等しい。
- 数式: $c^\* = \dfrac{\operatorname{Cov}(X,Y)}{\operatorname{Var}(Y)}, \quad \operatorname{Var}(X(c^\*)) = \operatorname{Var}(X)(1-\rho_{XY}^2)$
- 関連: 標準誤差, 対照変量法
- 初出: S4-4

### 準乱数（quasi-random）
- 定義: 定義域 $[0,1)^d$ を意図的に均一に埋める決定的な点列（低食い違い量列）。
  擬似乱数の $O(n^{-1/2})$ に対し積分誤差が速く縮む。逆累積分布 $\Phi^{-1}$ で
  正規乱数に変換して MC に使う。決定的なので誤差はスクランブルの反復で見積もる。
- 数式: $\left|\frac1n\sum_i g(u_i)-\int g\,du\right| \le V(g)\,D_n^\*$（Koksma-Hlawka）
- 関連: Sobol系列, 標準誤差
- 初出: S4-4

### Sobol系列（Sobol sequence）
- 定義: 代表的な低食い違い量列の一つ。$2^m$ 個の点で均一性が最良になる。次元が
  小さければ食い違い量が実質 $O(n^{-1})$ に近く、$1/\sqrt{n}$ より速く収束する。
- 数式: $D_n^\* = O\!\bigl((\log n)^d / n\bigr)$
- 関連: 準乱数
- 初出: S4-4

### リスク中立測度（risk-neutral measure）
- 定義: 割引資産をマルチンゲールにする確率測度 $\mathbb{Q}$。この測度の下では、任意の資産の
  現在価格が「割引した将来ペイオフの期待値」に一致する。価格づけに使う測度で、現実の
  当たりやすさ（実測度）とは別物。
- 数式: $S_0 = \mathbb{E}^{\mathbb{Q}}\!\left[ e^{-rT} S_T \right]$
- 関連: 複製ポートフォリオ, ニュメレール, Girsanovの定理
- 初出: S4-5

### ニュメレール（numéraire）
- 定義: 価値を測る基準に選ぶ、正の価格を持つ資産。ニュメレールを取り替えると対応する
  確率測度も変わり、その資産で割った価格がマルチンゲールになる。無リスク債ならリスク中立
  測度、割引債ならフォワード測度、アニュイティならアニュイティ測度に対応する。
- 数式: $\dfrac{V_t}{N_t} = \mathbb{E}^{\mathbb{Q}^N}\!\left[\dfrac{V_T}{N_T}\,\middle|\,\mathcal{F}_t\right]$
- 関連: リスク中立測度
- 初出: S4-5

### Girsanovの定理（Girsanov's theorem）
- 定義: 測度を実測度 $\mathbb{P}$ からリスク中立測度 $\mathbb{Q}$ へ取り替えると、ブラウン運動に
  ドリフトが付け替わることを述べる定理。拡散項は不変で、変わるのはドリフトだけ。付け替えの
  重みはラドン・ニコディム微分で与えられる。
- 数式: $W_t^{\mathbb{Q}} = W_t^{\mathbb{P}} + \theta t$、$\dfrac{d\mathbb{Q}}{d\mathbb{P}} = \exp\!\left(-\theta W_T^{\mathbb{P}} - \tfrac12\theta^2 T\right)$
- 関連: リスクの市場価格, リスク中立測度, ブラウン運動
- 初出: S4-5

### 複製ポートフォリオ（replicating portfolio）
- 定義: デリバティブと同じ将来ペイオフを再現する、原資産と無リスク債の組み合わせ。無裁定
  より、デリバティブ価格は複製ポートフォリオの組成コストに一致する。これが「期待値で価格が
  出る」ことの根拠。
- 数式: $f_0 = \Delta S_0 + B$、$\Delta = \dfrac{f_u - f_d}{S_0(u-d)}$
- 関連: リスク中立測度
- 初出: S4-5

### リスクの市場価格（market price of risk）
- 定義: リスク 1 単位あたりの超過リターン $\theta=(\mu-r)/\sigma$。Girsanov の付け替え量が
  ちょうどこの $\theta$ で、実測度とリスク中立測度を橋渡しする。$\mathbb{Q}$ の下では全資産が
  リスクプレミアムを剥がした金利 $r$ で成長する。
- 数式: $\theta = \dfrac{\mu - r}{\sigma}$
- 関連: Girsanovの定理, リスク中立測度
- 初出: S4-5
