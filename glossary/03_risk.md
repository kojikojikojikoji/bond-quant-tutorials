# 用語集 — リスク指標（S3）

定義の正はこのファイル。各 notebook 末尾の用語表はここへのリンクとして扱う。
項目は S3 の各 notebook（S3-1〜S3-5）の初出順にまとまっている。

<a id="modified-duration"></a>
### 修正デュレーション（modified duration）
- 定義: 利回りが1（=100%）動いたときの価格の相対変化率。マコーレーデュレーション
  $D_{\mathrm{mac}}$ を複利補正 $1/(1+y/f)$ で割った量。価格-利回り曲線の接線の傾きを
  価格で正規化したものにあたる。
- 数式: $D_{\mathrm{mod}} = -\dfrac{1}{P}\dfrac{dP}{dy} = \dfrac{D_{\mathrm{mac}}}{1+y/f}$
- 関連: コンベクシティ, 実効デュレーション
- 初出: S3-1

<a id="effective-duration"></a>
### 実効デュレーション（effective duration）
- 定義: 価格関数をバンプ（微小変化）して中心差分で数値微分した感応度。キャッシュ
  フローが利回りに依存するコーラブル債・MBS でも定義でき、固定キャッシュフロー債
  では修正デュレーションに一致する。
- 数式: $D_{\mathrm{eff}} = -\dfrac{P(y_0+\Delta y)-P(y_0-\Delta y)}{2\,\Delta y\,P(y_0)}$
- 関連: 修正デュレーション, バンプ法
- 初出: S3-1

<a id="convexity"></a>
### コンベクシティ（convexity）
- 定義: 価格-利回り曲線の曲率を価格で正規化した量。価格の利回りに対する2階微分。
  将来キャッシュフローが正の債券では常に正で、利回りがどちらへ動いても価格を接線
  より上へ押し上げる。期限前償還のある商品では負になりうる（負のコンベクシティ）。
- 数式: $C = \dfrac{1}{P}\dfrac{d^2P}{dy^2} = \dfrac{1}{P}\sum_j \dfrac{n_j(n_j+1)}{f^2}\,c_j\left(1+\tfrac{y}{f}\right)^{-n_j-2}$
- 関連: 修正デュレーション, テイラー展開
- 初出: S3-1

<a id="bump-and-revalue"></a>
### バンプ法（bump-and-revalue）
- 定義: 入力（利回り・カーブ等）を微小変化させ、再評価した価格差から感応度を数値
  微分する手法。解析式が無い商品にも使える汎用手段で、中心差分の打ち切り誤差は
  $O(\Delta y^2)$。
- 数式: $\dfrac{\partial P}{\partial y}\approx\dfrac{P(y_0+\Delta y)-P(y_0-\Delta y)}{2\,\Delta y}$
- 関連: 実効デュレーション
- 初出: S3-1

<a id="taylor-expansion"></a>
### テイラー展開（Taylor expansion）
- 定義: 関数を基準点まわりの多項式で近似する展開。価格変化を修正デュレーション
  （1次）とコンベクシティ（2次）で表す近似式の土台。3次以降を打ち切った誤差が
  大幅シフト時に効く。
- 数式: $\dfrac{\Delta P}{P}\approx -D_{\mathrm{mod}}\,\Delta y+\tfrac12 C\,\Delta y^2$
- 関連: 修正デュレーション, コンベクシティ
- 初出: S3-1

<a id="dollar-value-of-a-basis-point"></a>
### DV01（dollar value of a basis point）
- 定義: 利回りが1ベーシスポイント（$10^{-4}$）動いたときの価格変化額。利回り上昇に
  対する下落幅を正で測る。デュレーション（年）でも％でもなく、額面あたりの金額なので
  異なる銘柄をそのまま足し引きできる。
- 数式: $\mathrm{DV01} = -\dfrac{\partial P}{\partial y}\times 10^{-4} = D_{\mathrm{mod}}\,P\times 10^{-4}$
- 関連: BPV, 修正デュレーション, ヘッジ比率
- 初出: S3-2

<a id="basis-point-value"></a>
### BPV（basis point value）
- 定義: 1bpあたりの価格感応度を金額で表したもの。実務では DV01 と同義に使う。
- 数式: $\mathrm{BPV}=\mathrm{DV01}$
- 関連: DV01, PV01
- 初出: S3-2

<a id="present-value-of-a-basis-point"></a>
### PV01（present value of a basis point）
- 定義: 各利払日に1bpのクーポンが乗った年金（アニュイティ）の現在価値。割引カーブ
  から直接求め、スワップのフィックス脚などキャッシュフロー側を1bp動かす感応度。利回り
  を動かして再評価する DV01 とは近い値になるが出所が異なる。
- 数式: $\mathrm{PV01}\approx \sum_i \tau_i\,\mathrm{DF}_i\times 10^{-4}$
- 関連: DV01, BPV
- 初出: S3-2

<a id="hedge-ratio"></a>
### ヘッジ比率（hedge ratio）
- 定義: 対象ポジションのDV01を打ち消すために保有するヘッジ商品の数量。単一商品では
  対象とヘッジのDV01比で決まり、平行シフトの一次損益を消す。傾きまで消すにはヘッジ
  商品を2本に増やし、水準・傾きの連立を解く。
- 数式: $h = -\dfrac{\mathrm{DV01}_P}{\mathrm{DV01}_H}$
- 関連: DV01, ベーシスリスク
- 初出: S3-2

<a id="basis-risk"></a>
### ベーシスリスク（basis risk）
- 定義: 対象とヘッジ商品の利回りが完全には連動しないために残るリスク。DV01を揃えても
  両者のスプレッドが動けば損益が残る。ヘッジ後PnLの残差要因の一つ。
- 関連: ヘッジ比率, DV01
- 初出: S3-2

<a id="central-difference"></a>
### 中心差分（central difference）
- 定義: 基準点の前後を対称にバンプして数値微分を取る方式。片側差分（打ち切り誤差
  $O(h)$）より精度が高く $O(h^2)$。DV01のバンプ再評価で標準的に使う。
- 数式: $\dfrac{\partial P}{\partial y}\approx \dfrac{P(y+h)-P(y-h)}{2h}$
- 関連: DV01, バンプ法
- 初出: S3-2

<a id="key-rate-duration-krd"></a>
### キーレートデュレーション（key rate duration, KRD）
- 定義: イールドカーブを代表テナー（キーレート）に分け、特定テナー近傍のゼロ
  レートだけを局所的に動かしたときの価値の相対感応度（単位は年）。カーブリスクを
  テナー別に分解する。キーテナーが全ノードを覆えば合計は修正デュレーションに一致
  する。
- 数式: $\mathrm{KRD}_k = -\dfrac{1}{V}\dfrac{\partial V}{\partial r_k},\qquad \sum_k \mathrm{KRD}_k \approx D_{\mathrm{mod}}$
- 関連: 修正デュレーション, パーシャルDV01, バケッティング, バンプ法
- 初出: S3-3

<a id="partial-dv01-dv01"></a>
### パーシャルDV01（partial DV01, キーレートDV01）
- 定義: 特定テナーのゼロレートを1bp動かしたときの価値変化額。KRDの金額版で、
  テナー別のヘッジ数量（当該年限の先物・スワップDV01で割る）に直結する。全テナーの
  和は並行バンプの全体DV01に一致する。
- 数式: $\mathrm{PDV01}_k = -\bigl(V(\tau_k,+1\text{bp})-V\bigr)=\mathrm{KRD}_k\cdot V\cdot 10^{-4}$
- 関連: キーレートデュレーション, DV01, バケッティング
- 初出: S3-3

<a id="bucketing"></a>
### バケッティング（bucketing）
- 定義: 連続的なカーブを有限個のキーテナーへ割り当てるリスク分解の設計。論点は
  キーテナーの選択（流動性の高い年限）、粒度（細かいほど解像度は上がるが隣接
  バケットの相関が増え不安定）、キー間の配分（三角バンプによる隣接テナーへの
  染み出し）。
- 関連: キーレートデュレーション, パーシャルDV01
- 初出: S3-3

<a id="barbell"></a>
### バーベル（barbell）
- 定義: 短期と長期の年限に分けて保有し、中期を薄くしたポジション。同一デュレー
  ションのブレットに比べKRDの分散（テナー方向の広がり）とコンベクシティが大きく、
  フラットナー・並行シフトで有利、スティープナーで不利になりやすい。
- 関連: ブレット, キーレートデュレーション, コンベクシティ
- 初出: S3-3

<a id="bullet"></a>
### ブレット（bullet）
- 定義: 単一年限に集中させたポジション。同一デュレーションのバーベルに比べKRDが
  一点に集まり分散・コンベクシティが小さい。回転中心付近に集中させればスティープ
  ナー・フラットナーに鈍感で、スティープナーで相対的に有利になりやすい。
- 関連: バーベル, キーレートデュレーション
- 初出: S3-3

<a id="principal-component-analysis-pca"></a>
### 主成分分析（principal component analysis, PCA）
- 定義: テナー間で相関して動くカーブ変動を、互いに無相関な少数の軸（主成分）へ
  座標変換する手法。標本共分散行列 $\Sigma$ を固有値分解し、固有ベクトルを負荷
  ベクトル、固有値を各軸の分散として、寄与率の降順に並べる。金利カーブでは第1〜3
  主成分がレベル・スロープ・曲率に対応し、分散の約95〜99%を説明する。
- 数式: $\Sigma = \tfrac{1}{T-1}\tilde X^\top\tilde X,\qquad \Sigma w_i=\lambda_i w_i$
- 関連: 固有値, 寄与率, パラレルシフト, バタフライ
- 初出: S3-4

<a id="eigenvalue"></a>
### 固有値（eigenvalue）
- 定義: $\Sigma w=\lambda w$ を満たすスカラー $\lambda$。PCAでは、対応する主成分方向へ
  射影したときの分散に等しい（$w^\top\Sigma w=\lambda$）。共分散行列は対称半正定値
  なので固有値は非負の実数で、降順に並べた順が主成分の順になる。
- 数式: $w_i=\arg\max_{\|w\|=1} w^\top\Sigma w \ \Rightarrow\ \Sigma w_i=\lambda_i w_i,\ \ \operatorname{Var}=\lambda_i$
- 関連: 主成分分析, 寄与率
- 初出: S3-4

<a id="explained-variance-ratio"></a>
### 寄与率（explained variance ratio）
- 定義: 全分散に対する各主成分の分散（固有値）の割合。第 $i$ 主成分の寄与率は
  $\lambda_i/\sum_j\lambda_j$、上位 $k$ 個の和を累積寄与率と呼び、少数因子で
  カーブ変動をどれだけ説明できるかを測る。金利では第1〜3主成分の累積で約95〜99%。
- 数式: $\text{寄与率}_i=\dfrac{\lambda_i}{\sum_j\lambda_j},\qquad \text{累積}=\dfrac{\sum_{i\le k}\lambda_i}{\sum_j\lambda_j}$
- 関連: 主成分分析, 固有値
- 初出: S3-4

<a id="stress-test"></a>
### ストレステスト（stress test）
- 定義: 極端だが起こりうるシナリオ（大幅な金利上昇・スティープ化・ヒストリカルな
  危機再現など）の下でポートフォリオ損益と耐性を評価する手法。ヒストリカル
  シナリオ（過去の実現変動を再現）と仮想シナリオ（PCA因子に沿った $k$ シグマ
  ショック等）を併用し、想定の抜けを補い合う。
- 関連: シナリオ損益, 主成分分析, パラレルシフト
- 初出: S3-4

<a id="scenario-p-l"></a>
### シナリオ損益（scenario P&L）
- 定義: あるシナリオ（カーブへの bp シフト等）でポートフォリオを再評価した価値と、
  基準価値との差。正が利益。全建玉を割引カーブで再評価し、変化後−基準を加重合計
  する。最小値がワースト損失で、ストレステストの主要な出力になる。
- 数式: $\text{P\&L}(s)=\sum_b w_b\bigl(V_b(\text{shifted}_s)-V_b(\text{base})\bigr)$
- 関連: ストレステスト, パラレルシフト, バタフライ
- 初出: S3-4

<a id="parallel-shift"></a>
### パラレルシフト（parallel shift）
- 定義: 全テナーのゼロレートが同量だけ動くカーブ変動。PCAの第1主成分（レベル）に
  対応し、正のデュレーションを持つポートフォリオでは金利上昇側が最大損失になり
  やすい。スティープ化・フラット化は短期を軸に長期が上下する第2主成分（スロープ）
  の $\pm$ 側。
- 数式: $\Delta z(\tau)=a$（レベル）, $\Delta z(\tau)=b(\tau-\bar\tau)$（スロープ）
- 関連: 主成分分析, バタフライ, 修正デュレーション
- 初出: S3-4

<a id="butterfly"></a>
### バタフライ（butterfly）
- 定義: 中期（belly）と両端（wings）が逆方向に動く曲率変化。PCAの第3主成分に対応し、
  中期を買い両端を売る（またはその逆）トレードが該当する。金利では寄与率が小さく、
  低分散ゆえに負荷ベクトルの形は標本ごとに不安定になりやすい（回転の任意性）。
- 数式: $\Delta z(\tau)=c\bigl((\tau-\bar\tau)^2-k\bigr)$
- 関連: 主成分分析, パラレルシフト, コンベクシティ
- 初出: S3-4

<a id="value-at-risk"></a>
### VaR（Value at Risk）
- 定義: 信頼水準 $\alpha$ のもとで、損失がそれ以下に収まる確率が $\alpha$ となる損失水準。
  損益分布（正が利益）の下側 $(1-\alpha)$ 分位点を正の損失で表す。分位点という一点しか見ない
  ため裾の深さは測れず、劣加法性も一般には満たさない。
- 数式: $\mathrm{VaR}_\alpha(X) = -\,q_{1-\alpha}(X)$（$q$ は損益分布の分位点）
- 関連: エクスペクテッド・ショートフォール, 劣加法性, バックテスティング
- 初出: S3-5

<a id="expected-shortfall"></a>
### エクスペクテッド・ショートフォール（Expected Shortfall）
- 定義: VaR を超える損失の条件付き期待値。CVaR とも呼ぶ。裾の平均損失を測るため、
  超過損失の深さまで捉える。常に VaR 以上で、劣加法性を満たすコヒーレントなリスク尺度。
  FRTB が内部モデルの主指標を VaR から 97.5% ES へ移した対象。
- 数式: $\mathrm{ES}_\alpha(X) = -\,\mathbb{E}[X \mid X \le q_{1-\alpha}(X)] = \dfrac{1}{1-\alpha}\displaystyle\int_0^{1-\alpha}\mathrm{VaR}_u\,du$
- 関連: VaR, 劣加法性
- 初出: S3-5

<a id="subadditivity"></a>
### 劣加法性（subadditivity）
- 定義: リスク尺度 $\rho$ が $\rho(A+B)\le\rho(A)+\rho(B)$ を満たす性質。「分散でリスクは
  増えない」という直観に対応し、コヒーレント性の一条件。ES は常に満たすが、VaR は一般に
  満たさない（独立でまれに大損する2資産で反例が組める）。
- 数式: $\rho(A+B)\le\rho(A)+\rho(B)$
- 関連: VaR, エクスペクテッド・ショートフォール
- 初出: S3-5

<a id="kupiec-test-pof-test"></a>
### Kupiec検定（Kupiec test, POF test）
- 定義: VaR の被覆率が正しいかを、観測例外数 $x$ が二項 $\mathrm{Bin}(n,1-\alpha)$ と整合するかで
  検定する尤度比検定（proportion of failures 検定）。帰無仮説のもとで統計量は $\chi^2_1$ に従い、
  $p$ 値が小さければモデルを棄却する。例外が少なすぎる（過度に保守的）場合も棄却する。
- 数式: $LR_{POF} = -2\ln\dfrac{(1-p)^{\,n-x}p^{\,x}}{(1-\hat p)^{\,n-x}\hat p^{\,x}}\sim\chi^2_1,\quad p=1-\alpha,\ \hat p=x/n$
- 関連: バックテスティング, VaR
- 初出: S3-5

<a id="backtesting"></a>
### バックテスティング（backtesting）
- 定義: 実現損益が VaR 予測を超えた回数（例外）を数え、モデルの妥当性を事後検証すること。
  例外は確率 $1-\alpha$ で独立に起こるはずで、期待例外数は $(1-\alpha)n$。検出力を確保するため
  規制上は少なくとも約250営業日を要求する。
- 数式: 例外数 $x=\#\{t:\ \mathrm{loss}_t > \mathrm{VaR}_t\}$、期待値 $(1-\alpha)n$
- 関連: Kupiec検定, VaR
- 初出: S3-5
