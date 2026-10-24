# 第二章　文獻回顧

<!--
寫作依據：專題計畫提案書.pdf 第五部分的敘事與模組拆解方式。
技術內容依 Crab.pdf（arXiv:2603.23673v1）；不是原提案文字的直接替換。
圖像、公式與數值來源對照見 assets/crab_literature_review/README.md。
此檔獨立供結報整合，未替換現有第二章情緒辨識機制。
-->

## 2.1 語音情緒辨識之演進與研究問題

語音情緒辨識（Speech Emotion Recognition, SER）旨在分析語音中所呈現的情緒資訊，使人機互動系統除了理解說話內容，也能掌握使用者的表達狀態。根據 Ueda 等人 [1] 的文獻整理，SER 的特徵建模已由人工設計的聲學特徵，逐步轉向利用大規模未標註語音訓練的自監督學習（Self-Supervised Learning, SSL）模型。wav2vec 2.0、HuBERT 與 WavLM 等模型能從原始音訊學習具有上下文資訊的表徵，並透過下游微調應用於情緒分類。另一方面，文字內容可補充語音特徵所未直接呈現的語意資訊，因此結合聲音與文字的雙模態方法，亦成為語音情緒辨識的重要研究方向。

然而，自然對話中的情緒並非均勻分布，中性語音通常占較高比例，部分情緒則較少出現。僅在表演式或較平衡資料上取得良好分類成績，仍不足以反映模型對自然情緒與少數類別的辨識能力。此外，若訓練目標只作用於最後的分類輸出，中間層表徵便缺少直接的情緒區辨引導。針對上述問題，Ueda 等人 [1] 提出 CRAB（Contrastive Representation and Multimodal Aligned Bottleneck），結合語音與文字的跨模態融合，以及多層對比監督（Multi Layer Contrastive Supervision, MLCS），使情緒資訊在單模態建模、跨模態互動及最終融合等階段持續受到訓練目標引導。

## 2.2 CRAB：語音與文字之雙模態融合架構

CRAB 以雙模態跨模態 Transformer（Cross-Modal Transformer, CMT）為核心，分別使用 WavLM Large 與 RoBERTa Large 擷取語音及文字表徵，再透過時序建模與跨模態注意力整合兩側資訊。其設計並非只將兩種特徵直接串接，而是在融合前讓聲學與語意表徵彼此參照，以建立用於情緒分類的共同表徵 [1]。

![CRAB原論文的雙模態架構與對比監督分支](assets/crab_literature_review/crab_original_architecture.png)

圖 2-1　CRAB 模型架構。灰色區域依序為單模態區塊、跨模態區塊與分類區塊；綠色標示為加入對比監督分支的位置。（資料來源：[1]，Fig. 1，第 5 頁。）

如圖 2-1 所示，模型可拆解為以下四個關鍵部分：

1. **預訓練特徵擷取（Pre-trained Representation Extraction）**

   語音側以 WavLM Large 取得音框層級（frame-level）的聲學表徵；文字側以 RoBERTa Large 取得 token 層級的語意表徵。原論文將兩者分別記為 $S^{F_s\times D_s}\in\mathbb{R}^{F_s\times D_s}$ 與 $T^{L\times D_t}\in\mathbb{R}^{L\times D_t}$，其中 $F_s$ 為音框數、$L$ 為文字 token 數，$D_s$ 與 $D_t$ 為兩側的特徵維度。此設計保留語音與文字各自的序列形式，供後續時序建模與跨模態互動使用。

2. **單模態時序建模（Unimodal Block）**

   對應圖中左側區塊，兩側特徵先通過全連接層（Fully Connected Layer, FC），投影至共同的隱藏維度 $h=512$，再以層正規化（Layer Normalization）處理特徵尺度。接著，模型分別使用雙向閘控循環單元（Bidirectional Gated Recurrent Unit, bi-GRU）整合序列前後文。由於正向與反向隱藏狀態相互串接，兩側輸出特徵維度皆為 $2h$，但音框數與文字 token 數仍可不同。

3. **跨模態注意力與池化（Cross-Modal Attention and Pooling）**

   對應圖中中央區塊，模型以其中一種模態作為查詢（Query），另一種模態提供鍵與值（Key、Value），並在語音查詢文字、文字查詢語音兩個方向進行互動。原論文使用注意力機制進行表徵對齊，並將注意力頭數設為 1，以降低過擬合風險。各方向的注意力輸出再與對應的 bi-GRU 輸出進行殘差相加（Residual Addition），使融合後特徵同時保有模態自身資訊及另一模態提供的線索。

   為將可變長度序列轉為固定維度向量，CRAB 採用注意力池化（Attention Pooling）。其權重與聚合方式如下 [1]：

   <!-- CRAB Eq. (3), p.4；保留 sqrt(D) 尺度。 -->
   $$
   w_i=
   \frac{
     \exp\!\left(\dfrac{r_i^{\top}m}{\sqrt{D}}\right)
   }{
     \displaystyle\sum_{l=1}^{L}
     \exp\!\left(\dfrac{r_l^{\top}m}{\sqrt{D}}\right)
   }.
   \tag{2-1}
   $$

   <!-- CRAB Eq. (4), p.4。 -->
   $$
   r=\sum_{i=1}^{L}w_i r_i.
   \tag{2-2}
   $$

   其中，$r_i\in\mathbb{R}^{D}$ 為第 $i$ 個 token 或音框的特徵，$m\in\mathbb{R}^{D}$ 為可學習參數，$w_i$ 為正規化後的注意力權重，$r$ 為池化後的向量；池化式中的 $L$ 表示所處理序列的長度。相較於讓所有位置具有相同權重，此機制可依訓練結果調整各位置對聚合表徵的貢獻。語音與文字的池化結果最後進行串接，形成 $C\in\mathbb{R}^{D_f}$，其中 $D_f=4h$。

4. **情緒分類（Classifier Block）**

   對應圖中右側區塊，串接後的向量依序通過 Layer Normalization、FC、ReLU、圖示中的 Dropout 與最後的 FC，輸出各情緒類別的 logits。此區塊負責將融合表徵轉換為分類結果；其第一個 FC 亦是多層對比監督的其中一個作用位置，使分類前的融合資訊同時受到情緒分類與對比學習引導。

## 2.3 多正樣本對比學習與多層對比監督

CRAB 的核心貢獻在於：不只要求最終輸出正確，也直接引導中間表徵具備情緒區辨性。其方法結合多正樣本對比學習（Multi-Positive Contrastive Learning, MPCL）與多層監督，讓模型在不同處理階段學習情緒相關的表徵結構 [1]。

### 2.3.1 多正樣本對比學習（MPCL）

對比學習透過比較樣本在特徵空間中的關係，促使相關樣本具有較接近的表徵，並區分不同類別。CRAB 採用的 MPCL 可將同一情緒類別的多個候選視為正樣本，而非只指定單一配對；其運作可分為以下三項定義：

1. **候選相容性分布（Contrastive Categorical Distribution）**

   設 $a$ 為錨點（anchor），$b_i$ 為第 $i$ 個候選表徵，候選總數為 $K$。兩者皆經過 $\ell_2$ 正規化後，以內積衡量相似性，再轉換為分布 $q$：

   <!-- CRAB Eq. (5), p.5。 -->
   $$
   q_i=
   \frac{\exp(a\cdot b_i/\tau)}
   {\displaystyle\sum_{j=1}^{K}\exp(a\cdot b_j/\tau)}.
   \tag{2-3}
   $$

   其中，$\tau\in\mathbb{R}_{+}$ 為溫度參數，用於控制分布的集中程度。$q_i$ 表示候選 $b_i$ 相對於其他候選，與錨點 $a$ 的相容程度。

2. **同情緒目標分布（Target Distribution）**

   模型以情緒標籤定義正樣本，將目標機率平均分配給與錨點具有相同情緒標籤的候選：

   <!-- CRAB Eq. (6), p.5。 -->
   $$
   c_i=
   \frac{\mathbb{1}_{\mathrm{match}}(a,b_i)}
   {\displaystyle\sum_{j=1}^{K}
   \mathbb{1}_{\mathrm{match}}(a,b_j)}.
   \tag{2-4}
   $$

   其中，$\mathbb{1}_{\mathrm{match}}(a,b_i)$ 為指示函數：當 $a$ 與 $b_i$ 的情緒標籤相同時為 1，否則為 0。分母為候選集合中的同情緒正樣本數，因此此定義使用具有正樣本的候選集合。此處的 $c$ 是由情緒類別建立的對比分布，而不是標註者投票比例所形成的情緒軟標籤。

3. **對比損失（MPCL Loss）**

   MPCL 以目標分布 $c$ 與模型分布 $q$ 的交叉熵作為損失：

   <!-- CRAB Eq. (7), p.5。 -->
   $$
   \mathcal{L}_{\mathrm{MPCL}}
   =-\sum_{i=1}^{K}c_i\log q_i.
   \tag{2-5}
   $$

   最小化此損失，可引導模型提升同情緒候選的相對相容性，形成有利於情緒區分的特徵空間。由於目標分布可同時涵蓋多個同類樣本，模型可利用一組候選中的多個正樣本共同提供訓練訊號。

### 2.3.2 多層對比監督（MLCS）

為將上述對比目標作用於模型內部，CRAB 在圖 2-1 的綠色位置加入對比監督分支（Contrastive Supervision Leg, CSL）。每個分支由「FC → ReLU → FC」組成，將所在位置的中間特徵投影至對比嵌入空間，再以 MPCL 進行監督。原論文共設置五個作用位置 [1]：

1. **語音 bi-GRU 輸出：** 對序列先進行平均池化，再送入 CSL，強化語音側的情緒表徵。
2. **文字 bi-GRU 輸出：** 同樣以平均池化取得分支輸入，引導文字側的情緒區辨。
3. **語音注意力池化輸出：** 對已融合文字線索的語音向量施加對比監督。
4. **文字注意力池化輸出：** 對已融合聲學線索的文字向量施加對比監督。
5. **分類區塊第一個 FC：** 在兩模態串接後的融合特徵上加入對比目標。

此設計將對比監督限制於 CRAB 的雙模態 CMT 架構內，並非在 WavLM 與 RoBERTa 編碼器的每一層都增加分支。CSL 是訓練時的輔助監督，因此不需為這些分支增加推論階段的參數。

最終訓練目標結合分類交叉熵與各監督位置的平均 MPCL 損失：

<!-- CRAB Eq. (8), p.5。 -->
$$
\mathcal{L}_{\mathrm{ser}}
=
\mathcal{L}_{\mathrm{CE}}
\!\left(m_K(\mathcal{X}),\mathcal{Y}\right)
+\alpha\cdot\frac{1}{|\Omega|}
\sum_{i\in\Omega}
\mathcal{L}_{\mathrm{MPCL}}
\!\left(m_i(\mathcal{X}),\mathcal{Y}\right).
\tag{2-6}
$$

其中，$\mathcal{X}$ 為輸入資料、$\mathcal{Y}$ 為情緒標籤，$m_K$ 表示最終分類輸出，$m_i$ 表示對應監督位置的表徵，$\Omega$ 為加入 CSL 的位置集合，$|\Omega|=5$。$\alpha$ 為對比損失權重，用於調整分類學習與中間表徵學習的相對貢獻。式（2-6）中的 $K$ 用於表示最終層，與式（2-3）至式（2-5）中候選數量的 $K$ 依各式語境區分。

## 2.4 類別不平衡處理與訓練策略

自然語音中的情緒樣本量通常不均，若各類樣本採用相同的損失權重，數量較多的類別可能主導訓練。CRAB 因此使用加權交叉熵（Weighted Cross-Entropy, WCE）作為分類損失，並依類別頻率的倒數設定權重 [1]：

<!-- CRAB Eq. (9), p.5。 -->
$$
w_{\mathrm{prior}}
=
\left\{
w_j=\frac{N}{N_j}
\;\middle|\;
j\in[1,\ldots,E]
\right\}.
\tag{2-7}
$$

其中，$N$ 為樣本總數，$N_j$ 為第 $j$ 類的樣本數，$E$ 為情緒類別總數。當某類樣本較少時，其權重較高，使模型在訓練時增加對該類分類錯誤的重視。式（2-6）的分類項在實際訓練中採用此加權方式，並與多層對比目標共同最佳化。

在最佳化策略上，原論文使用 AdamW 與餘弦退火學習率排程（Cosine Annealing），並採分層學習率（Hierarchical Learning Rate）：CRAB 新增架構的學習率設為 $1\times10^{-5}$，WavLM Large 與 RoBERTa Large 則採較小的 $1\times10^{-6}$。此設計讓新加入的融合及分類模組能進行較大幅度的調整，同時以較小步幅更新預訓練表徵。原論文實驗訓練 20 個 epochs，batch size 為 32、梯度累積為 4，並使用單張 NVIDIA L40S GPU [1]。

## 2.5 原論文之資料集與評估結果

CRAB 使用三個具有不同情緒自然程度的資料集進行評估，藉此觀察模型在表演式語音、情境式對話與自然語音中的表現 [1]：

1. **IEMOCAP：** 收錄演員的雙人互動，包含腳本與即興表演。原論文採用 angry、happy、neutral、sad 四類，並依常見做法將 happy 與 excited 合併。
2. **MELD：** 取自影集《Friends》的多方對話，包含七類情緒。原論文將其視為弱誘發／表演式（weakly-prompted/acted）情境，並使用官方 Train、Validation 與 Test 切分。
3. **MSP-Podcast 2.0：** 取自 Podcast 的自然語音，包含八類情緒。原論文分別評估已見說話者的 Test 1、未見說話者的 Test 2，以及未見說話者與不同錄音條件的 Test 3；其文字輸入在各 split 統一由 Canary-Qwen 產生。

原論文以加權平均召回率（Weighted Average Recall, WAR）及非加權平均召回率（Unweighted Average Recall, UAR）評估分類表現。WAR 等同整體分類正確率，會受到各類樣本量影響；UAR 為各類 recall 的等權平均，可反映模型對不同情緒類別的辨識情形。MSP-Podcast 2.0 Test 3 的官方平台另以 Macro-F1 評估各類 precision 與 recall 的綜合表現。主要結果整理如表 2-1。

表 2-1　CRAB 原論文之主要評估結果。（資料來源：[1]，Table II、VII、VIII；數值換算為百分比。）

| 資料集／評估分割 | 情緒類別數 | WAR | UAR | Macro-F1 |
|---|---:|---:|---:|---:|
| IEMOCAP | 4 | 73.48% | 75.85% | — |
| MELD Test | 7 | 62.57% | 55.65% | — |
| MSP-Podcast 2.0 Test 1 | 8 | 52.32% | 43.90% | — |
| MSP-Podcast 2.0 Test 2 | 8 | 48.61% | 33.82% | — |
| MSP-Podcast 2.0 Test 3 | 8 | 42.12% | — | 43.13% |

表中「—」表示該原始表格未列出此指標，各資料集的類別與評估條件不同。於原論文所列比較中，CRAB 在 IEMOCAP、MELD 與 MSP-Podcast Test 1／2 的 UAR 均優於表中基準方法，而 MSP-Podcast Test 3 的 Macro-F1 為 43.13%。作者亦透過混淆矩陣分析少數情緒的辨識情形，強調模型評估不能只依多數類別所主導的整體正確率。

## 2.6 消融實驗：雙模態與多層監督的作用

為釐清各組件的貢獻，CRAB 原論文進一步比較模態組合、監督位置、損失設計與對比權重。其消融實驗可從以下三個方向理解 [1]。

### 2.6.1 語音與文字的互補性

在 MELD 上，僅使用語音的 UAR 為 31.76%，僅使用文字為 53.81%，結合語音與文字則達 55.65%；在 IEMOCAP 上，對應 UAR 分別為 65.18%、71.98% 與 75.85%（原論文 Table III）。雙模態組合在這兩組消融中取得最高 UAR，顯示文字內容與聲學表徵對情緒辨識具有互補作用，也提供採用跨模態融合架構的實驗依據。

### 2.6.2 多層監督與損失設計

原論文以 MELD 比較只使用分類交叉熵、只在分類區塊加入 MPCL、在多層加入交叉熵監督，以及完整 MLCS 等訓練方式。其結果如表 2-2。

表 2-2　不同訓練目標與策略於 MELD 的消融結果。（資料來源：[1]，Table IV；MLCS 權重 $\alpha=2$。）

| 訓練設定 | WAR | UAR |
|---|---:|---:|
| 僅分類交叉熵（CE） | 57.97% | 52.53% |
| CE＋分類區塊 MPCL | 60.50% | 53.75% |
| 多層交叉熵監督（MLS＋CE） | 60.77% | 55.39% |
| MLCS，將 MPCL 改為 SCL | 63.37% | 55.34% |
| MLCS，不使用分層學習率 | 59.04% | 51.82% |
| 完整 MLCS | 62.57% | 55.65% |

完整 MLCS 相較僅使用 CE，UAR 由 52.53% 提升至 55.65%，增加 3.12 個百分點；相較只在分類區塊施加 MPCL，亦有較高的 UAR。另一方面，將 MPCL 換成監督式對比學習（Supervised Contrastive Learning, SCL）雖得到較高 WAR，UAR 仍略低於完整方法。移除分層學習率後，UAR 則降至 51.82%。這些結果顯示，對比目標的作用位置與預訓練編碼器的更新方式，皆是原論文訓練設計中的重要因素。

### 2.6.3 對比損失權重

![CRAB原論文的對比損失權重消融](assets/crab_literature_review/crab_original_alpha_ablation.png)

圖 2-2　不同對比損失權重 $\alpha$ 下的 UAR 與 WAR。綠色實線為 UAR、藍色虛線為 WAR；水平參考線為僅使用 CE 的模型與 Medusa 之 UAR。（資料來源：[1]，Fig. 3，第 8 頁；左右兩側座標軸分別對應 UAR 與 WAR。）

原論文比較 $\alpha=0.25$ 至 $3.00$ 的設定，並以 UAR 作為權重選擇依據。UAR 在 $\alpha=2.00$ 時最高，而 WAR 的最高點位於 $\alpha=2.75$，因此作者採用 $\alpha=2.00$。此結果說明，最佳權重會受到評估指標影響；若研究目的包含少數情緒辨識，便需要同時檢視類別平衡指標，而非只追求整體正確率。

## 2.7 本專題之方法選用依據

綜合上述文獻，CRAB 提供三項與面試情緒辨識相關的設計基礎：其一，以語音與文字雙模態結合說話方式與回答內容；其二，透過多層對比監督，使單模態與融合表徵皆受到情緒區辨目標引導；其三，結合類別加權與分層學習率，處理自然情緒資料的不平衡及預訓練模型適配問題。因此，本專題以 CRAB 的雙模態融合與 MLCS 作為情緒辨識模組的架構基礎。

文獻回顧所述的 WavLM／RoBERTa、資料集、公式與實驗數值屬於 CRAB 原論文；本專題的中文編碼器替換、LoRA 微調、三類情緒映射與 TTS 擴增，則於研究方法及實驗章節說明。

## 參考文獻

[1] L. H. Ueda, J. G. T. Lima, and P. D. P. Costa, “Crab: Multi Layer Contrastive Supervision to Improve Speech Emotion Recognition Under Both Acted and Natural Speech Condition,” *arXiv preprint*, arXiv:2603.23673v1, Mar. 24, 2026. [原論文版本](https://arxiv.org/abs/2603.23673v1)；[本機 PDF](../../Crab.pdf)。
