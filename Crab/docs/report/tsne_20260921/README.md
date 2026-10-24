# NNIME 映射與 TTS 訓練前後：t-SNE 視覺化

本文件是結報用的探索性特徵分析；不新增訓練、不追加 seed、不更改標籤或權重。
完成紀錄、實際投影診斷與推論回歸檢查見 [summary.json](summary.json)。

**已完成：** 2張t-SNE主圖＋1張量化對照圖，均提供PNG／SVG／PDF；實際圖像已逐張檢視。4組模型的638筆Test預測類別全部與歷史紀錄一致，機率最大絕對差均約1.19×10⁻⁷。GPU任務正常退出0，耗時214.63秒；SSAMBA已由queue自動恢復。

## 1. NNIME 映射前後

![NNIME 映射前後 t-SNE](assets/nnime_mapping_tsne.png)

**建議圖說：** NNIME Train 同一批 3,054 支音訊之 t-SNE 特徵分布。使用未套用 LoRA 的預訓練 XLS-R-300m 最後一層音框平均特徵；左圖以六位標註者原始細情緒的最高票摘要上色，右圖以現行映射及合票產生的 E／U／N 上色。兩圖使用完全相同座標。這是標籤表達方式的比較，不是映射前後重新訓練的分類效果。

### 範圍與標註處理

- 使用現行保留的 Train 3,054 筆（E808／U628／N1,618），不是全部原始事件；未畫被排除的音訊，也未混入 Dev／Test。
- NNIME 原始標註是六位評分者、每人可多標籤。原子依既有程式以半形冒號拆分，沒有把檔名中的情境名稱當成真實情緒。
- 左圖只有在某個原始細標籤票數唯一最高時，才以該標籤上色。最高票並列的 **522 筆**以「最高票並列」呈現，不擅自選一個標籤；這是原始細標籤平票，不是三類映射的平票數。
- 右圖逐筆核對現行 v3 映射結果；它是「先映射所有標註，再合票」，不是把左圖的最高票標籤直接轉成三類。因此左圖的「生氣」不一定意味著整支音訊要排除。
- 所有原始票數、並列標籤及映射標籤保留於 [mapping_labels.json](mapping_labels.json)，圖例未隱藏少數細標籤。
- 以不含 N3 三類微調的固定語音編碼器畫圖，避免把分類器學到的三類結構當成映射本身的證據。此表徵仍有預訓練模型偏好，不是唯一的「真實情緒空間」。

## 2. 未加 TTS／加入 TTS 訓練後的模型表徵

![四組歷史模型 t-SNE](assets/tts_model_comparison_tsne.png)

**建議圖說：** 原 N3 與加入不同劑量 U TTS 的歷史 A25／A50／A75，對同一批 NNIME 真人 Test 638 筆所產生的分類前 512 維特徵。顏色表示真人標籤；每格標示該 checkpoint 的 Test Macro-F1 及 Accuracy。四格分別以相同 t-SNE 參數降維，座標軸不具有跨模型可比的方向或距離意義。

| 項目 | 固定方式 |
|---|---|
| 比較模型 | 原 N3＋歷史 A25／A50／A75，全部呈現，不按 Test 挑最好的一組 |
| A 版本 | 2026-09-09 Long12、seed42、最多10輪／patience5；不是後來 full10 A |
| 模型權重 | 原先已由 Dev 選定的最佳權重；不重新選 epoch |
| 訓練關係 | 原 N3 與各 A 組獨立由 V1 初始化；不是在原 N3 權重上繼續加 TTS 訓練 |
| 特徵位置 | 雙模態分類頭第一個 FC 後的 ReLU，512維；不是三維 logits 或預測機率 |
| 畫圖音訊 | 相同638筆真人 Test；E181／U165／N292，沒有TTS混入Test |
| 前處理 | 沿用各模型保存的 Train 正規化；16kHz、前12秒、文字128 tokens |
| 推論 | batch1、FP32、eval、no_grad、0 workers；不做optimizer更新 |
| 回歸檢查 | 與既有逐筆預測比較：預測類別完全一致、機率最大容許絕對差1e-5；失敗即停止 |

![同一Test的Macro-F1與Accuracy](assets/test_metrics_context.png)

**重要解讀：** 本輪不是所有劑量、所有指標都改善。歷史 A25／A75 的 Macro-F1 高於原 N3，但三組 A 的 Accuracy 均未高於原 N3。t-SNE 只能輔助觀察表徵，不可把較漂亮的群集當作效能、顯著性或 TTS 因果效果的證明；本圖也不是新 full10 A 的結果。

### 本次圖像的初步判讀

| 圖 | 實際觀察 | 適合結報的說法 |
|---|---|---|
| NNIME 映射前後 | 在固定的預訓練語音投影中，細情緒與三類標籤都存在明顯混合，映射後未形成三個乾淨分離的群集。 | 「映射將細情緒標註整合為三類任務，但三類在此聲學表徵中仍有重疊。」 |
| 四模型 Test 表徵 | 各模型皆可看到部分E集中區域，U與N仍有重疊；未見足以單憑圖像宣稱所有劑量一致改善的證據。 | 「以t-SNE觀察模型表徵，並以相同真人Test的Macro-F1及逐類指標判定辨識效果。」 |
| 量化對照 | N3／A25／A50／A75的Macro-F1為58.77／59.25／58.35／59.67%；Accuracy為61.76／61.60／59.87／60.82%。 | 「A25、A75的Macro-F1較原N3提高約0.48、0.91個百分點，但並非所有劑量或指標均提升。」 |

上述百分點差值使用未四捨五入的原值計算。圖中群集間空白、大小與方向受降維影響，不能拿來衡量改善幅度；也不能把Train預訓練語音圖與Test雙模態分類特徵圖直接當成同條件訓練前後比較。

## 3. 降維方法與重現

1. 每筆特徵做 L2 正規化，再以不使用標籤的 PCA 降至50維（full SVD，不whiten）。
2. t-SNE：2維、perplexity30、PCA初始化、learning_rate=auto、early_exaggeration12、1,500次迭代、Barnes–Hut angle0.5、random_state42、最多2個CPU執行緒。
3. 預先固定一組繪圖設定，未依圖形好看程度搜尋參數／seed。繪圖random_state不是追加模型訓練seed。
4. 映射兩格只計算一次投影；模型四格各自投影，不能比較某個點「移向左／右」或群集大小的絕對差異。
5. summary記錄PCA保留變異、t-SNE KL loss與相對原始正規化特徵的trustworthiness（k15）；這些是投影診斷，不是情緒辨識成績。

圖像提供 PNG、SVG及PDF；逐筆座標在 coordinates/；小型特徵快取在 features/。未輸出音訊或逐字稿。因涉及NNIME研究資料，所有產物暫存本機，不commit／push／公開託管。

## 4. Queue 與資料來源

- 本次queue任務：ee18f34e67bb4ac88a21548f875f099d；僅推論，urgent=true、pause_mode=none，讓SSAMBA在epoch邊界存檔讓位後執行。
- RAM限制：MemoryHigh16GiB／MemoryMax20GiB、禁swap、CPU200%；每個模型使用獨立子程序、依序釋放GPU。CPU繪圖另在受限服務執行，不占GPU。
- 本次每5秒取樣的queue cgroup RAM最高約5.73GiB、主機可用RAM最低約55.82GiB，最後OOM／OOM kill計數均為0；這是取樣結果，不是保證精確峰值。CPU繪圖另限4GiB RAM、CPU200%，約36.70秒完成。
- [特徵擷取程式](extract_features.py)、[繪圖程式](plot_tsne.py)、[queue設定](queue_job.json)、[凍結輸入與SHA256](manifest.json)。
- [歷史Test評估](../../../experiments/n3_phase_a_test_20260910/README.md)、[NNIME映射程式](../../../scripts/build_nnime_crab_csv.py)。
- [t-SNE原論文](https://jmlr.org/papers/v9/vandermaaten08a.html)、[scikit-learn t-SNE文件](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)。

本次不更改訓練實驗或W&B紀錄。特徵擷取重用已看過的Test作描述性分析，不應再用這些圖挑新的模型或調參。
