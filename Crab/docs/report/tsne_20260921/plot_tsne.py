"""CPU-only deterministic report plots from frozen, regression-checked features."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import normalize
import sklearn

HERE = Path(__file__).resolve().parent
COLORS = ['#D88722','#8357A5','#2B8594']
LABELS = ['E 興奮','U 不自信','N 中性']
NAMES = ['原 N3｜無 TTS','A25｜新增 157 筆 U TTS',
         'A50｜新增 314 筆 U TTS','A75｜新增 471 筆 U TTS']
ARMS = ['n3_reference','n3a25','n3a50','n3a75']
PARAMS = dict(n_components=2, perplexity=30, learning_rate='auto', init='pca',
              random_state=42, n_iter=1500, early_exaggeration=12,
              metric='euclidean', method='barnes_hut', angle=0.5, n_jobs=2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def embed(features):
    if not np.isfinite(features).all(): raise ValueError('Non-finite features')
    x = normalize(features.astype(np.float32), norm='l2')
    pca = PCA(n_components=50, svd_solver='full')
    x50 = pca.fit_transform(x)
    reducer = TSNE(**PARAMS)
    coords = reducer.fit_transform(x50)
    assert np.isfinite(coords).all()
    return coords, {'input_shape':list(x.shape), 'pca_variance_retained':float(pca.explained_variance_ratio_.sum()),
        'kl_divergence':float(reducer.kl_divergence_),
        'trustworthiness_k15_vs_normalized_original':float(trustworthiness(x, coords, n_neighbors=15)),
        'parameters':PARAMS,'preprocessing':'row L2 normalization; PCA 50, full SVD; no labels supplied'}


def save(fig,stem):
    for ext in ['png','svg','pdf']:
        fig.savefig(HERE/'assets'/f'{stem}.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def csv_out(path,headers,rows):
    with path.open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f); writer.writerow(headers); writer.writerows(rows)


def axes_style(ax):
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_color('#CBD5E1')


def mapping_plot(source):
    data=np.load(source/'mapping_features.npz',allow_pickle=False)
    meta=json.loads((HERE/'mapping_labels.json').read_text())
    assert data['ids'].tolist()==[r['id'] for r in meta]
    coords,diagnostics=embed(data['features'])
    original=np.array([r['original_display'] for r in meta])
    mapped=np.array([r['label'] for r in meta])
    assert np.bincount(mapped).tolist()==[808,628,1618]
    categories=sorted(set(original),key=lambda c:(c=='最高票並列',-int((original==c).sum()),c))
    palette={c:plt.get_cmap('tab20')(i) for i,c in enumerate(categories)}
    palette['最高票並列']='#929AAA'
    fig,axes=plt.subplots(1,2,figsize=(15,7))
    for c in categories:
        mask=original==c
        axes[0].scatter(*coords[mask].T,s=10,alpha=.65,c=[palette[c]],linewidths=0,
                        label=f'{c} ({mask.sum():,})',rasterized=True)
    for i in range(3):
        mask=mapped==i
        axes[1].scatter(*coords[mask].T,s=10,alpha=.65,c=COLORS[i],linewidths=0,
                        label=f'{LABELS[i]} ({mask.sum():,})',rasterized=True)
    axes[0].set_title('映射前：原始細情緒最高票摘要',fontsize=14,pad=13)
    axes[1].set_title('映射後：現行 E／U／N 合票結果',fontsize=14,pad=13)
    for ax in axes:
        axes_style(ax)
        ax.set_xlim(coords[:,0].min()-3,coords[:,0].max()+3)
        ax.set_ylim(coords[:,1].min()-3,coords[:,1].max()+3)
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.10),ncol=4,fontsize=9,frameon=False,markerscale=2)
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.10),ncol=3,fontsize=10,frameon=False,markerscale=2)
    fig.suptitle('NNIME Train 映射前後｜同一批 3,054 支音訊、完全相同座標',fontsize=18,y=.99)
    fig.text(.5,.025,'固定預訓練 XLS-R 語音特徵；未用情緒標籤降維。此圖呈現標籤合併，不表示分類準確度提升。',ha='center',fontsize=10,color='#475569')
    fig.subplots_adjust(top=.86,bottom=.26,wspace=.15)
    save(fig,'nnime_mapping_tsne')
    csv_out(HERE/'coordinates'/'mapping.csv',
            ['sample_id','tsne_1','tsne_2','original_top_labels','original_display','mapped_label'],
            [[r['id'],float(xy[0]),float(xy[1]),' / '.join(r['original_top_labels']),r['original_display'],LABELS[r['label']]] for r,xy in zip(meta,coords)])
    diagnostics['original_display_counts']={c:int((original==c).sum()) for c in categories}
    diagnostics['mapped_counts']=np.bincount(mapped).tolist()
    diagnostics['original_top_tie_count']=int((original=='最高票並列').sum())
    return diagnostics


def model_plots(source):
    fig,axes=plt.subplots(2,2,figsize=(13,11))
    rows=[]; details={}; ids_reference=None
    for index,(arm,ax) in enumerate(zip(ARMS,axes.flat)):
        data=np.load(source/(arm+'_features.npz'),allow_pickle=False)
        metrics=json.loads((source/(arm+'_metrics.json')).read_text())
        assert metrics['regression_passed'] is True
        if ids_reference is None: ids_reference=data['ids'].tolist()
        assert data['ids'].tolist()==ids_reference
        assert np.bincount(data['labels']).tolist()==[181,165,292]
        coords,diagnostics=embed(data['features'])
        for label in range(3):
            mask=data['labels']==label
            ax.scatter(*coords[mask].T,s=17,c=COLORS[label],alpha=.7,linewidths=0,rasterized=True)
        axes_style(ax)
        ax.set_title(f'{NAMES[index]}\nMacro-F1 {metrics["macro_f1"]*100:.2f}%｜Accuracy {metrics["accuracy"]*100:.2f}%',fontsize=13,pad=10)
        csv_out(HERE/'coordinates'/(arm+'.csv'),['sample_id','tsne_1','tsne_2','true_label','predicted_label'],
                [[sid,float(xy[0]),float(xy[1]),LABELS[int(y)],LABELS[int(p)]]
                 for sid,xy,y,p in zip(data['ids'],coords,data['labels'],data['probabilities'].argmax(1))])
        rows.append([arm,metrics['macro_f1'],metrics['accuracy']])
        details[arm]={'projection':diagnostics,'metrics':metrics}
    fig.suptitle('加入 TTS 訓練前後的模型表徵｜同一批 NNIME 真人 Test 638 筆',fontsize=17,y=.99)
    handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=9,label=l)
             for c,l in zip(COLORS,LABELS)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.045),ncol=3,frameon=False)
    fig.text(.5,.018,'每格獨立 t-SNE、相同參數；座標不可跨模型比較位移或距離。色彩＝真人標籤；Test 沒有 TTS。',ha='center',fontsize=10,color='#475569')
    fig.subplots_adjust(top=.89,bottom=.12,hspace=.33,wspace=.13)
    save(fig,'tts_model_comparison_tsne')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    x=np.arange(4)
    for ax,col,title,color in zip(axes,[1,2],['Macro-F1（三類等權）','Accuracy（整體正確率）'],['#8357A5','#2B8594']):
        vals=np.array([r[col]*100 for r in rows])
        ax.plot(x,vals,'o-',color=color,lw=2,ms=7)
        ax.axhline(vals[0],ls='--',color='#94A3B8',lw=1)
        for xx,yy in zip(x,vals): ax.annotate(f'{yy:.2f}%',(xx,yy),xytext=(0,9),textcoords='offset points',ha='center',fontsize=11)
        ax.set_xticks(x,['無 TTS','＋U25%','＋U50%','＋U75%']); ax.set_ylim(55,64)
        ax.set_title(title,fontsize=13); ax.set_ylabel('百分比 (%)'); ax.grid(axis='y',alpha=.15)
    fig.suptitle('同一 Test 的量化對照｜Macro-F1 改善不等於 Accuracy 提升',fontsize=15)
    fig.text(.5,.01,'歷史 A、seed42、最佳 Dev checkpoint；非新 full10、非多 seed 平均，未做顯著性檢定。',ha='center',fontsize=9,color='#475569')
    fig.tight_layout(rect=(0,.06,1,.94)); save(fig,'test_metrics_context')
    csv_out(HERE/'coordinates'/'test_metrics.csv',['arm','macro_f1','accuracy'],rows)
    return details


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args(); source=args.source.resolve()
    status=json.loads((source/'status.json').read_text())
    if status['state']!='complete': raise RuntimeError('Feature extraction not complete')
    if (HERE/'summary.json').exists(): raise RuntimeError('Refusing to replace existing plots')
    for folder in ['assets','coordinates','features']: (HERE/folder).mkdir(exist_ok=True)
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,
                         'svg.fonttype':'none','pdf.fonttype':42,'font.size':11})
    manifest=json.loads((HERE/'manifest.json').read_text())
    assert manifest==json.loads((source/'manifest.json').read_text())
    copied={}
    for name in ['mapping_features.npz']+[a+s for a in ARMS for s in ['_features.npz','_metrics.json']]:
        src=source/name; dst=HERE/'features'/name
        if dst.exists(): raise RuntimeError('Existing feature output '+str(dst))
        shutil.copy2(src,dst); copied[name]=sha(dst)
    start=time.monotonic()
    print('Rendering fixed-coordinate NNIME mapping',flush=True)
    mapping=mapping_plot(HERE/'features')
    print('Rendering all 4 historical models, no checkpoint selection',flush=True)
    models=model_plots(HERE/'features')
    summary={'source':str(source),'training':False,'mapping':mapping,'models':models,
             'feature_sha256':copied,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__,
             'plot_script_sha256':sha(Path(__file__)),'elapsed_seconds':time.monotonic()-start}
    dump(HERE/'summary.json',summary)
    print('Complete; PNG / SVG / PDF, coordinates and provenance saved.',flush=True)


if __name__=='__main__': main()
