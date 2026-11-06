"""Static chart figures with consistent colours and clearly labelled axes."""
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from modeling import clean, FEATURES

sns.set_theme(style='whitegrid', palette=['#1769e8','#ed9c42'], font_scale=1.0)

def png(fig):
    out=BytesIO()
    fig.savefig(out,format='png',dpi=150,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return out.getvalue()

def correlation(data):
    fig,ax=plt.subplots(figsize=(10,8))
    sns.heatmap(clean(data).corr(numeric_only=True),annot=True,fmt='.2f',cmap='rocket',vmin=-1,vmax=1,square=True,ax=ax)
    ax.set_title('Correlation between dataset variables',pad=20,weight='bold')
    return png(fig)

def pairs(data, columns):
    frame=clean(data).copy()
    frame['Outcome group']=frame['Outcome'].map({0:'No diabetes',1:'Diabetes'})
    grid=sns.pairplot(frame,vars=columns,hue='Outcome group',hue_order=['No diabetes','Diabetes'], palette={'No diabetes':'#1769e8','Diabetes':'#ed9c42'},diag_kind='hist',
        plot_kws={'s':15,'alpha':.55,'edgecolor':'none'},diag_kws={'alpha':.5},height=2.1)
    return png(grid.figure)

def distribution(data):
    fig,ax=plt.subplots(figsize=(8,3.5))
    counts=data['Outcome'].value_counts().reindex([0,1],fill_value=0)
    bars=ax.bar(['No diabetes (0)','Diabetes (1)'],counts,color=['#1769e8','#ed9c42'],width=.5)
    ax.bar_label(bars,padding=5)
    ax.set(ylabel='Number of records',title='Outcome distribution',ylim=(0,counts.max()*1.18))
    sns.despine(ax=ax)
    return png(fig)

def confusion(cm, name):
    fig,ax=plt.subplots(figsize=(6,4.5))
    labels=np.array([[f'True negatives\n{cm[0,0]}',f'False positives\n{cm[0,1]}'],
                     [f'False negatives\n{cm[1,0]}',f'True positives\n{cm[1,1]}']])
    sns.heatmap(cm,annot=labels,fmt='',cmap='rocket',square=True,ax=ax,
        xticklabels=['No diabetes','Diabetes'],yticklabels=['No diabetes','Diabetes'])
    ax.set(xlabel='Predicted outcome',ylabel='Actual outcome',title=f'{name}: test confusion matrix')
    return png(fig)

def comparison(results):
    fig,ax=plt.subplots(figsize=(10,5))
    names=list(results); y=np.arange(len(names)); h=.35
    ax.barh(y-h/2,[results[n]['cv_accuracy']*100 for n in names],h,label='Tuning CV accuracy',color='#1769e8')
    ax.barh(y+h/2,[results[n]['test_accuracy']*100 for n in names],h,label='Held-out test accuracy',color='#ed9c42')
    ax.set(yticks=y,yticklabels=names,xlim=(0,100),xlabel='Accuracy (%)',title='Model comparison')
    ax.legend(loc='lower left'); sns.despine(ax=ax)
    return png(fig)

def roc_chart(results):
    fig,ax=plt.subplots(figsize=(8,5))
    for name,r in results.items():
        ax.plot(*r['roc'],label=f"{name} (AUC {r['auc']:.2f})")
    ax.plot([0,1],[0,1],'--',color='#9ca9bb')
    ax.set(xlabel='False positive rate',ylabel='True positive rate (recall)',title='ROC curves on held-out test data')
    ax.legend(fontsize=8); sns.despine(ax=ax)
    return png(fig)
