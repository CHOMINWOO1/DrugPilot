"""Render the checked-in metric extract. Requires matplotlib; no model training occurs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(directory):
    directory=Path(directory)
    data=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.spines.left':False,'axes.titleweight':'bold',
                         'svg.fonttype':'none','savefig.facecolor':'#f6f8fb'})
    fig, axes=plt.subplots(1,len(data['panels']),figsize=(15,6.5),squeeze=False)
    fig.set_facecolor('#f6f8fb')
    colors=['#107d8d','#5260ac','#d99132','#bd5366','#607586','#82bcb2','#223d54']
    for ax,panel in zip(axes[0],data['panels']):
        ax.set_facecolor('#f6f8fb')
        y=np.arange(len(panel['labels']))
        kind=panel.get('kind','bar')
        if kind=='stacked':
            left=np.zeros(len(y))
            for index,series in enumerate(panel['series']):
                values=np.array(series['values'])
                bars=ax.barh(y,values,left=left,label=series['name'],color=colors[index],height=.6)
                for bar,value in zip(bars,values):
                    if value:
                        ax.text(bar.get_x()+bar.get_width()/2,bar.get_y()+bar.get_height()/2,
                                str(value),ha='center',va='center',color='white',fontsize=10,weight='bold')
                left+=values
            ax.legend(frameon=False,loc='lower right')
        elif kind=='interval':
            values=np.array(panel['values'])
            errors=np.array([values-np.array(panel['low']),np.array(panel['high'])-values])
            ax.errorbar(values,y,xerr=errors,fmt='o',color=colors[0],capsize=5,linewidth=2,markersize=7)
            ax.axvline(panel.get('reference',0),color='#8a99a8',linestyle='--',linewidth=1)
        else:
            values=panel['values']
            ax.barh(y,values,xerr=panel.get('error'),capsize=4,color=colors[:len(y)],height=.58)
            for index,value in enumerate(values):
                label=panel.get('value_labels',[f'{v:g}' for v in values])[index]
                offset=panel.get('error',[0]*len(values))[index]
                ax.text(value+offset,index,'  '+label,va='center',fontsize=10,color='#233d51')
            ax.margins(x=.25)
        ax.set_yticks(y,panel['labels'])
        ax.invert_yaxis()
        ax.tick_params(axis='y',length=0,pad=10)
        ax.set_title(panel['title'],loc='left',pad=18,fontsize=13)
        ax.set_xlabel(panel['xlabel'],labelpad=12)
        ax.grid(axis='x',alpha=.15)
        ax.set_axisbelow(True)
        if 'xlim' in panel:
            ax.set_xlim(panel['xlim'])
    fig.suptitle(data['title'],x=.03,y=.97,ha='left',fontsize=21,fontweight='bold',color='#18394d')
    fig.text(.03,.89,data['subtitle'],fontsize=11,color='#596e7e')
    fig.text(.03,.035,data['boundary'],fontsize=10,color='#596e7e',linespacing=1.5)
    fig.subplots_adjust(left=.17,right=.96,top=.76,bottom=.22,wspace=.65)
    fig.savefig(directory/'results.png',dpi=160)
    fig.savefig(directory/'results.svg')
    plt.close(fig)


if __name__=='__main__':
    render(Path(__file__).resolve().parent)
