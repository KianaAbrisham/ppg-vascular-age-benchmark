"""Predict new rows using a saved fold checkpoint and its preprocessing."""
import argparse
from pathlib import Path
import pandas as pd
from . import engine
from .common import AGES,inverse,load_signals,read_json,represent,scale


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--signals',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();folder=Path(a.model_dir)
    meta=read_json(folder/'preprocessing.json')
    ids,raw=load_signals(a.signals,meta['config']['length'])
    x=scale(represent(raw,meta['kind'],meta['config']),meta['x_scale'])
    if list(x.shape[1:])!=meta['input_shape']: raise ValueError('Input shape differs from training.')
    engine.initialize(42)
    prediction=engine.predict(engine.load(folder),x)
    df=pd.DataFrame({'subject_id':ids})
    if meta['classification']:
        df['predicted_age_category']=AGES[prediction.argmax(axis=1)]
        for i,age in enumerate(AGES):df[f'p_age_{age}']=prediction[:,i]
    else:df['predicted_cfpwv_m_s']=inverse(prediction,meta['y_scale'])
    df['model_status']=meta['status']
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as file:df.to_csv(file,index=False)
    print(f'Saved {len(df)} predictions.')


if __name__=='__main__': main()
