"""Explicit experiment configuration with disjoint train, validation and test subjects."""
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from . import engine
from .common import (AGES,demo_data,digest,environment,fit_scale,inverse,load_data,metrics,
    plot_evaluation,represent,scale,split_data,write_json)
from .settings import CLASSIFICATION,ENGINE,MODELS,PROJECT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--signals'); parser.add_argument('--targets')
    parser.add_argument('--site',choices=['digital','radial','brachial'])
    parser.add_argument('--models',nargs='+',choices=MODELS)
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--output',required=True)
    parser.add_argument('--length',type=int,default=2000)
    parser.add_argument('--fs',type=float,default=500)
    parser.add_argument('--nperseg',type=int,default=76)
    parser.add_argument('--window',choices=['hann','hamming','tukey'],default='hamming' if ENGINE=='torch' else 'hann')
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--epochs',type=int,default=200 if ENGINE=='torch' else (100 if CLASSIFICATION else 500))
    parser.add_argument('--patience',type=int,default=20 if ENGINE=='torch' else (10 if CLASSIFICATION else 50))
    parser.add_argument('--batch-size',type=int,default=32 if CLASSIFICATION or ENGINE=='torch' else 16)
    parser.add_argument('--weights',choices=['imagenet','none'],default='imagenet')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.demo and (args.signals or args.targets): parser.error('--demo generates artificial inputs; omit supplied file paths.')
    if not args.demo and not (args.signals and args.targets and args.site): parser.error('Supply --signals, --targets and verified --site.')
    if min(args.epochs,args.patience)<1 or args.batch_size<2 or args.folds<2 or args.length<76:
        parser.error('Require epochs/patience>=1, batch-size/folds>=2 and length>=76.')
    if args.fs<=0 or not 2<=args.nperseg<=(512 if args.demo else args.length): parser.error('Invalid sampling rate or segment length.')
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    if args.demo:
        args.signals,args.targets=demo_data(root/'demo_input',CLASSIFICATION,args.seed)
        args.length,args.epochs,args.folds,args.weights,args.site=512,1,2,'none','synthetic'
    kinds=args.models or ([MODELS[0]] if args.demo else MODELS)
    ids,raw,y=load_data(args.signals,args.targets,args.length,CLASSIFICATION)
    splits=split_data(y,args.seed,args.folds,CLASSIFICATION,ENGINE=='torch')
    config={'length':args.length,'fs':args.fs,'nperseg':args.nperseg,'window':args.window,'image_size':64 if args.demo else 224}
    status='artificial_data_software_check' if args.demo else 'local_dataset_evaluation'
    write_json(root/'run.json',{'project':PROJECT,'status':status,'paper_reproduction_verified':False,
        'arguments':vars(args),'environment':environment(),'signals_sha256':digest(args.signals),'targets_sha256':digest(args.targets)})
    write_json(root/'splits.json',[{'fold':i,**{name:ids[indices].tolist() for name,indices in zip(['train','validation','test'],parts)}}
        for i,parts in enumerate(splits,1)])
    engine.initialize(args.seed)
    overview={}
    for kind in kinds:
        features=represent(raw,kind,config)
        rows,tables=[],[]
        for fold,(train,val,test) in enumerate(splits,1):
            print(f'{kind}: fold {fold}/{len(splits)}',flush=True)
            folder=root/kind/f'fold_{fold}'
            folder.mkdir(parents=True)
            xs=fit_scale(features[train]) if kind in ['spectrogram','cnn2d'] else {'mean':0.,'std':1.}
            ys={'mean':0.,'std':1.} if CLASSIFICATION else fit_scale(y[train])
            x=scale(features,xs); target=y if CLASSIFICATION else scale(y,ys)
            write_json(folder/'preprocessing.json',{'kind':kind,'config':config,'x_scale':xs,'y_scale':ys,
                'site':args.site,'status':status,'classification':CLASSIFICATION,'age_classes':AGES.tolist() if CLASSIFICATION else None,
                'input_shape':list(x.shape[1:])})
            model,history=engine.fit(kind,x,target,train,val,args)
            checkpoint=engine.save(model,folder)
            pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
            output=engine.predict(model,x[test],args.batch_size)
            prediction=output.argmax(axis=1) if CLASSIFICATION else inverse(output,ys)
            score=metrics(y[test],prediction,CLASSIFICATION)
            baseline=np.bincount(y[train]).argmax() if CLASSIFICATION else y[train].mean()
            base=metrics(y[test],np.full(len(test),baseline),CLASSIFICATION)
            # Warmed, synchronous batch-one forward latency; input preprocessing excluded.
            for _ in range(3): engine.predict(model,x[test[:1]])
            times=[]
            for _ in range(20):
                start=time.perf_counter(); engine.predict(model,x[test[:1]])
                times.append((time.perf_counter()-start)*1000)
            score.update(latency_median_ms=float(np.median(times)),latency_p95_ms=float(np.percentile(times,95)),
                checkpoint_mb=checkpoint.stat().st_size/1e6)
            write_json(folder/'metrics.json',{'status':status,'model':score,'training_only_baseline':base,
                'best_epoch':int(np.argmin(history['val_loss']))+1,'latency_protocol':'3 warmups, 20 batch-one forwards, excludes preprocessing'})
            rows.append({'fold':fold,**score})
            table=pd.DataFrame({'subject_id':ids[test],'fold':fold,'actual':AGES[y[test]] if CLASSIFICATION else y[test],
                'predicted':AGES[prediction] if CLASSIFICATION else prediction})
            if CLASSIFICATION:
                for j,age in enumerate(AGES): table[f'p_age_{age}']=output[:,j]
            tables.append(table)
        combined=pd.concat(tables,ignore_index=True)
        combined.to_csv(root/kind/'held_out_predictions.csv',index=False)
        pd.DataFrame(rows).to_csv(root/kind/'fold_metrics.csv',index=False)
        overview[kind]={key:{'mean':float(np.mean([r[key] for r in rows])),
                            'std':float(np.std([r[key] for r in rows],ddof=1)) if len(rows)>1 else 0.}
                        for key in rows[0] if key!='fold'}
        actual=np.searchsorted(AGES,combined.actual) if CLASSIFICATION else combined.actual.to_numpy()
        predicted=np.searchsorted(AGES,combined.predicted) if CLASSIFICATION else combined.predicted.to_numpy()
        plot_evaluation(actual,predicted,root/kind/'evaluation.png',CLASSIFICATION,args.demo)
    write_json(root/'summary.json',{'status':status,'folds':len(splits),'models':overview,
        'interpretation':'Fold SD is not a confidence interval. Repeated model selection requires an independent final test or nested CV.'})
    print('Completed:',root)


if __name__=='__main__': main()
