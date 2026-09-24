"""Convert wide CSV exports using explicit identifiers or confirmed row correspondence."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--signals',required=True);p.add_argument('--targets',required=True)
    p.add_argument('--signal-id-column');p.add_argument('--target-id-column')
    p.add_argument('--target-column',required=True)
    p.add_argument('--task',choices=['classification','regression'],required=True)
    p.add_argument('--drop-signal-columns',nargs='*',default=[])
    p.add_argument('--assume-row-aligned',action='store_true')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    s=pd.read_csv(a.signals,dtype={a.signal_id_column:str} if a.signal_id_column else None)
    t=pd.read_csv(a.targets,dtype={a.target_id_column:str} if a.target_id_column else None)
    if a.signal_id_column and a.target_id_column:
        sid=s.pop(a.signal_id_column);tid=t[a.target_id_column]
    elif not a.signal_id_column and not a.target_id_column and a.assume_row_aligned:
        if len(s)!=len(t):p.error('Row counts differ.')
        sid=tid=pd.Series([f'row-{i:05d}' for i in range(len(s))])
    else:p.error('Supply both ID columns or explicitly confirm --assume-row-aligned with neither.')
    for values in [sid,tid]:
        if values.isna().any() or values.astype(str).str.strip().eq('').any() or values.astype(str).str.strip().duplicated().any():
            p.error('Missing/duplicate subject IDs.')
    sid=sid.astype(str).str.strip();tid=tid.astype(str).str.strip()
    if set(sid)!=set(tid):p.error('Signal and target ID sets differ.')
    s=s.drop(columns=a.drop_signal_columns)
    if any(str(c).lower().startswith('unnamed') for c in s):p.error('Remove exported index columns explicitly with --drop-signal-columns.')
    s=s.apply(pd.to_numeric,errors='raise')
    s.columns=[f's{i:04d}' for i in range(s.shape[1])]
    s.insert(0,'subject_id',sid.to_numpy())
    labels=pd.DataFrame({'subject_id':tid,'age' if a.task=='classification' else 'cfpwv_m_s':pd.to_numeric(t[a.target_column],errors='raise')})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    s.to_csv(out/'signals.csv',index=False);labels.to_csv(out/'targets.csv',index=False)
    print(f'Converted {len(s)} rows. Confirm that signal columns contain waveform samples only.')


if __name__=='__main__':main()
