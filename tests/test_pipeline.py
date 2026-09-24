"""Guard the concrete data integrity, preprocessing and checkpoint failure modes."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from ppg.common import demo_data,fit_scale,inverse,load_data,load_signals,represent,scale,split_data
from ppg.settings import ENGINE,MODELS,CLASSIFICATION


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.sig,self.lab=demo_data(self.root,False)
    def tearDown(self):self.tmp.cleanup()
    def test_id_alignment(self):
        _,_,y=load_data(self.sig,self.lab,512,False)
        pd.read_csv(self.lab).sample(frac=1,random_state=7).to_csv(self.lab,index=False)
        np.testing.assert_array_equal(load_data(self.sig,self.lab,512,False)[2],y)
    def test_missing_id_fails(self):
        pd.read_csv(self.lab).iloc[:-1].to_csv(self.lab,index=False)
        with self.assertRaisesRegex(ValueError,'ID sets'):load_data(self.sig,self.lab,512,False)
    def test_duplicate_id_fails(self):
        df=pd.read_csv(self.sig);df.loc[1,'subject_id']=df.loc[0,'subject_id'];df.to_csv(self.sig,index=False)
        with self.assertRaisesRegex(ValueError,'unique'):load_signals(self.sig,512)
    def test_no_silent_cropping(self):
        with self.assertRaisesRegex(ValueError,'cropping'):load_signals(self.sig,100)
    def test_internal_missing_fails(self):
        df=pd.read_csv(self.sig);df.loc[0,'s0002']=np.nan;df.to_csv(self.sig,index=False)
        with self.assertRaisesRegex(ValueError,'Internal'):load_signals(self.sig,512)
    def test_trailing_missing_padding(self):
        df=pd.read_csv(self.sig);df.loc[0,['s0510','s0511']]=np.nan;df.to_csv(self.sig,index=False)
        np.testing.assert_array_equal(load_signals(self.sig,512)[1][0,-2:],[0,0])
    def test_scaling_boundary(self):
        train=np.array([1.,2.,3.]);state=fit_scale(train)
        self.assertEqual(state['mean'],2.)
        self.assertGreater(scale(np.array([1000.]),state)[0],100)
        np.testing.assert_allclose(inverse(scale(train,state),state),train,atol=1e-6)
    def test_splits_disjoint_and_complete(self):
        y=np.repeat(np.arange(6),12);first=split_data(y,42,5,True);second=split_data(y,42,5,True)
        seen=[]
        for (tr,val,te),other in zip(first,second):
            self.assertFalse(set(tr)&set(val) or set(tr)&set(te) or set(val)&set(te))
            self.assertEqual(set(y[val]),set(range(6)))
            self.assertEqual(len(tr)+len(val)+len(te),len(y))
            for a,b in zip([tr,val,te],other):np.testing.assert_array_equal(a,b)
            seen.extend(te)
        self.assertEqual(sorted(seen),list(range(len(y))))
    def test_window_is_used(self):
        x=load_signals(self.sig,512)[1]
        config={'fs':500,'nperseg':76,'window':'hann'}
        first=represent(x,'spectrogram',config);config['window']='hamming'
        self.assertFalse(np.allclose(first,represent(x,'spectrogram',config)))


class ModelTests(unittest.TestCase):
    def test_all_model_checkpoint_roundtrips(self):
        from ppg import engine
        from ppg.models import build
        shapes={'waveform':(128,1),'spectrogram':(39,7,1),'mlp':(128,),
                'cnn1d':(128,1),'cnn2d':(39,7,1),'vgg16':(32,32,3),'resnet18':(64,64,3)}
        engine.initialize(42)
        for kind in MODELS:
            with self.subTest(model=kind),tempfile.TemporaryDirectory() as temp:
                model=build(kind,shapes[kind],False)
                x=np.random.default_rng(42).normal(size=(2,*shapes[kind])).astype(np.float32)
                expected=engine.predict(model,x)
                self.assertEqual(expected.shape,(2,6 if CLASSIFICATION else 1))
                self.assertTrue(np.isfinite(expected).all())
                if CLASSIFICATION:np.testing.assert_allclose(expected.sum(axis=1),1,atol=1e-6)
                engine.save(model,Path(temp))
                np.testing.assert_allclose(engine.predict(engine.load(Path(temp)),x),expected,atol=1e-6,rtol=1e-5)


if __name__=='__main__':unittest.main()
