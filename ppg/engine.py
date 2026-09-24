"""Framework-specific training and native checkpoint restoration."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import numpy as np
from .settings import ENGINE
from .models import build


def initialize(seed):
    if ENGINE == 'keras':
        import tensorflow as tf
        import keras
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.experimental.enable_op_determinism()
        keras.utils.set_random_seed(seed)
    else:
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)


def predict(model,x,batch=32):
    if ENGINE == 'keras':
        return np.concatenate([model(x[i:i+batch],training=False).numpy() for i in range(0,len(x),batch)])
    import torch
    model.eval()
    output=[]
    with torch.inference_mode():
        for i in range(0,len(x),batch):
            tensor=torch.from_numpy(x[i:i+batch].transpose(0,3,1,2).copy())
            output.append(model(tensor).numpy())
    return np.concatenate(output)


def fit(kind,x,y,train,val,args):
    if ENGINE == 'keras':
        import keras
        import tensorflow as tf
        keras.backend.clear_session()
        keras.utils.set_random_seed(args.seed)
        def dataset(indices,shuffle=False):
            ds=tf.data.Dataset.from_tensor_slices((x[indices],y[indices]))
            if shuffle: ds=ds.shuffle(len(indices),seed=args.seed,reshuffle_each_iteration=True)
            options=tf.data.Options()
            options.threading.private_threadpool_size=1
            options.threading.max_intra_op_parallelism=1
            return ds.batch(args.batch_size).with_options(options).prefetch(1)
        model=build(kind,x.shape[1:],args.weights=='imagenet')
        history=model.fit(dataset(train,True),validation_data=dataset(val),epochs=args.epochs,verbose=2,
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=args.patience,restore_best_weights=True),
                       keras.callbacks.TerminateOnNaN()]).history
        if not all(np.isfinite(v).all() for v in history.values()): raise RuntimeError('Nonfinite training/validation loss.')
        return model,history
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    torch.manual_seed(args.seed)
    model=build(pretrained=args.weights=='imagenet')
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    tensor=torch.from_numpy(x.transpose(0,3,1,2).copy())
    targets=torch.from_numpy(y.astype(np.float32))
    batch=min(args.batch_size,len(train))
    while len(train)%batch==1 and batch<len(train): batch+=1
    loader=DataLoader(TensorDataset(tensor[train],targets[train]),batch_size=batch,shuffle=True,
        num_workers=0,generator=torch.Generator().manual_seed(args.seed))
    best,stale,state=float('inf'),0,None
    history={'loss':[],'val_loss':[]}
    for epoch in range(args.epochs):
        model.train()
        total=0.
        for xb,yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(model(xb).squeeze(1),yb)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss.')
            loss.backward()
            optimizer.step()
            total+=loss.item()*len(yb)
        value=float(np.mean((predict(model,x[val]).reshape(-1)-y[val])**2))
        if not np.isfinite(value): raise RuntimeError('Nonfinite validation loss.')
        history['loss'].append(total/len(train)); history['val_loss'].append(value)
        print(f'Epoch {epoch+1}: train={total/len(train):.5f}, validation={value:.5f}',flush=True)
        if value<best:
            best,stale=value,0
            state={key:value.detach().clone() for key,value in model.state_dict().items()}
        else:
            stale+=1
            if stale>=args.patience: break
    model.load_state_dict(state)
    return model,history


def save(model,folder):
    if ENGINE=='keras':
        path=folder/'model.keras'; model.save(path)
    else:
        import torch
        path=folder/'model.pt'; torch.save(model.state_dict(),path)
    return path


def load(folder):
    if ENGINE=='keras':
        import keras
        return keras.models.load_model(folder/'model.keras',compile=False,safe_mode=True)
    import torch
    model=build(pretrained=False)
    model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True))
    return model
