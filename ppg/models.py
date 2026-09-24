"""Four model families for six fixed age categories."""
import keras
from keras import layers


def build(kind, shape, pretrained=False):
    inputs = keras.Input(shape=shape)
    x = inputs
    if kind == 'mlp':
        for width in [128,128,32]: x = layers.Dense(width,activation='tanh')(x)
    elif kind == 'cnn1d':
        for filters in [32,16]:
            x = layers.MaxPooling1D(2)(layers.Conv1D(filters,7,activation='relu')(x))
        x = layers.Flatten()(x)
        for width in [128,64,32,16]: x = layers.Dense(width,activation='relu')(x)
    elif kind == 'cnn2d':
        for _ in range(2): x = layers.MaxPooling2D(2)(layers.Conv2D(32,7,padding='same',activation='relu')(x))
        x = layers.Flatten()(x)
        for width in [64,32,16]: x = layers.Dense(width,activation='relu')(x)
    else:
        base = keras.applications.VGG16(weights='imagenet' if pretrained else None,include_top=False,input_shape=shape)
        x = layers.Flatten()(base(x))
    model = keras.Model(inputs,layers.Dense(6,activation='softmax')(x))
    model.compile(optimizer=keras.optimizers.Adam(.0001 if kind=='vgg16' else .001),loss='sparse_categorical_crossentropy')
    return model
