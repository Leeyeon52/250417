import tensorflow as tf
from tensorflow.keras import Sequential, layers

def build_vgg16():
    # Sequential 모델 선언
    model = Sequential()
    
    # TODO: [지시시항 1번] 첫번째 Block을 완성하세요.
    model.add(layers.Conv2D(filters=64,kernel_size=(3,3), padding='same',activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.Conv2D(filters=64,kernel_size=(3,3), padding='same',activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    # TODO: [지시시항 2번] 두번째 Block을 완성하세요.
    model.add(layers.Conv2D(filters=128,kernel_size=(3,3), padding='same',activation='relu'))
    model.add(layers.Conv2D(filters=128,kernel_size=(3,3), padding='same',activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    # TODO: [지시시항 3번] 세번째 Block을 완성하세요.
    model.add(layers.Conv2D(filters=256,kernel_size=(3,3), padding='same',activation='relu'))
    model.add(layers.Conv2D(filters=256,kernel_size=(3,3), padding='same',activation='relu'))
    model.add(layers.Conv2D(filters=256,kernel_size=(3,3), padding='same',activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    # TODO: [지시시항 4번] 네번째 Block을 완성하세요.
    model.add(layers.Conv2D(filters=512,kernel_size=(3,3), padding='same',activation='relu'))
    model.add(layers.Conv2D(filters=512,kernel_size=(3,3), padding='same',activation='relu'))
    model.add(layers.Conv2D(filters=512,kernel_size=(3,3), padding='same',activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    # TODO: [지시시항 5번] 다섯번째 Block을 완성하세요.
    model.add(layers.Conv2D(filters=512,kernel_size=(3,3), padding='same',activation='relu'))
    model.add(layers.Conv2D(filters=512,kernel_size=(3,3), padding='same',activation='relu'))
    model.add(layers.Conv2D(filters=512,kernel_size=(3,3), padding='same',activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    # Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dense(1000, activation="softmax"))
    
    return model

def main():
    model = build_vgg16()
    model.summary()
    
if __name__ == "__main__":
    main()
