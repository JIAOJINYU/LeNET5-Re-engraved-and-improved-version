# coding = utf-8

import time
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D,Flatten,MaxPooling2D,Dense

from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array,array_to_img
import os,glob,random

base_path = 'dataset-resized'
img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))
print(len(img_list))

start = time.time()

# 数据预处理
train_datagen = ImageDataGenerator(
    rescale=1. / 225, # 对每个像素点的像素值进行放缩操作，从0 ~ 225放缩到 0 ~ 1
    shear_range=0.1, # 剪切变换幅度
    zoom_range=0.1, # 放缩幅度，占整个图片的比例0.1
    width_shift_range=0.1, # 在水平方向上的平移幅度占整个图片宽度比例0.1
    height_shift_range=0.1,  #在垂直方向上的平移幅度占整个图片宽度比例的0.1
    horizontal_flip=True, # 水平翻转
    vertical_flip=True, # 垂直翻转
    validation_split=0.1 # 划分90%的数据为训练集， 10%的数据为验证集
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255, validation_split=0.1) #不需要做任何图像变换的操作，因为测试集上必须是真实的图片


train_generator = train_datagen.flow_from_directory(
    base_path, #数据集的路径
    target_size=(300, 300), #调整所有图片大小为300 * 300
    batch_size=16, # 每次迭代使用的图片数量
    class_mode='categorical', # 返回one-hot 编码
    subset='training', # 这个数据集是训练集
    seed=0 #随机种子
    )

validation_generator = test_datagen.flow_from_directory(
    base_path,
    target_size=(300, 300),
    batch_size=16,
    class_mode='categorical',
    subset='validation', #这个数据集是验证集
    seed=0)


labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())

print('labels',labels)
# labels {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

# 4.模型的建立和训练

model = Sequential([
    Conv2D(filters=32, # 卷积核数量
           kernel_size=3, # 卷积核大小
           padding='same', # padding 方式
           activation='relu', # 激活函数种类
           input_shape=(300, 300, 3) #输入的数据格式，每个图片300*300，用RGB三原色，所以每个图片的每个像素有3个像素值
           ),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Flatten(),

    Dense(64, activation='relu'),

    Dense(6, activation='softmax')
])


model.compile(
    loss='categorical_crossentropy', # 损失函数，使用交叉熵损失函数
              optimizer='adam', # adam优化器
              metrics=['acc'] # 训练评估的指标是正确率 accuracy
              )


model.fit_generator(train_generator, #使用训练集来训练
                    epochs=1, #训练轮数
                    steps_per_epoch=2276//32, # 每轮训练多少步
                    validation_data=validation_generator, #用验证集来验证
                    validation_steps=251//32 # 多少步验证一次
                    )

#
# 5.结果展示





model.save('saved_model') #保存模型
print('模型保存成功')

end = time.time()
t = end-start
print('运行time',t)



saved_model = load_model("saved_model") #加载模型
print('模型加载成功')
test_x, test_y = validation_generator.__getitem__(1)

preds = saved_model.predict(test_x)

# plt.figure(figsize=(16, 16))
for i in range(16):
    print('For image %d, pred:%s / truth:%s' % (i, labels[np.argmax(preds[i])], labels[np.argmax(test_y[i])]))