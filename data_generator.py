from tensorflow.keras.utils import Sequence
import albumentations as A
import numpy as np
import cv2
import random

class DataGenerator(Sequence):
    
    def __init__(self, df, batch_size, dimension=(256, 256), shuffle = False, augment=False):
        
        self.df = df.copy()
        self.df['flipped_angle'] = 1 - self.df['angle']
        self.batch_size = batch_size
        self.dimension = dimension
        self.shuffle = shuffle
        self.augment = augment
        
        self.list_index = self.df.index
        self.indices = np.arange(len(self.list_index))
        self.on_epoch_end()
        
        self.augmentor = A.Compose([
            A.RandomShadow(p=0.33),
            A.ColorJitter(p=0.33),
            A.Rotate(limit=5, p=0.33),
            A.RandomCrop(height=215, width=295, p=0.33)
        ]
        )
            
    def __len__(self):
        return int(np.ceil(len(self.df))/float(self.batch_size))
    
    def __getitem__(self,index):
        batch_index = self.indices[index*self.batch_size: (index+1)*self.batch_size]
        idx = [self.list_index[k] for k in batch_index]
        
        Data1 = np.zeros((self.batch_size, self.dimension[0], self.dimension[1], 3))
        Data2 = np.zeros((self.batch_size, 224, 224, 3))
        Target = np.zeros((self.batch_size, 2))
        for i, k in enumerate(idx):
            image = cv2.imread(
                self.df['filepath'][k] + str(self.df['image_id'][k]) + ".png"
            )
            flipped = False
            if self.augment:
                flipped = random.random() > 0.5
                if flipped:
                    image = cv2.flip(image, 1)
                image = self.augmentor(image=image)['image']
            image = image.astype(np.float32)/255.0
            image1 = cv2.resize(image, (self.dimension[1], self.dimension[0]))
            image2 = cv2.resize(image, (224, 224))
            Data1[i,:,:,:] = image1
            Data2[i,:,:,:] = image2
            if flipped:
                Target[i, :] = self.df.loc[k][['flipped_angle', 'speed']].values
            else:
                Target[i, :] = self.df.loc[k][['angle', 'speed']].values
        return [Data1, Data2], Target
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)