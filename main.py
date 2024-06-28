import pandas as pd
import os
import numpy as np
from numpy.random import randint
import matplotlib.pylab as plt
import cv2

from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Input, Resizing
from keras import Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
from keras.activations import relu

# %% Data Loading

annotations = pd.read_csv(r"S:\Dataset\GigaFlexicle\archive\annotation.csv")

parent_path = r"S:\Dataset\GigaFlexicle\archive\images"
brands = os.listdir(parent_path)
print(f'Brands Found: {len(brands)}')

data = pd.DataFrame(columns=['Brand', 'Model', 'Variant', 'Images_Count', 'Folder_Path'])

row = pd.Series([0, 0, 0, 0, 0], index=data.columns, dtype='object')
for brand in brands:
    img_count = 0
    for model in os.listdir(os.path.join(parent_path, brand)):
        for variant in os.listdir(os.path.join(parent_path, brand, model)):
            __sub_dir = os.path.join(parent_path, brand, model, variant)
            img_count = len(os.listdir(__sub_dir))
            row[data.columns[0]] = brand
            row[data.columns[1]] = model
            row[data.columns[2]] = variant
            row[data.columns[3]] = img_count
            row[data.columns[4]] = __sub_dir
            data = pd.concat([data, pd.DataFrame(row).T], ignore_index=True, axis=0)

brand_images_count = data.pivot_table(values=['Images_Count'], columns=['Brand'], aggfunc='sum').T
# %% Visualization
brand_input = input("Enter your desired brand: ")
match_brands = []
for b_ind, b_val in enumerate(brands):
    if brand_input in b_val:
        match_brands.append((b_ind + 1, b_val))

print(f'Founded {len(match_brands)} brands')
print(*match_brands, sep='\n')
input_specific_brand = int(input('To get a specific brand, input the associated position: '))
selected_brand = ''
for b_ind, b_val in match_brands:
    if input_specific_brand == b_ind:
        selected_brand = b_val
        print(b_ind, b_val)
        break

print(f'Selected Brand: {str.upper(selected_brand)}')

files_visualize = []

for model in os.listdir(os.path.join(parent_path, selected_brand)):
    for variant in os.listdir(os.path.join(parent_path, selected_brand, model)):
        __images = os.listdir(os.path.join(parent_path, selected_brand, model, variant))
        files_visualize.append(
            os.path.join(parent_path, selected_brand, model, variant, __images[randint(0, high=len(__images))]))

print(f'Plotting {len(files_visualize)} images')

plt.figure(figsize=(24, 24))

xy = int(np.ceil(np.sqrt(len(files_visualize))))

for file_ind, file_path in enumerate(files_visualize):
    plt.subplot(xy, xy, file_ind + 1)
    __f = file_path
    plt.imshow(cv2.imread(__f))
    __f = __f.split("\\")
    plt.title(str.upper(" ".join([__f[5], __f[6], __f[7]])))
    plt.axis('off')

plt.suptitle(str.upper(selected_brand))
plt.tight_layout()
plt.show()

# %%

target_brand = 'mclaren'

target_data = data[data['Brand'] == target_brand]
target_data.reset_index(inplace=True, drop=True)

X_train, y_train = [], []

for ind in target_data.index:
    for __img_path in os.listdir(target_data.loc[ind, 'Folder_Path']):
        # TODO: Preprocess the images
        __img_matrix = cv2.imread(os.path.join(target_data.loc[ind, 'Folder_Path'], __img_path))
        __img_matrix = cv2.cvtColor(__img_matrix, cv2.COLOR_BGR2GRAY)
        __img_matrix = cv2.resize(__img_matrix, (128, 128))
        X_train.append(__img_matrix)
        y_train.append(
            f"{target_data.loc[ind, 'Brand']}_{target_data.loc[ind, 'Model']}_{target_data.loc[ind, 'Variant']}")

X_train = np.array(X_train)
y_train = np.array(y_train)

# %%

print(X_train.shape)
print(X_train[0].shape)
print(y_train.shape)
# %% MODEL DEVELOPMENT

# FIXME: Depending upon the accuracy if necessary add Data Augmentation

model = Sequential(layers=[
    Input(X_train.shape, 32, ragged=True),

    Conv2D(4, 3, padding='same', activation=relu),
    MaxPooling2D(),

    Conv2D(8, 3, padding='same', activation=relu),
    MaxPooling2D(),

    Conv2D(16, 3, padding='same', activation=relu),
    MaxPooling2D(),

    Conv2D(32, 3, padding='same', activation=relu),
    MaxPooling2D(),

    Dense(y_train.shape)

], name='Sequential_Main')

model.compile(
    optimizer=Adam(),
    loss=categorical_crossentropy,
    metrics=['accuracy']
)

history = model.fit(X_train, y_train, batch_size=32, epochs=1, workers=-1, use_multiprocessing=True)
