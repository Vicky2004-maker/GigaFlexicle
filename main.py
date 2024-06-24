import pandas as pd
import os
import numpy as np
from numpy.random import randint
import matplotlib.pylab as plt
import cv2

# %% Data Loading

annotations = pd.read_csv(r"S:\Dataset\GigaFlexicle\archive\annotation.csv")

parent_path = r"S:\Dataset\GigaFlexicle\archive\images"
brands = os.listdir(parent_path)
print(f'Brands Found: {len(brands)}')

data = pd.DataFrame(columns=['Brand', 'Model', 'Variant', 'Images_Count'])

row = pd.Series([0, 0, 0, 0], index=data.columns, dtype='object')
for brand in brands:
    img_count = 0
    for model in os.listdir(os.path.join(parent_path, brand)):
        for variant in os.listdir(os.path.join(parent_path, brand, model)):
            img_count = len(os.listdir(os.path.join(parent_path, brand, model, variant)))
            row[data.columns[0]] = brand
            row[data.columns[1]] = model
            row[data.columns[2]] = variant
            row[data.columns[3]] = img_count
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

# %% Data Preprocessing

target_brand = 'bmw'

target_data = data[data['Brand'] == target_brand]
