import pandas as pd
import os
import numpy as np
import matplotlib.pylab as plt

# %%

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
# %%
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
