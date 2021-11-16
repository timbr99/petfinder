
import swin
import os

#%%
st = swin.PetfinderTransformer(seed=1337, dataset_parent="../petfinder-pawpularity-score",
                               train_test_json_path="../petfinder-pawpularity-score/train_test_val.json")

#%%

st.wrap_model(dataset_parent="../petfinder-pawpularity-score",
              train_test_json_path="../petfinder-pawpularity-score/train_test_val.json",
              batch_size=4, image_size=224,
              model_name="swin_large_patch4_window7_224", epochs=2)
