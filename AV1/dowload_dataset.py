import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files(dataset="itachi9604/disease-symptom-description-dataset",unzip= True)