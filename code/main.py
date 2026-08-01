from loaders.csv_loader import CSVLoader

loader = CSVLoader("../dataset")

data = loader.load_everything()

print("Datasets Loaded Successfully!\n")

for name, df in data.items():
    print(f"{name}: {len(df)} rows")