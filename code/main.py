from loaders.csv_loader import CSVLoader
from context.index_builder import IndexBuilder

loader = CSVLoader("../dataset")

data = loader.load_everything()

indexes = IndexBuilder(data).build()

print(indexes["users"]["u_001"])