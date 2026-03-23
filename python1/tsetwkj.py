from neuromaps.datasets import fetch_annotation
from neuromaps.datasets import available_annotations
for annotation in available_annotations():
    print(annotation)