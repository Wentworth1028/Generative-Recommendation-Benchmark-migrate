"""TIGER dataset adapter that preserves item-level history for DROS sampling."""

from typing import Any, Dict, List, Union

from genrec.data.datasets.generative.tiger_dataset import TigerDataset


class DROSTigerDataset(TigerDataset):
    def __getitem__(self, index: int) -> Dict[str, Union[int, List[int]]]:
        feature: Dict[str, Any] = super().__getitem__(index)
        feature["history_items"] = list(self.samples[index]["history_items"])
        return feature
