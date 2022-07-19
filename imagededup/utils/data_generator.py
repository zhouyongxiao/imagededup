from pathlib import PurePath
from typing import Tuple, List, Callable
import os
from pathlib import Path
import json
import numpy as np
from tensorflow.keras.utils import Sequence

from imagededup.utils.image_utils import load_image


class DataGenerator(Sequence):
    """Class inherits from Keras Sequence base object, allows to use multiprocessing in .fit_generator.

    Attributes:
        image_dirs: Path of image directory.
        batch_size: Number of images per batch.
        basenet_preprocess: Basenet specific preprocessing function.
        target_size: Dimensions that images get resized into when loaded.
    """

    def __init__(
        self,
        image_dirs: List[PurePath],
        batch_size: int,
        basenet_preprocess: Callable,
        target_size: Tuple[int, int],
        filter_file: str,
    ) -> None:
        """Init DataGenerator object.
        """
        self.image_dirs = image_dirs
        self.batch_size = batch_size
        self.basenet_preprocess = basenet_preprocess
        self.target_size = target_size
        self.filter_file = filter_file
        self._get_image_files()
        self.indexes = np.arange(len(self.image_files))
        self.valid_image_files = self.image_files

    def _get_image_files(self) -> None:
        self.image_files = list()
        for image_dir in self.image_dirs:
            if not os.path.exists(image_dir / self.filter_file):
                self.image_files.append(
                        i.absolute()
                        for i in image_dir.glob('*')
                        if not i.name.startswith('.')
                )  # ignore hidden files
            else:
                file = open(str(image_dir / self.filter_file), "r")
                image_to_keep = json.loads(file.read())
                file.close()
                for i in image_to_keep:
                    self.image_files.append(Path(str(image_dir) + "/" + i)) # ignore hidden files
        print(len(self.image_files) + " images to encode")

    def __len__(self) -> int:
        """Number of batches in the Sequence."""
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        """Get batch at position `index`.
        """
        batch_indexes = self.indexes[
                        index * self.batch_size: (index + 1) * self.batch_size
                        ]
        batch_samples = [self.image_files[i] for i in batch_indexes]
        X = self._data_generator(batch_samples)
        return X

    def _data_generator(
            self, image_files: List[PurePath]
    ) -> Tuple[np.array, np.array]:
        """Generate data from samples in specified batch."""
        #  initialize images and labels tensors for faster processing
        X = np.empty((len(image_files), *self.target_size, 3))

        invalid_image_idx = []
        for i, image_file in enumerate(image_files):
            # load and randomly augment image
            img = load_image(
                image_file=image_file, target_size=self.target_size, grayscale=False
            )

            if img is not None:
                X[i, :] = img

            else:
                invalid_image_idx.append(i)
                self.valid_image_files = [_file for _file in self.valid_image_files if _file != image_file]

        if invalid_image_idx:
            X = np.delete(X, invalid_image_idx, axis=0)

        # apply basenet specific preprocessing
        # input is 4D numpy array of RGB values within [0, 255]
        X = self.basenet_preprocess(X)

        return X