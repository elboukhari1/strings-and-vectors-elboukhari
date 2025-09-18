"""
The main code for the Strings-to-Vectors assignment. See README.md for details.
"""
from typing import Sequence, Any

import numpy as np


class Index:
    """
    Represents a mapping from a vocabulary (e.g., strings) to integers.
    """

    def __init__(self, vocab: Sequence[Any], start=0):
        """
        Assigns an index to each unique item in the `vocab` iterable,
        with indexes starting from `start`.

        Indexes should be assigned in order, so that the first unique item in
        `vocab` has the index `start`, the second unique item has the index
        `start + 1`, etc.
        """
        self.start = start
        self.object_to_index = {}
        self.index_to_object = {}

        idx = start
        for obj in vocab:
            if obj not in self.object_to_index:
                self.object_to_index[obj] = idx
                self.index_to_object[idx] = obj
                idx += 1

        self.vocab_size = idx - start
        self.not_in_index = self.start - 1

    def objects_to_indexes(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a vector of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array of the object indexes.
        """
        return np.array(
            [self.object_to_index.get(obj, self.not_in_index) for obj in object_seq], dtype = int
        )



    def objects_to_index_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a matrix of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        If the sequences are not all of the same length, shorter sequences will
        have padding added at the end, with `start-1` used as the pad value.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array of the object indexes.
        """
        max_length = max(len(seq) for seq in object_seq_seq)
        matrix = np.full((len(object_seq_seq), max_length), self.not_in_index, dtype = int)

        for i, seq in enumerate(object_seq_seq):
            indexes = [self.object_to_index.get(obj, self.not_in_index) for obj in seq]
            matrix[i, :len(indexes)] = indexes
        return matrix

    def objects_to_binary_vector(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a binary vector, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """
        vector = np.zeros(self.vocab_size + self.start, dtype = int)
        for obj in object_seq:
            idx = self.object_to_index.get(obj, None)
            if idx is not None:
                vector[idx] = 1
        return vector

    def objects_to_binary_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a binary matrix, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array, where each row in the array corresponds
                 to a row in the input, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """
        matrix = np.zeros((len(object_seq_seq), self.vocab_size + self.start), dtype=int)
        for i, seq in enumerate(object_seq_seq):
            matrix[i] = self.objects_to_binary_vector(seq)
        return matrix


    def indexes_to_objects(self, index_vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of objects associated with the indexes in the input
        vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_vector: A 1-dimensional array of indexes
        :return: A sequence of objects, one for each index.
        """
        return [self.index_to_object[idx] for idx in index_vector if idx in self.index_to_object]

    def index_matrix_to_objects(
            self, index_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects associated with the indexes
        in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_matrix: A 2-dimensional array of indexes
        :return: A sequence of sequences of objects, one for each index.
        """
        return [self.indexes_to_objects(row) for row in index_matrix]

    def binary_vector_to_objects(self, vec: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of the objects identified by the nonzero indexes in
        the input vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param vector: A 1-dimensional binary array
        :return: A sequence of objects, one for each nonzero index.
        """
        return [self.index_to_object[idx] for idx, val in enumerate(vec) if val == 1 and idx in self.index_to_object]

    def binary_matrix_to_objects(
            self, binary_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects identified by the nonzero
        indices in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param binary_matrix: A 2-dimensional binary array
        :return: A sequence of sequences of objects, one for each nonzero index.
        """
        return [self.binary_vector_to_objects(row) for row in binary_matrix]
