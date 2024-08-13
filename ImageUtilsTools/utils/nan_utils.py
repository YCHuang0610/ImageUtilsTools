import numpy as np


def nan_rows(data):
    return np.where(np.isnan(data).sum(axis=1) > 0)[0]


def no_nan_rows(data):
    return np.where(np.isnan(data).sum(axis=1) == 0)[0]


class NanHelper:
    """
    A helper class for handling NaN values in a NumPy array.

    Parameters:
    - X (ndarray): The input array.

    Attributes:
    - rows (int): The number of rows in the input array.
    - nan_rows (ndarray): An array containing the indices of rows with NaN values.
    - no_nan_rows (ndarray): An array containing the indices of rows without NaN values.
    - have_nan (bool): Indicates whether the input array contains NaN values.

    Methods:
    - transform_delete_nan_rows(X): Removes rows with NaN values from the input array.
    - inverse_transform(XS): Restores the original shape of the input array, filling NaN rows with NaN values.
    - assign_nan_to_nan_rows(input): Assigns NaN values to the rows with NaN values in the input array.
    """

    def __init__(self, X):
        """
        Initializes a NanHelper object.

        Parameters:
        - X (ndarray): The input array.
        """
        self.rows = X.shape[0]
        self.nan_rows = nan_rows(X)
        self.no_nan_rows = no_nan_rows(X)

        if self.nan_rows.size == 0:
            assert len(self.no_nan_rows) == self.rows, "Something went wrong"
            self.have_nan = False
        else:
            assert (
                len(self.no_nan_rows) + len(self.nan_rows) == self.rows
            ), "Something went wrong"
            self.have_nan = True

    def transform_delete_nan_rows(self, X):
        """
        Removes rows with NaN values from the input array.

        Parameters:
        - X (ndarray): The input array.

        Returns:
        - ndarray: The input array with NaN rows removed.
        """
        if not self.have_nan:
            return X
        else:
            print(f"Removing {len(self.nan_rows)} rows with NaN values")
            return X[self.no_nan_rows, :]

    def inverse_transform(self, XS):
        """
        Restores the original shape of the input array, filling NaN rows with NaN values.

        Parameters:
        - XS (ndarray): The transformed array.

        Returns:
        - ndarray: The restored array.
        """
        was_one_dimensional = False
        if XS.ndim == 1:
            XS = XS.reshape(-1, 1)
            was_one_dimensional = True

        if not self.have_nan:
            output = XS
        else:
            output = np.full((self.rows, XS.shape[1]), np.nan)
            output[self.no_nan_rows, :] = XS

        if was_one_dimensional:
            output = output.ravel()

        return output

    def assign_nan_to_nan_rows(self, input):
        """
        Assigns NaN values to the rows that should be NaN.

        Parameters:
        - input (ndarray): The input array.

        Returns:
        - ndarray: The input array with NaN values assigned to NaN rows.
        """
        if not self.have_nan:
            return input
        else:
            output = np.full(input.shape, np.nan)
            output[self.no_nan_rows, :] = input[self.no_nan_rows, :]
            return output
