import numpy as np
import logging
import argparse

def _arrays_equal(a, b, rtol=1e-6, atol=1e-6):
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":  # float or complex
        return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
    else:
        return np.array_equal(a, b, equal_nan=False)

def compare_npz_files(file1, file2, assert_on_difference=True):
    logging.debug(f"Loading files: {file1}, {file2}")
    data1 = np.load(file1, allow_pickle=True)
    data2 = np.load(file2, allow_pickle=True)

    keys1 = set(data1.keys())
    keys2 = set(data2.keys())

    if keys1 != keys2:
        msg = (
            f"Keys differ between the two files.\n"
            f"Keys only in {file1}: {keys1 - keys2}\n"
            f"Keys only in {file2}: {keys2 - keys1}"
        )
        logging.error(msg)
        if assert_on_difference:
            assert keys1 == keys2, msg

    else:
        logging.debug("All keys match between the two files.")

    for key in sorted(keys1 & keys2):
        value1 = data1[key]
        value2 = data2[key]
        if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
            if not _arrays_equal(value1, value2):
                msg = f"np.ndarray values for key '{key}' are different."
                logging.error(msg)
                if assert_on_difference:
                    assert _arrays_equal(value1, value2), msg
            else:
                logging.debug(f"Key '{key}': np.ndarray values are the SAME")
        else:
            logging.debug(f"Key '{key}': Non-array values or mismatched types, skipping detailed comparison")

    data1.close()
    data2.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two .npz files for matching keys and ndarray values.")
    parser.add_argument("file1", help="First .npz file")
    parser.add_argument("file2", help="Second .npz file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase verbosity of logging")
    parser.add_argument("--no-assert", action="store_true", help="Do not assert on differences, only log them")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format='%(levelname)s: %(message)s'
    )

    compare_npz_files(args.file1, args.file2, assert_on_difference=not args.no_assert)
