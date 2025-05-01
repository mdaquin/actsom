import dataset


def get_ld_info(idx, nb):
    ds = dataset.load_dataset(3000, split=False, SEED=42)
    item = ds[idx]
    print(item)


if __name__ == "__main__":
    # test
    res = get_ld_info(2572, 2)
    print(res)