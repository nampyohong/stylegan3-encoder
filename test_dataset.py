from training.dataset_encoder import ImagesDataset


if __name__ == '__main__':
    # TODO : get this path from config
    ffhqs_dataset_dir = 'data/ffhqs'
    ffhqs_dataset = ImagesDataset(ffhqs_dataset_dir, mode='train')
    print(f'dataset length: {len(ffhqs_dataset)}')
    print('transforms')
    print(ffhqs_dataset.transforms)
    print(f'input image shape: {ffhqs_dataset.__getitem__(0)[0].shape}')
    print("Done.")
