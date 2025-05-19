import torch
from matplotlib import pyplot as plt

from dataset import DataSet
from noise_scheduler import NoiseScheduler
from trainer import Trainer
from unet.unet import SimpleUnet

# Press Shift+F10 to execute it


if __name__ == '__main__':


    dataset = DataSet(64, 128,'./datasets/scraped_faces_dataset')

    dataloader_iterator = iter(dataset.get_dataloader()) # iterable with the data in it
    #print(len(dataloader_iterator)) # length is num_of_train_pictures / batch_size (etc. 512 / 128 = 4)

    # test dataset
    # dataset.test()
    """image = next(iter(dataset.get_dataloader()))[0]
    plt.figure(figsize=(20, 20))
    plt.axis('off')
    dataset.show_tensor_image(image)
    plt.show()"""

    model = SimpleUnet()
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    print(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    noise_scheduler = NoiseScheduler(T = 300, device = device)

    trainer = Trainer(model, dataset, noise_scheduler, device)

    for i in range(1000):
        trainer.train()
        torch.save(model.state_dict(), "./saves/{i}.txt".format(i=i))

    # igy kell majd visszatölteni a modelt:
    """
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()
    """





