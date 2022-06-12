if __name__ == "__main__":

    import platform, torch
    # /Users/jwher/miniconda3/envs/py39_native/bin/python
    print(platform.platform())

    CPU= True
    device = "cpu" if CPU else torch.device("mps")
    print("Device is : {}".format(device))

    class CFG:
        lr = 0.001
        train_batch_size = 4
        total_epochs = 1
        num_classes = 10
        input_shape = (224,224)

    # Important to fix random seed
    torch.manual_seed(1)
    np.random.seed(1)

    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    import torchvision
    from torchvision import datasets, transforms
    from torchvision.transforms import ToTensor

    image_path = "./data/"
    mnist_dataset = torchvision.datasets.MNIST(
        image_path, 'train', download=False,
        transform=transforms.Compose(
            [transforms.Resize(CFG.input_shape), transforms.Grayscale(3),ToTensor()]
        )
    )
    trainset_1 = torch.utils.data.Subset(mnist_dataset, list(range(1000)))
    mnist_loader  = DataLoader(trainset_1,batch_size=CFG.train_batch_size,shuffle=True,num_workers=4)
    x_batch, y_batch = (next(iter(mnist_loader)))

    import torchvision.models as models
    import torch.nn as nn
    import time, numpy as np
    from tqdm import tqdm

    class MODELS:
        vgg16_model = models.vgg16(pretrained=True)
        alexnet = models.alexnet(pretrained=True)
        resnet_18 = models.resnet18(pretrained=True)
        mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        efficientnet_b0 = models.efficientnet_b0(pretrained=True)
        squeezenet = models.squeezenet1_0(pretrained=True)

        vgg16_model.classifier[6] = nn.Linear(vgg16_model.classifier[6].in_features,CFG.num_classes)
        alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features,CFG.num_classes)
        resnet_18.fc = nn.Linear(resnet_18.fc.in_features,CFG.num_classes)
        mobilenet_v2.classifier[1] = nn.Linear(mobilenet_v2.classifier[1].in_features,CFG.num_classes)
        efficientnet_b0.classifier[1] = nn.Linear(efficientnet_b0.classifier[1].in_features,CFG.num_classes)
        squeezenet.classifier[1] = nn.Linear(squeezenet.classifier[1].in_channels,CFG.num_classes)

    def train(model_name,model,train_dl,n_epochs=CFG.total_epochs):
        '''
        call train_one_epoch:
        we will take average time taken to train per epoch for a max of 5 epochs
        '''
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=CFG.lr)
        average_time = []
        for epoch in range(n_epochs):
            start_time = time.time()
            print(f"Epoch {epoch} -->")
            pbar = tqdm(enumerate(train_dl), total=len(train_dl), desc='Train : '+model_name)
            for step, (x_batch, y_batch) in pbar:   
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)[:,0]
                loss = loss_fn(pred,y_batch.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            end_time = time.time() - start_time
            average_time.append(end_time)
        return np.mean(average_time)

    model_dict = {
        'vgg16_model' : MODELS.vgg16_model, 
        'alexnet' : MODELS.alexnet, 
        'resnet_18' : MODELS.resnet_18, 
        'mobilenet_v2' : MODELS.mobilenet_v2, 
        'efficientnet_b0' : MODELS.efficientnet_b0, 
        # 'squeezenet' : MODELS.squeezenet, 
    }

    time_calc = {}
    for model_name,model in model_dict.items():
        print("Model name is : {}".format(model_name))
        print("-----------------------------------------")
        model_epoch_avg_time = train(model_name,model.to(device),mnist_loader)
        time_calc[model_name] = model_epoch_avg_time

    print("Time result with device={}".format(device))
    for key, value in time_calc.items():
        print(f"{key}: {value}")
