from dataset import DIV2KDataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from metrics import PSNR
import global_config
from global_config import device_type
from model import Model, ModelVDSR

# Get the avarage PSNR of the model-output compared to the original images in the validation-set
def eval_model(model, test_dataset, metric):
    total_sum = 0
    for i in range(len(test_dataset)):
        (lr, hr) = test_dataset[i]

        result_img = model.evalvalidate(lr)

        total_sum += metric(result_img, hr)
    avg = total_sum / len(test_dataset)
    print("average psnr of validation: ", avg)
    return avg

# Get the avarage PSNR of bicubic interpolated low-res images compared to the original images in the validation-set
def eval_model_bicubic(test_dataset, metric):
    total_sum = 0
    for i in range(len(test_dataset)):
        (lr, hr) = test_dataset[i]
        lr_bicubic = test_dataset.get_bicubic(i)
        total_sum += metric(lr_bicubic, hr)
    avg = total_sum / len(test_dataset)
    print("average psnr of validation: ", avg)
    return avg


def train_model(optimizer, model, train_dataset, learning_rate, num_epochs):
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    batch_size = global_config.batch_size
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_set = DIV2KDataset(train=False, ycbcr=train_dataset.ycbcr)

    eval_psnr = PSNR()
    avg_psnr_over_time = []
    for i_epoch in range(num_epochs):
        print("epoch: ", i_epoch)
        model.train()  # used batchnorm and dropout
        for i_batch, (X_batch, y_batch) in enumerate(dataloader):
            model.zero_grad()  # reset model gradients
            output = model(X_batch)  # conduct forward pass
            loss = -eval_psnr(output, y_batch)
            loss.backward()  # backpropogate loss to calculate gradients
            optimizer.step()  # update model weights: w = w_old - lr* gradient
            print("   Batch #", i_batch)
        with torch.no_grad():  # no need to calculate gradients when assessing accuracy
            model.eval()
            psnr_res = float(eval_model(model, test_set, eval_psnr))
            avg_psnr_over_time.append(psnr_res)

            # Save model every 10 epochs so that we can continue where we left off in case of crash
            if (i_epoch+1) % 10 == 0:
                torch.save(model, "./model_" + str(i_epoch) + ".temp")
    return model


if __name__ == "__main__":
    optimizer = torch.optim.Adam
    eval_psnr = PSNR()

    print("Instantiating ModelVDSR...")
    model = ModelVDSR().to(device_type)
    print("Creating training dataset...")
    train_dset = DIV2KDataset(ycbcr=True, train=True)
    print("Starting training...")
    val_acc = train_model(optimizer, model, train_dset,
                          learning_rate=0.001, num_epochs=1)
    print("Training completed.")

    test_set = DIV2KDataset(train=False)
    print("BICUBIC PSNR")
    psnr_res = eval_model_bicubic(test_set, eval_psnr)

    test_set2 = DIV2KDataset(ycbcr=True, train=False)
    print("New VDSR architecture")
    psnr_res = eval_model(model, test_set2, eval_psnr)
