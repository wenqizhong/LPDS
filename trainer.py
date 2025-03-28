import os
import torch
torch.autograd.set_detect_anomaly(True)
import time
from torch.utils.tensorboard import SummaryWriter


def train_model(args, model, device, dataloaders, criterion, optimizer, num_epochs, times=None, check_point=1):
    if args.train_p is None:
        p_save = 'train'
    else:
        p_save = os.path.basename(args.train_p)[2:]
        
    writer = SummaryWriter(os.path.join(args.logs, (('%s-'%p_save) + time.strftime("%m%d-%H%M"))))
    model.train()
    valid_loss_min = float('inf')

    for epoch in range(num_epochs):
        print('dataset:{}, val split:{}'.format(os.path.basename(args.train_dataset_dir), os.path.basename(args.train_p)))
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 20)

        for phase in ['train', 'val']:
            running_loss = 0
            if phase == 'train':
                model.train()
            else:
                model.eval()

            cnt = 0
            for iteration, (img, scanpaths, labels, trans_factor) in enumerate(dataloaders[phase]):

                duration = trans_factor[4]
                img = img.to(device)
                scanpaths = scanpaths.to(device)
                duration = duration.to(device)
                labels = labels.to(device)

                with torch.autograd.set_grad_enabled(phase == 'train'):
                    outputs = model(img, scanpaths)
                    loss = criterion(outputs, labels)


                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                cnt = cnt + 1
                if cnt == len(dataloaders[phase]):
                    print("\r {} Complete: {:.2f}".format(phase, cnt / len(dataloaders[phase])), end='\n')
                else:
                    print("\r {} Complete: {:.2f}".format(phase, cnt / len(dataloaders[phase])), end="")

                running_loss += loss.item() * img.size(0)

            epoch_loss = running_loss / len(dataloaders[phase])


            print("{} Loss: {}".format(phase, epoch_loss))
        save_path = os.path.join(args.checkpoint, ('path-' + args.model_name))
        if (epoch+1) % check_point == 0:
            torch.save(model.state_dict(), (os.path.join(save_path, ('model-%s_%s.pth'%(str(epoch+1),(p_save))))))