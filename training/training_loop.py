import torch
from tqdm import tqdm


def training_loop(model_save_path: str, 
        network: torch.nn.Module, 
        train_data: torch.utils.data.Dataset, 
        eval_data: torch.utils.data.Dataset, 
        num_epochs: int, 
        show_progress: bool = True
    ):  
    # Main training loop for training the neural network.
    
    # This function handles the training and evaluation of the neural network over a specified number of epochs. 
    # It also saves the trained model to the specified path after training.
    
    # Args:
    #     model_save_path (str): Path to save the trained model.
    #     network (torch.nn.Module): The neural network to train.
    #     train_data (torch.utils.data.Dataset): The training data.
    #     eval_data (torch.utils.data.Dataset): The evaluation data.
    #     num_epochs (int): The number of epochs to train.
    #     show_progress (bool): Whether to show progress bars during training (default is True).
    
    # Returns:
    #     tuple: A tuple containing lists of training and evaluation losses per epoch.
    


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network.to(device)
    print(f"Using {device}.")


    optimizer = torch.optim.Adam(network.parameters(), lr=0.00001)
    loss_function = torch.nn.MSELoss()


    losses_train_dataloader = []
    losses_eval_dataloader = []

    no_improvement_counter = 0

    if show_progress:
        progress_bar = tqdm(total=num_epochs, desc="epochs")

    for epoch in range(num_epochs):
        network.train()
        average_batch_loss = []

        if show_progress:
            batch_progress_bar = tqdm(total=len(train_data), desc=f"Epoch {epoch+1}")

        for input_tensor, known_array, target_tensor, image_dir in train_data:
            input_tensor = input_tensor.to(device)

            output = network(input_tensor)

            
            loss = loss_function(output, torch.stack([t.clone().detach() for t in target_tensor]).to(device))



            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            average_batch_loss.append(loss.item())
            
            if show_progress:
                batch_progress_bar.update(1)
        if show_progress:
            batch_progress_bar.close()
        losses_train_dataloader.append(sum(average_batch_loss) / len(average_batch_loss))

        network.eval()
        with torch.no_grad():
            average_batch_loss_eval = []
            for input_tensor, target_tensor in eval_data:
                input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
                output = network(input_tensor)
                loss = loss_function(output, torch.stack([t.clone().detach() for t in target_tensor]).to(device))


                average_batch_loss_eval.append(loss.item())
        loss_eval = sum(average_batch_loss_eval) / len(average_batch_loss_eval)


        if (losses_eval_dataloader) and (loss_eval >= (losses_eval_dataloader[-1] - 1)):
                no_improvement_counter += 1

        else:
            no_improvement_counter = 0

        if no_improvement_counter == 3:
            break

        losses_eval_dataloader.append(loss_eval)

        if show_progress:
            progress_bar.update(1)

    if show_progress:
        progress_bar.close()

    torch.save(network.state_dict(), model_save_path)
    print(f"Trained model parameters saved to {model_save_path}")

    
    # Save the trained model to the specified path
    torch.save(network.state_dict(), model_save_path)

    for epoch, (tl, el) in enumerate(zip(losses_train_dataloader, losses_eval_dataloader)):
        print(f"Epoch: {epoch} --- Train loss: {tl:7.2f} --- Eval loss: {el:7.2f}")

    return losses_train_dataloader, losses_eval_dataloader


