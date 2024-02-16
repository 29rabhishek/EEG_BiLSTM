import torch
import torch.nn.functional as F
from utils import model_checkpoint, regional_info_extraction, latest_weight_file_path
import logging



class Trainer():
    """
    Trainer class 
    function in this class
    train_step(), test_step(), trainer_engin(), resume_training()
    """
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        loss_fn: torch.nn.Module,
        device: torch.device,
        model_type: str,
        ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.model_type = model_type

            
    
    def train_step(
        self,
        train_dataloader,
        ):
        """
        This function train the model and return training loss and validation accuracy

        Parameter
        train_dataloader: type DataLoader, DataLoader of train dataset

        Return 
        train_loss: list[tensor]
        train_acc: list[tensor]
        """
        self.model.train()
        train_loss = 0
        train_acc = 0
        for X, y in train_dataloader:
            X = X.to(self.device)
            y = y.to(self.device)
            yhat = self.model(X)
            loss = self.loss_fn(yhat, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += float(loss)
            acc = (F.softmax(yhat, dim = 1).argmax(dim = 1) == y).sum().item()/len(y)
            train_acc += acc
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        return round(train_loss, 4), round(train_acc, 4)


    def test_step(
        self,
        test_dataloader,
        ):
        """
        This function test the model and return test loss and test accuracy

        Parameter
        test_dataloader: type DataLoader, DataLoader of test dataset

        Return 
        test_loss: list[tensor]
        test_acc: list[tensor]
        """
        self.model.eval()
        test_loss = 0
        test_acc = 0
        with torch.inference_mode():
            for X, y in test_dataloader:  
                X = X.to(self.device)
                y = y.to(self.device)
                yhat = self.model(X)
                # feature extraction using ica

                loss = self.loss_fn(yhat, y)
                test_loss += float(loss)
                acc = (F.softmax(yhat, dim = 1).argmax(dim = 1) == y).sum().item()/len(y)
                test_acc += acc
            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
        return round(test_loss, 4), round(test_acc, 4)


    def reload_states_dict(
        self,
        checkpoint_path : str
        ):
        """
        This function load the state dict of model and optimizer from the save checkpoint
        Parameter:
            path_to_checkpoint: checkpoint we want to load
        return:
            epoch: checkpoint epoch(epoch at which we want to load the 
                states of model and optimizer) 
            loss: loss untils this epoch (checkpoint loss)

        """
        if not checkpoint_path:
            raise ValueError("checkpoint path is not provided") 
        try:
            checkpoint_dict = torch.load(checkpoint_path)

        except Exception as e:
            raise ValueError(f'Error loading in checkpoint from {checkpoint_path}')

        self.model.load_state_dict(checkpoint_dict[self.model_type])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        self.scheduler.load_state_dict(checkpoint_dict['scheduler'])

        return checkpoint_dict['hist_dict']



    def train_engin(
        self,
        epochs : int,
        train_dataloader : torch.utils.data.DataLoader,
        test_dataloader : torch.utils.data.DataLoader,
        checkpoint_freq = 5,
        checkpoint_to_save_path = None,
        is_checkpoint_to_load = False,
        checkpoint_to_load_path = None 
        ):

        """
        This Function Train and do validation testing, return result dictionary
        Parameter:
            epochs: type int, number of iteration,
            train_dataloader: dataloder for training data,
            test_dataloader: dataloader for testing data,
        return:
            hist_dict: type Dict, keys train_loss, train_acc, test_loss, test_acc
        """

        start_epoch = 0
        previous_loss = None
        self.model.to(self.device)
        hist_dict = {
            "epochs":0,
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
            }
        if is_checkpoint_to_load:
            if checkpoint_to_load_path is not None :
                hist_dict = self.reload_states_dict(
                    checkpoint_path = checkpoint_to_load_path
                    )
                start_epoch = hist_dict['epochs']
            else:
                raise ValueError(f"checkpoint can't be load {checkpoint_to_load_path}")
        

        
        for epoch in range(start_epoch, epochs):
            train_loss, train_acc = self.train_step(
                train_dataloader = train_dataloader,
                )
            

            
            # validation part
            test_loss, test_acc = self.test_step(
                test_dataloader = test_dataloader,
                )
            
            
            print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
            )
            hist_dict['epochs']= epoch+1
            hist_dict['train_loss'].append(train_loss)
            hist_dict['train_acc'].append(train_acc)
            hist_dict['test_loss'].append(test_loss)
            hist_dict['test_acc'].append(test_acc)

            # self.scheduler.step(train_loss)#checkpoint saving logic
            if checkpoint_to_save_path is not None:
                if (epoch+1) % checkpoint_freq == 0:
                    # logger
                    logging.info(f"Epoch: {epoch+1}/{epochs} " 
                                 f"Train Loss: {train_loss}, Train Accuracy: {train_acc} "
                                 f"Test Acurracy {test_acc}" 
                                 )
                    if previous_loss is None or train_loss < previous_loss:
                        kwarg = {
                            'model_type': self.model_type,
                            'model': self.model,
                            'optimizer' : self.optimizer,
                            'scheduler' : self.scheduler,
                            'hist_dict' : hist_dict,
                            'path': checkpoint_to_save_path
                        }

                        model_checkpoint(**kwarg)
                        previous_loss = train_loss

        return hist_dict




    



        
        