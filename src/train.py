import torch
from utils import checkpoint, regional_info_extraction, grad_flow
# from tqdm.auto import tqdm

class Trainer():
    """
    Trainer class 
    function in this class
    train_step(), test_step(), trainer_engin(), resume_training()
    """
    def __init__(
        self,
        encoding_model: torch.nn.Module,
        classification_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_fn: torch.nn.Module,
        device: torch.device,
        laterization_dict: dict,
        ica_model = None,
        ):
        self.encoding_model = encoding_model
        self.cls_model = classification_model
        self.ica_model = ica_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.laterization_dict = laterization_dict
        self.device = device

    
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
        self.encoding_model.train()
        self.cls_model.train()
        train_loss = 0
        train_acc = 0
        for X, y in train_dataloader:  
            X = regional_info_extraction(X, self.laterization_dict)
            X = X.to(self.device)
            y = y.to(self.device)
            X = self.encoding_model(X)
            print(X.requires_grad)
            if self.ica_model is not None:
                with torch.no_grad():
                    X = self.ica_model.fit_transform(X.to('cpu').T).T
                    X = torch.tensor(X).to(self.device)
                    print(X.requires_grad)
            print(X.requires_grad)
            yhat = self.cls_model(X)
            loss = self.loss_fn(yhat, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += float(loss)
            acc = ((yhat).argmax(dim = 1) == y).sum().item()/len(y)
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
        self.encoding_model.eval()
        self.cls_model.eval()
        test_loss = 0
        test_acc = 0
        with torch.inference_mode():
            for X, y in test_dataloader:  
                X = regional_info_extraction(X, self.laterization_dict)
                X = X.to(self.device)
                y = y.to(self.device)
                X = self.encoding_model(X)
                if self.ica_model is not None:
                    X = self.ica_model.fit_transform(X.to('cpu').T).T
                    X = X.to(self.device)
                yhat = self.cls_model(X)
                loss = self.loss_fn(yhat, y)
                test_loss += float(loss)
                acc = ((yhat).argmax(dim = 1) == y).sum().item()/len(y)
                test_acc += acc
            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
        return round(test_loss, 4), round(test_acc, 4)


    def reload_states_dict(
        self,
        checkpoint_path
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

        self.encoding_model.load_state_dict(checkpoint_dict['encoding_model'])
        self.cls_model.load_state_dict(checkpoint_dict['cls_model'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])

        return checkpoint_dict['epoch'], checkpoint_dict['loss']



    def train_engin(
        self,
        epochs : int,
        train_dataloader : torch.utils.data.DataLoader,
        test_dataloader : torch.utils.data.DataLoader,
        return_grad_flow = False,
        checkpoint_freq = 5,
        checkpoint_to_save_path = None,
        checkpoint_to_load_path = None
        ):

        """
        This Function Train and do validation testing, return result dictionary
        Parameter:
            epochs: type int, number of iteration,
            train_dataloader: dataloder for training data,
            test_dataloader: dataloader for testing data,
        return:
            res_dict: type Dict, keys train_loss, train_acc, test_loss, test_acc
        """

        start_epoch = 0
        previous_loss = None
        self.encoding_model.to(self.device)
        self.cls_model.to(self.device)

        if checkpoint_to_load_path is not None :
            start_epoch, _ = self.reload_states_dict(
                checkpoint_path = checkpoint_to_load_path
                )
        
        res_dict = {"train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
            }
        grad_flow_dict  = dict()

        for epoch in range(start_epoch, epochs):
            train_loss, train_acc = self.train_step(
                train_dataloader = train_dataloader,
                )
            
            #checkpoint saving logic
            if checkpoint_to_save_path is not None:
                if (epoch+1) % checkpoint_freq == 0:
                    if previous_loss is None or train_loss < previous_loss:
                        checkpoint(
                            {
                                'encoding_model': self.encoding_model,
                                'cls_model': self.cls_model
                            },
                            self.optimizer,
                            epoch,
                            train_loss,
                            checkpoint_to_save_path
                            )
                        previous_loss = train_loss
            
            # validation part
            test_loss, test_acc = self.test_step(
                test_dataloader = test_dataloader,
                )
            
            # grad flow part
            if return_grad_flow :
                grad_flow_dict[f'epoch_{epoch+1}'] = grad_flow(
                    list(self.encoding_model.named_parameters())+list(self.cls_model.named_parameters())
                    )
            
            print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
            )
            res_dict['train_loss'].append(train_loss)
            res_dict['train_acc'].append(train_acc)
            res_dict['test_loss'].append(test_loss)
            res_dict['test_acc'].append(test_acc)
        if return_grad_flow:
            return res_dict, grad_flow_dict
        else:
            return res_dict




    



        
        