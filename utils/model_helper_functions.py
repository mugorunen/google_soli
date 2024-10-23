import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ModelHelper:
    """Helper functions for training and evaluating the model"""
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def prepare_for_qat(self):
        """
        Prepare the model for quantization aware training
        """
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

        torch.quantization.prepare_qat(self.model, inplace=True)
        print("Model prepared for quantization aware training")

    def convert_to_quantized(self):
        """
        Convert the trained QAT model to a quantized model

        Returns:
        - model_quantized: Quantized model
        """
        self.model.to("cpu")
        self.model.eval()
        model_quantized = torch.quantization.convert(self.model, inplace=False)
        print("Model converted to quantized version")
        # Move back to the original device (CUDA)
        self.model.to(self.device)
        return model_quantized

    def train_model(self, train_loader, val_loader, num_epochs=20, learning_rate=0.001, mult_learning_rate=0.1):
        """
        To train the model

        :param train_loader: Train Dataloader
        :param val_loader: Validation Dataloader
        :param num_epochs (int): Number of epochs to train for
        :param learning_rate (float): Initial learning rate
        :param mult_learning_rate (float): To update learning rate

        Returns:
        - train_losses (list): Train lost per epoch
        - val_losses (list): Validation loss per epoch
        - train_accuracies (list): Training accuracy per epoch
        - val_accuracies (list): Validation accuracy per epoch
        """
        criterion = nn.CrossEntropyLoss()  # Loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)  # Optimizer

        self.model.to(self.device)

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        print("Learning rate: ", learning_rate)

        for epoch in range(num_epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0
            correct = 0
            total = 0

            if epoch == 20 or epoch == 30:
                # Update learning rate in epoch 20 and 30
                learning_rate = learning_rate*mult_learning_rate
                optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
                print("New learning rate: ", learning_rate)

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)  # Move inputs to the device
                labels = labels.to(self.device)    # Move labels to the device

                optimizer.zero_grad()  # Zero the parameter gradients
                
                # Forward pass
                outputs = self.model(inputs) 
                # Calculate loss 
                loss = criterion(outputs, labels)  

                # Backward pass
                loss.backward() 
                # Update parameters 
                optimizer.step()  

                running_loss += loss.item()

                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Calculate average training loss and accuracy
            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            train_accuracy = 100 * correct / total
            train_accuracies.append(train_accuracy)

            # Evaluate the model on the validation set
            val_accuracy, avg_val_loss, _, _ = self.evaluate_model(self.model, val_loader, self.device)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            # Print training and validation metrics
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, '
                  f'Train Accuracy: {train_accuracy:.2f}%, '
                  f'Validation Loss: {avg_val_loss:.4f}, '
                  f'Validation Accuracy: {val_accuracy:.2f}%')

        return train_losses, val_losses, train_accuracies, val_accuracies

    def evaluate_model(self, model_ev, loader, device):  
        """
        To evaluate model on different dataloader

        :param model_ev: Model to evaluate
        :param loader: Dataloader to evaluate
        :param device: Device to move the model and data to

        Returns:
        - accuracy (float): Accuracy of the model on the dataset
        - avg_loss (float): Average loss of the model on the dataset
        - all_labels (numpy array): Actual labels of the dataset
        - all_preds (numpy array): Predicted labels of the dataset
        """          
        model_ev.to(device)
        model_ev.eval() 
        all_labels = []
        all_preds = []
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()  # Loss function

        with torch.no_grad():  # Disable gradient tracking
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model_ev(inputs)
                loss = criterion(outputs, labels) 
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                # Store predictions
                all_preds.extend(predicted.cpu().numpy())  
                # Store actual labels
                all_labels.extend(labels.cpu().numpy())    
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(loader)

        return accuracy, avg_loss, np.array(all_labels), np.array(all_preds)