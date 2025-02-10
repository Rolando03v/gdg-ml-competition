import torch
import torch.nn as nn


# import test_train_split and cross validation from sklearn
from sklearn.model_selection import train_test_split

# import data processing function from prep_data.py
from prep_data import get_prepared_data

# import model from model.py
from model import create_model


# TODO modify this function however you want to train the model
def train_model(model, optimizer, criterion, X_train, y_train, X_val, y_val, training_updates=True):

    num_epochs = 500 
    best_val_loss = float("inf")
    patience, patience_counter = 20, 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % (num_epochs // 10) == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                output = model(X_val)
                val_loss = criterion(output, y_val)
                print(f"Epoch {epoch} | Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), "saved_weights/best_model.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break

    return model


# example training loop
if __name__ == '__main__':
    # Load data
    features, target = get_prepared_data()

    # create training and validation sets
    # use 80% of the data for training and 20% for validation
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

    # Define model (feed-forward, two hidden layers)
    model, optimizer = create_model(X_train)

    # Define loss function and optimizer
    criterion = nn.MSELoss()

    # train model
    model = train_model(model, optimizer, criterion, X_train, y_train, X_val, y_val)

    model.load_state_dict(torch.load("saved_weights/best_model.pth"))
    model.eval()

    # basic evaluation (more in test.py)
    with torch.no_grad():
        output = model(X_val)
        loss = criterion(output, y_val)
        print(f"Final Validation Loss: {loss.item()}")
        # validation accuracy
        print(f"Final Validation Accuracy: {1 - loss.item() / y_val.var()}")

    # Save model
    torch.save(model, "saved_weights/model.pth")
    print("Model saved as model.pth")