import copy

import numpy as np
import torch
import torch.nn.functional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class LogisticRegression:

    def __init__(
            self,
            num_features,
            max_epochs=500,
            minibatch_size=64,
            # validation_size=23680,  # 資料筆數 非比例(20%驗證)
            validation_size=2048,
            learning_rate=1e-4,
            patience_lr=5,  # 50 minibatches
            patience=20,  # 100 minibatches
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.name = "LogisticRegression"
        self.args = {
            "num_features": num_features,
            "validation_size": validation_size,
            "minibatch_size": minibatch_size,
            "lr": learning_rate,
            "max_epochs": max_epochs,
            "patience_lr": patience_lr,
            "patience": patience,
        }

        self.model = None
        self.device = device
        self.classes = None
        self.scaler = None
        self.num_classes = None

    def fit(self, x_train, y_train):
        self.classes = np.unique(y_train)
        self.num_classes = len(self.classes)

        num_outputs = self.num_classes if self.num_classes > 2 else 1
        train_steps = int(x_train.shape[0] / self.args["minibatch_size"])

        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x_train)

        model = torch.nn.Sequential(
            torch.nn.Dropout(0.6),  # dropout 0.6
            torch.nn.Linear(self.args["num_features"], num_outputs)).to(self.device)

        if num_outputs == 1:
            loss_function = torch.nn.BCEWithLogitsLoss()
        else:
            loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            min_lr=1e-8,
            patience=self.args["patience_lr"]
        )

        training_size = x_train.shape[0]
        if self.args["validation_size"] < training_size:
            x_training, x_validation, y_training, y_validation = train_test_split(
                x_train, y_train,
                test_size=self.args["validation_size"],
                stratify=y_train
            )

            # 資料保持在 CPU 上
            train_data = TensorDataset(
                torch.tensor(x_training, dtype=torch.float32),  # 留在 CPU
                torch.tensor(y_training, dtype=torch.long)  # 留在 CPU
            )
            val_data = TensorDataset(
                torch.tensor(x_validation, dtype=torch.float32),  # 留在 CPU
                torch.tensor(y_validation, dtype=torch.long)  # 留在 CPU
            )

            # DataLoader 不變，資料批次大小設定
            train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.args["minibatch_size"])
            val_dataloader = DataLoader(val_data, batch_size=self.args["minibatch_size"])

            # 在訓練迴圈內，逐批將資料移到 GPU
            for epoch in range(self.args["max_epochs"]):
                model.train()
                for i, data in tqdm(enumerate(train_dataloader), desc=f"epoch: {epoch}"):
                    # 每個批次的資料移到 GPU
                    x_batch, y_batch = data
                    x_batch = x_batch.to(self.device)  # 逐批轉移到 GPU
                    y_batch = y_batch.to(self.device)  # 逐批轉移到 GPU

                    # 前向傳播
                    y_hat = model(x_batch)

                    # 如果使用的是二元分類，並且輸出是 [64, 1]，則需要擴展 y_batch 維度
                    if y_hat.size(1) == 1:  # 如果模型輸出為 (batch_size, 1)
                        y_batch = y_batch.unsqueeze(1)  # 使 y_batch 的形狀與 y_hat 相同

                    # 將 y_batch 轉換為 float 型別以匹配 BCEWithLogitsLoss 的輸入需求
                    y_batch = y_batch.float()

                    loss = loss_function(y_hat, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # 驗證階段（如果有驗證集）
                if val_dataloader is not None:
                    model.eval()
                    with torch.no_grad():  # 關閉梯度計算
                        total_val_loss = 0
                        for x_val_batch, y_val_batch in val_dataloader:
                            # 每個批次的驗證資料移到 GPU
                            x_val_batch = x_val_batch.to(self.device)
                            y_val_batch = y_val_batch.to(self.device)

                            # 驗證前向傳播
                            val_output = model(x_val_batch)

                            if val_output.size(1) == 1:  # 如果模型輸出為 (batch_size, 1)
                                y_val_batch = y_val_batch.unsqueeze(1)  # 使 y_batch 的形狀與 y_hat 相同

                            # 將 y_batch 轉換為 float 型別以匹配 BCEWithLogitsLoss 的輸入需求
                            y_val_batch = y_val_batch.float()

                            val_loss = loss_function(val_output, y_val_batch)
                            total_val_loss += val_loss.item()

        else:
            train_data = TensorDataset(
                torch.tensor(x_train, dtype=torch.float32).to(self.device),
                torch.tensor(y_train, dtype=torch.long).to(self.device)
            )
            train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.args["minibatch_size"])
            val_dataloader = None
 #           val_dataloader = DataLoader(val_data, batch_size=self.args["minibatch_size"])

        best_loss = np.inf
        best_model = None
        stall_count = 0
        stop = False

        for epoch in range(self.args["max_epochs"]):
            if epoch > 0 and stop:
                break
            model.train()

            # loop over the training set
            total_train_loss = 0
            steps = 0
            for i, data in tqdm(enumerate(train_dataloader), desc=f"epoch: {epoch}", total=train_steps):
                x, y = data

                y_hat = model(x)
                if num_outputs == 1:

                    ysize = y.size(0)
                    y = torch.reshape(y, (ysize,1)) #這幾行是我加來解決pytorch中損失函數中輸入輸出不匹配問題
                    loss = loss_function(y_hat.sigmoid(), y.float()) #y改成y.float()解決Pytorch: RuntimeError: result type Float can't be cast to the desired output type Long
                else:
                    yhat = torch.nn.functional.softmax(y_hat, dim=1)
                    loss = loss_function(yhat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss
                steps += 1

            total_train_loss = total_train_loss.cpu().detach().numpy() / steps

            if val_dataloader is not None:
                total_val_loss = 0
                # switch off autograd for evaluation
                with torch.no_grad():
                    # set the model in evaluation mode
                    model.eval()
                    for i, data in enumerate(val_dataloader):
                        x, y = data

                        y_hat = model(x)
                        if num_outputs == 1:
                            ysize = y.size(0)
                            y = torch.reshape(y, (ysize, 1))  # 這幾行是我加來解決pytorch中損失函數中輸入輸出不匹配問題
                            total_val_loss += loss_function(y_hat.sigmoid(), y.float()) #y改成y.float()解決Pytorch: RuntimeError: result type Float can't be cast to the desired output type Long
                        else:
                            yhat = torch.nn.functional.softmax(y_hat, dim=1)
                            total_val_loss += loss_function(yhat, y)
                total_val_loss = total_val_loss.cpu().detach().numpy() / steps
                scheduler.step(total_val_loss)

                if total_val_loss >= best_loss:
                    stall_count += 1
                    if stall_count >= self.args["patience"]:
                        stop = True
                        print(f"\n<Stopped at Epoch {epoch + 1}>")
                else:
                    best_loss = total_val_loss
                    best_model = copy.deepcopy(model)
                    if not stop:
                        stall_count = 0
            else:
                scheduler.step(total_train_loss)
                if total_train_loss >= best_loss:
                    stall_count += 1
                    if stall_count >= self.args["patience"]:
                        stop = True
                        print(f"\n<Stopped at Epoch {epoch + 1}>")
                else:
                    best_loss = total_train_loss
                    best_model = copy.deepcopy(model)
                    if not stop:
                        stall_count = 0

        self.model = best_model
        return self.model

    def predict(self, x):
        x = self.scaler.transform(x)

        with torch.no_grad():
            # set the model in evaluation mode
            self.model.eval()

            yhat = self.model(torch.tensor(x, dtype=torch.float32).to(self.device))

            if self.num_classes > 2:
                yhat = self.classes[np.argmax(yhat.cpu().detach().numpy(), axis=1)]
            else:
                yhat = torch.sigmoid(yhat)
####
#                yhat_original_output = yhat.cpu().detach().numpy()
#                yhat = np.round(yhat.cpu().detach().numpy())

                yhat = yhat.cpu().detach().numpy()
####為了顯示原本的預測值做的修改

            return yhat
