import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from captcha.image import ImageCaptcha
import random
import string
# 1設定參數
WIDTH, HEIGHT = 64, 64        # 驗證碼圖片的寬與高
CHARACTERS = string.digits    # 定義字元集為 '0123456789'
NUM_CLASSES = len(CHARACTERS) # 總共 10 個類別
EPOCHS = 10                   # 訓練週期數
BATCH_SIZE = 32               # 批次大小


#建立資料生成器 (Data Generation)

def generate_dataset(num_samples):
    """
    使用 captcha 函式庫隨機生成帶有干擾的 1 位數驗證碼
    """
    # 建立產生器物件
    generator = ImageCaptcha(width=WIDTH, height=HEIGHT)
    
    # 初始化儲存圖片(X)與標籤(y)的陣列
    X = np.zeros((num_samples, HEIGHT, WIDTH, 3), dtype=np.float32)
    y = np.zeros((num_samples,), dtype=np.int32)
    
    for i in range(num_samples):
        # 隨機挑選一個數字 (0-9)
        random_digit = random.choice(CHARACTERS)
        
        # 產生圖片 (回傳為 PIL Image 格式)
        img = generator.generate_image(random_digit)
        
        # 將圖片轉換為 numpy 陣列，並將像素值正規化到 0 ~ 1 之間
        img_array = np.array(img) / 255.0 
        
        X[i] = img_array
        y[i] = int(random_digit)
        
    return X, y

print("正在生成訓練集與測試集資料，請稍候...")
X_train, y_train = generate_dataset(2000) # 生成 2000 張訓練圖
X_test, y_test = generate_dataset(400)    # 生成 400 張測試圖
print("資料生成完畢！")

# 2建立 CNN 卷積神經網路模型
model = models.Sequential([
    # 第一層卷積與池化 (提取邊緣等低階特徵)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(HEIGHT, WIDTH, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # 第二層卷積與池化 (提取形狀等中階特徵)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # 第三層卷積 (提取高階特徵)
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # 攤平層 (將 2D 特徵圖攤平為 1D 向量)
    layers.Flatten(),
    
    # 全連接層
    layers.Dense(64, activation='relu'),
    
    # Dropout層 (隨機丟棄 50% 神經元，防止模型死背資料/過擬合)
    layers.Dropout(0.5),
    
    # 輸出層 (10 個神經元代表 0-9，使用 softmax 轉為機率分布)
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# 3編譯與訓練模型
# 因為我們的標籤(y)是整數型態 (0, 1, 2...)，所以使用 sparse_categorical_crossentropy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\n開始訓練模型...")
history = model.fit(X_train, y_train, 
                    epochs=EPOCHS, 
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test))

#測試並視覺化預測結果

def predict_and_show(index):
    # 取出一張測試圖片
    img = X_test[index]
    true_label = y_test[index]
    
    # 由於 Keras 預測時預期輸入是批次 (Batch) 格式
    # 我們需要用 np.expand_dims 增加一個維度：從 (64, 64, 3) 變成 (1, 64, 64, 3)
    img_expanded = np.expand_dims(img, axis=0)
    
    # 進行預測
    prediction = model.predict(img_expanded, verbose=0)
    
    # 取出機率最高的類別作為預測結果
    predicted_label = np.argmax(prediction)
    
    # 畫出圖片與結果
    plt.imshow(img)
    plt.title(f"True: {true_label} | Predict: {predicted_label}", 
              color='green' if true_label == predicted_label else 'red',
              fontsize=14)
    plt.axis('off')
    plt.show()

# 4隨機挑選 3 張測試集圖片來看看模型的表現
print("\n展示預測結果：")
for _ in range(3):
    random_idx = random.randint(0, len(X_test)-1)
    predict_and_show(random_idx)

# 5儲存整個模型
model.save('my_captcha_model.keras')
print("模型已成功儲存！")
