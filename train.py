import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mlp_model import MNISTMLPModel
from cnn_model import MNISTCNNModel

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 归一化
x_train, x_test = x_train / 255.0, x_test / 255.0
# 独热
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


def train_mlp():
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    model = MNISTMLPModel()

    # 编译模型
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.fit(x_train, y_train, epochs=5, batch_size=32)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    # 随机选择一些测试图像进行可视化
    num_images_to_show = 10
    random_indices = np.random.choice(len(x_test), num_images_to_show, replace=False)
    for i in random_indices:
        image_data = x_test[i].reshape(28, 28)  # 重塑为28x28图像
        true_label = np.argmax(y_test[i])
        prediction = np.argmax(model.predict(np.array([x_test[i]])))
        plt.subplot(2, 5, np.where(random_indices == i)[0][0] + 1)
        plt.imshow(image_data, cmap="gray")
        plt.title(f"True: {true_label}\nPred: {prediction}")
        plt.axis("off")
    plt.show()


def train_cnn():
    model = MNISTCNNModel()
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    x_train_expanded = np.expand_dims(x_train, -1)  # 为卷积层添加通道维度
    x_test_expanded = np.expand_dims(x_test, -1)
    model.fit(x_train_expanded, y_train, epochs=5, batch_size=32)
    test_loss, test_accuracy = model.evaluate(x_test_expanded, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)


if __name__ == "__main__":
    train_cnn()
