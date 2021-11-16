{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>1. Import module yang dibutuhkan</h3>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.metrics import classification_report\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import InputLayer, Flatten, Dense, Conv2D, MaxPool2D, Dropout\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "import numpy as np\n",
    "import cv2\n",
    "import glob\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>2. Load Dataset</h3>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "imagePaths = 'dataset\\\\daun\\\\'\n",
    "label_list = ['Jeruk Nipis', 'Sirih']\n",
    "data = []\n",
    "labels = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "for label in label_list:\n",
    "    for imagePath in glob.glob(imagePaths+label+'\\\\*.jpg'):\n",
    "        #print(imagePath)\n",
    "        image = cv2.imread(imagePath)\n",
    "        image = cv2.resize(image, (32, 32))\n",
    "        data.append(image)\n",
    "        labels.append(label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(100, 32, 32, 3)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.array(data).shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>3. Data Preprocessing</h3>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ubah type data dari list menjadi array\n",
    "# ubah nilai dari tiap pixel menjadi range [0..1]\n",
    "data = np.array(data, dtype='float') / 255.0\n",
    "labels = np.array(labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Jeruk Nipis', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih', 'Sirih']\n"
     ]
    }
   ],
   "source": [
    "print(labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n",
      " 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
      " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
     ]
    }
   ],
   "source": [
    "# ubah nilai dari labels menjadi binary\n",
    "lb = LabelEncoder()\n",
    "labels = lb.fit_transform(labels)\n",
    "print(labels)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>4. Split Dataset</h3>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Ukuran data train = (80, 32, 32, 3)\n",
      "Ukuran data test = (20, 32, 32, 3)\n"
     ]
    }
   ],
   "source": [
    "print('Ukuran data train =', x_train.shape)\n",
    "print('Ukuran data test =', x_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>5. Build CNN Architecture</h3>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "# Extracted Feature Layer\n",
    "model.add(InputLayer(input_shape=[32,32,3]))\n",
    "model.add(Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu'))\n",
    "model.add(MaxPool2D(pool_size=2, padding='same'))\n",
    "model.add(Conv2D(filters=50, kernel_size=2, strides=1, padding='same', activation='relu'))\n",
    "model.add(MaxPool2D(pool_size=2, padding='same'))\n",
    "model.add(Dropout(0.25))\n",
    "model.add(Flatten())\n",
    "# Fully Connected Layer\n",
    "model.add(Dense(512, activation='relu'))\n",
    "model.add(Dropout(0.5))\n",
    "model.add(Dense(1, activation='sigmoid'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " conv2d (Conv2D)             (None, 32, 32, 32)        416       \n",
      "                                                                 \n",
      " max_pooling2d (MaxPooling2D  (None, 16, 16, 32)       0         \n",
      " )                                                               \n",
      "                                                                 \n",
      " conv2d_1 (Conv2D)           (None, 16, 16, 50)        6450      \n",
      "                                                                 \n",
      " max_pooling2d_1 (MaxPooling  (None, 8, 8, 50)         0         \n",
      " 2D)                                                             \n",
      "                                                                 \n",
      " dropout (Dropout)           (None, 8, 8, 50)          0         \n",
      "                                                                 \n",
      " flatten (Flatten)           (None, 3200)              0         \n",
      "                                                                 \n",
      " dense (Dense)               (None, 512)               1638912   \n",
      "                                                                 \n",
      " dropout_1 (Dropout)         (None, 512)               0         \n",
      "                                                                 \n",
      " dense_1 (Dense)             (None, 1)                 513       \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 1,646,291\n",
      "Trainable params: 1,646,291\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# tentukan hyperparameter\n",
    "lr = 0.001\n",
    "max_epochs = 100\n",
    "opt_funct = Adam(learning_rate=lr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# compile arsitektur yang telah dibuat\n",
    "model.compile(loss = 'binary_crossentropy', \n",
    "              optimizer = opt_funct, \n",
    "              metrics = ['accuracy'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>6. Train Model</h3>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100\n",
      "3/3 [==============================] - 2s 209ms/step - loss: 0.8073 - accuracy: 0.5000 - val_loss: 0.6746 - val_accuracy: 0.6000\n",
      "Epoch 2/100\n",
      "3/3 [==============================] - 0s 53ms/step - loss: 0.7778 - accuracy: 0.4875 - val_loss: 0.6219 - val_accuracy: 0.6000\n",
      "Epoch 3/100\n",
      "3/3 [==============================] - 0s 54ms/step - loss: 0.6232 - accuracy: 0.7375 - val_loss: 0.6759 - val_accuracy: 0.4000\n",
      "Epoch 4/100\n",
      "3/3 [==============================] - 0s 55ms/step - loss: 0.6130 - accuracy: 0.6875 - val_loss: 0.6290 - val_accuracy: 0.5500\n",
      "Epoch 5/100\n",
      "3/3 [==============================] - 0s 47ms/step - loss: 0.5921 - accuracy: 0.7375 - val_loss: 0.5216 - val_accuracy: 0.9500\n",
      "Epoch 6/100\n",
      "3/3 [==============================] - 0s 43ms/step - loss: 0.5156 - accuracy: 0.8875 - val_loss: 0.4569 - val_accuracy: 0.9500\n",
      "Epoch 7/100\n",
      "3/3 [==============================] - 0s 47ms/step - loss: 0.4253 - accuracy: 0.9250 - val_loss: 0.3832 - val_accuracy: 1.0000\n",
      "Epoch 8/100\n",
      "3/3 [==============================] - 0s 48ms/step - loss: 0.3898 - accuracy: 0.9125 - val_loss: 0.3731 - val_accuracy: 0.8500\n",
      "Epoch 9/100\n",
      "3/3 [==============================] - 0s 43ms/step - loss: 0.3414 - accuracy: 0.8750 - val_loss: 0.2350 - val_accuracy: 1.0000\n",
      "Epoch 10/100\n",
      "3/3 [==============================] - 0s 45ms/step - loss: 0.2475 - accuracy: 0.9750 - val_loss: 0.2123 - val_accuracy: 1.0000\n",
      "Epoch 11/100\n",
      "3/3 [==============================] - 0s 55ms/step - loss: 0.1931 - accuracy: 0.9750 - val_loss: 0.1422 - val_accuracy: 1.0000\n",
      "Epoch 12/100\n",
      "3/3 [==============================] - 0s 52ms/step - loss: 0.1608 - accuracy: 0.9625 - val_loss: 0.1341 - val_accuracy: 1.0000\n",
      "Epoch 13/100\n",
      "3/3 [==============================] - 0s 46ms/step - loss: 0.1348 - accuracy: 0.9750 - val_loss: 0.1026 - val_accuracy: 1.0000\n",
      "Epoch 14/100\n",
      "3/3 [==============================] - 0s 45ms/step - loss: 0.0966 - accuracy: 0.9875 - val_loss: 0.1001 - val_accuracy: 0.9500\n",
      "Epoch 15/100\n",
      "3/3 [==============================] - 0s 48ms/step - loss: 0.0906 - accuracy: 0.9875 - val_loss: 0.0543 - val_accuracy: 1.0000\n",
      "Epoch 16/100\n",
      "3/3 [==============================] - 0s 53ms/step - loss: 0.0630 - accuracy: 0.9875 - val_loss: 0.0538 - val_accuracy: 1.0000\n",
      "Epoch 17/100\n",
      "3/3 [==============================] - 0s 60ms/step - loss: 0.0554 - accuracy: 1.0000 - val_loss: 0.0364 - val_accuracy: 1.0000\n",
      "Epoch 18/100\n",
      "3/3 [==============================] - 0s 59ms/step - loss: 0.0372 - accuracy: 1.0000 - val_loss: 0.0705 - val_accuracy: 0.9500\n",
      "Epoch 19/100\n",
      "3/3 [==============================] - 0s 51ms/step - loss: 0.0385 - accuracy: 1.0000 - val_loss: 0.0241 - val_accuracy: 1.0000\n",
      "Epoch 20/100\n",
      "3/3 [==============================] - 0s 54ms/step - loss: 0.0535 - accuracy: 0.9750 - val_loss: 0.0932 - val_accuracy: 0.9500\n",
      "Epoch 21/100\n",
      "3/3 [==============================] - 0s 48ms/step - loss: 0.0646 - accuracy: 0.9875 - val_loss: 0.0174 - val_accuracy: 1.0000\n",
      "Epoch 22/100\n",
      "3/3 [==============================] - 0s 56ms/step - loss: 0.0222 - accuracy: 1.0000 - val_loss: 0.0269 - val_accuracy: 1.0000\n",
      "Epoch 23/100\n",
      "3/3 [==============================] - 0s 70ms/step - loss: 0.0228 - accuracy: 1.0000 - val_loss: 0.0323 - val_accuracy: 1.0000\n",
      "Epoch 24/100\n",
      "3/3 [==============================] - 0s 48ms/step - loss: 0.0110 - accuracy: 1.0000 - val_loss: 0.0122 - val_accuracy: 1.0000\n",
      "Epoch 25/100\n",
      "3/3 [==============================] - 0s 65ms/step - loss: 0.0207 - accuracy: 1.0000 - val_loss: 0.0268 - val_accuracy: 1.0000\n",
      "Epoch 26/100\n",
      "3/3 [==============================] - 0s 60ms/step - loss: 0.0186 - accuracy: 1.0000 - val_loss: 0.0126 - val_accuracy: 1.0000\n",
      "Epoch 27/100\n",
      "3/3 [==============================] - 0s 51ms/step - loss: 0.0142 - accuracy: 1.0000 - val_loss: 0.0155 - val_accuracy: 1.0000\n",
      "Epoch 28/100\n",
      "3/3 [==============================] - 0s 48ms/step - loss: 0.0128 - accuracy: 1.0000 - val_loss: 0.0123 - val_accuracy: 1.0000\n",
      "Epoch 29/100\n",
      "3/3 [==============================] - 0s 49ms/step - loss: 0.0090 - accuracy: 1.0000 - val_loss: 0.0114 - val_accuracy: 1.0000\n",
      "Epoch 30/100\n",
      "3/3 [==============================] - 0s 43ms/step - loss: 0.0102 - accuracy: 1.0000 - val_loss: 0.0450 - val_accuracy: 0.9500\n",
      "Epoch 31/100\n",
      "3/3 [==============================] - 0s 45ms/step - loss: 0.0100 - accuracy: 1.0000 - val_loss: 0.0115 - val_accuracy: 1.0000\n",
      "Epoch 32/100\n",
      "3/3 [==============================] - 0s 50ms/step - loss: 0.0117 - accuracy: 1.0000 - val_loss: 0.0072 - val_accuracy: 1.0000\n",
      "Epoch 33/100\n",
      "3/3 [==============================] - 0s 44ms/step - loss: 0.0045 - accuracy: 1.0000 - val_loss: 0.0617 - val_accuracy: 0.9500\n",
      "Epoch 34/100\n",
      "3/3 [==============================] - 0s 44ms/step - loss: 0.0114 - accuracy: 1.0000 - val_loss: 0.0094 - val_accuracy: 1.0000\n",
      "Epoch 35/100\n",
      "3/3 [==============================] - 0s 42ms/step - loss: 0.0096 - accuracy: 1.0000 - val_loss: 0.0044 - val_accuracy: 1.0000\n",
      "Epoch 36/100\n",
      "3/3 [==============================] - 0s 43ms/step - loss: 0.0100 - accuracy: 1.0000 - val_loss: 0.1062 - val_accuracy: 0.9500\n",
      "Epoch 37/100\n",
      "3/3 [==============================] - 0s 45ms/step - loss: 0.0242 - accuracy: 1.0000 - val_loss: 0.0045 - val_accuracy: 1.0000\n",
      "Epoch 38/100\n",
      "3/3 [==============================] - 0s 46ms/step - loss: 0.0446 - accuracy: 0.9875 - val_loss: 0.0422 - val_accuracy: 1.0000\n",
      "Epoch 39/100\n",
      "3/3 [==============================] - 0s 44ms/step - loss: 0.0711 - accuracy: 0.9625 - val_loss: 0.0050 - val_accuracy: 1.0000\n",
      "Epoch 40/100\n",
      "3/3 [==============================] - 0s 56ms/step - loss: 0.0406 - accuracy: 0.9875 - val_loss: 0.0070 - val_accuracy: 1.0000\n",
      "Epoch 41/100\n",
      "3/3 [==============================] - 0s 58ms/step - loss: 0.0045 - accuracy: 1.0000 - val_loss: 0.0516 - val_accuracy: 0.9500\n",
      "Epoch 42/100\n",
      "3/3 [==============================] - 0s 59ms/step - loss: 0.0158 - accuracy: 1.0000 - val_loss: 0.0192 - val_accuracy: 1.0000\n",
      "Epoch 43/100\n",
      "3/3 [==============================] - 0s 51ms/step - loss: 0.0066 - accuracy: 1.0000 - val_loss: 0.0048 - val_accuracy: 1.0000\n",
      "Epoch 44/100\n",
      "3/3 [==============================] - 0s 48ms/step - loss: 0.0100 - accuracy: 1.0000 - val_loss: 0.0067 - val_accuracy: 1.0000\n",
      "Epoch 45/100\n",
      "3/3 [==============================] - 0s 46ms/step - loss: 0.0063 - accuracy: 1.0000 - val_loss: 0.0313 - val_accuracy: 1.0000\n",
      "Epoch 46/100\n",
      "3/3 [==============================] - 0s 53ms/step - loss: 0.0040 - accuracy: 1.0000 - val_loss: 0.0140 - val_accuracy: 1.0000\n",
      "Epoch 47/100\n",
      "3/3 [==============================] - 0s 58ms/step - loss: 0.0039 - accuracy: 1.0000 - val_loss: 0.0063 - val_accuracy: 1.0000\n",
      "Epoch 48/100\n",
      "3/3 [==============================] - 0s 49ms/step - loss: 0.0032 - accuracy: 1.0000 - val_loss: 0.0035 - val_accuracy: 1.0000\n",
      "Epoch 49/100\n",
      "3/3 [==============================] - 0s 50ms/step - loss: 0.0057 - accuracy: 1.0000 - val_loss: 0.0064 - val_accuracy: 1.0000\n",
      "Epoch 50/100\n",
      "3/3 [==============================] - 0s 43ms/step - loss: 0.0022 - accuracy: 1.0000 - val_loss: 0.0235 - val_accuracy: 1.0000\n",
      "Epoch 51/100\n",
      "3/3 [==============================] - 0s 46ms/step - loss: 0.0040 - accuracy: 1.0000 - val_loss: 0.0150 - val_accuracy: 1.0000\n",
      "Epoch 52/100\n",
      "3/3 [==============================] - 0s 45ms/step - loss: 0.0038 - accuracy: 1.0000 - val_loss: 0.0028 - val_accuracy: 1.0000\n",
      "Epoch 53/100\n",
      "3/3 [==============================] - 0s 45ms/step - loss: 0.0033 - accuracy: 1.0000 - val_loss: 0.0020 - val_accuracy: 1.0000\n",
      "Epoch 54/100\n",
      "3/3 [==============================] - 0s 45ms/step - loss: 0.0027 - accuracy: 1.0000 - val_loss: 0.0039 - val_accuracy: 1.0000\n",
      "Epoch 55/100\n",
      "3/3 [==============================] - 0s 45ms/step - loss: 0.0034 - accuracy: 1.0000 - val_loss: 0.0091 - val_accuracy: 1.0000\n",
      "Epoch 56/100\n",
      "3/3 [==============================] - 0s 51ms/step - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.0123 - val_accuracy: 1.0000\n",
      "Epoch 57/100\n",
      "3/3 [==============================] - 0s 45ms/step - loss: 0.0031 - accuracy: 1.0000 - val_loss: 0.0059 - val_accuracy: 1.0000\n",
      "Epoch 58/100\n",
      "3/3 [==============================] - 0s 49ms/step - loss: 0.0028 - accuracy: 1.0000 - val_loss: 0.0017 - val_accuracy: 1.0000\n",
      "Epoch 59/100\n",
      "3/3 [==============================] - 0s 51ms/step - loss: 0.0025 - accuracy: 1.0000 - val_loss: 0.0012 - val_accuracy: 1.0000\n",
      "Epoch 60/100\n",
      "3/3 [==============================] - 0s 46ms/step - loss: 0.0029 - accuracy: 1.0000 - val_loss: 0.0022 - val_accuracy: 1.0000\n",
      "Epoch 61/100\n",
      "3/3 [==============================] - 0s 46ms/step - loss: 0.0015 - accuracy: 1.0000 - val_loss: 0.0083 - val_accuracy: 1.0000\n",
      "Epoch 62/100\n",
      "3/3 [==============================] - 0s 50ms/step - loss: 0.0018 - accuracy: 1.0000 - val_loss: 0.0127 - val_accuracy: 1.0000\n",
      "Epoch 63/100\n",
      "3/3 [==============================] - 0s 49ms/step - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.0132 - val_accuracy: 1.0000\n",
      "Epoch 64/100\n",
      "3/3 [==============================] - 0s 47ms/step - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.0059 - val_accuracy: 1.0000\n",
      "Epoch 65/100\n",
      "3/3 [==============================] - 0s 46ms/step - loss: 0.0013 - accuracy: 1.0000 - val_loss: 0.0023 - val_accuracy: 1.0000\n",
      "Epoch 66/100\n",
      "3/3 [==============================] - 0s 46ms/step - loss: 9.3858e-04 - accuracy: 1.0000 - val_loss: 0.0019 - val_accuracy: 1.0000\n",
      "Epoch 67/100\n",
      "3/3 [==============================] - 0s 46ms/step - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.0019 - val_accuracy: 1.0000\n",
      "Epoch 68/100\n",
      "3/3 [==============================] - 0s 47ms/step - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.0027 - val_accuracy: 1.0000\n",
      "Epoch 69/100\n",
      "3/3 [==============================] - 0s 48ms/step - loss: 9.9204e-04 - accuracy: 1.0000 - val_loss: 0.0049 - val_accuracy: 1.0000\n",
      "Epoch 70/100\n",
      "3/3 [==============================] - 0s 45ms/step - loss: 8.4197e-04 - accuracy: 1.0000 - val_loss: 0.0072 - val_accuracy: 1.0000\n",
      "Epoch 71/100\n",
      "3/3 [==============================] - 0s 45ms/step - loss: 8.9120e-04 - accuracy: 1.0000 - val_loss: 0.0067 - val_accuracy: 1.0000\n",
      "Epoch 72/100\n",
      "3/3 [==============================] - 0s 48ms/step - loss: 0.0015 - accuracy: 1.0000 - val_loss: 0.0032 - val_accuracy: 1.0000\n",
      "Epoch 73/100\n",
      "3/3 [==============================] - 0s 50ms/step - loss: 9.2946e-04 - accuracy: 1.0000 - val_loss: 0.0019 - val_accuracy: 1.0000\n",
      "Epoch 74/100\n",
      "3/3 [==============================] - 0s 37ms/step - loss: 0.0013 - accuracy: 1.0000 - val_loss: 0.0016 - val_accuracy: 1.0000\n",
      "Epoch 75/100\n",
      "3/3 [==============================] - 0s 40ms/step - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.0018 - val_accuracy: 1.0000\n",
      "Epoch 76/100\n",
      "3/3 [==============================] - 0s 44ms/step - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.0024 - val_accuracy: 1.0000\n",
      "Epoch 77/100\n",
      "3/3 [==============================] - 0s 49ms/step - loss: 8.1080e-04 - accuracy: 1.0000 - val_loss: 0.0031 - val_accuracy: 1.0000\n",
      "Epoch 78/100\n",
      "3/3 [==============================] - 0s 47ms/step - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.0032 - val_accuracy: 1.0000\n",
      "Epoch 79/100\n",
      "3/3 [==============================] - 0s 50ms/step - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.0033 - val_accuracy: 1.0000\n",
      "Epoch 80/100\n",
      "3/3 [==============================] - 0s 52ms/step - loss: 4.5449e-04 - accuracy: 1.0000 - val_loss: 0.0045 - val_accuracy: 1.0000\n",
      "Epoch 81/100\n",
      "3/3 [==============================] - 0s 45ms/step - loss: 6.1912e-04 - accuracy: 1.0000 - val_loss: 0.0049 - val_accuracy: 1.0000\n",
      "Epoch 82/100\n",
      "3/3 [==============================] - 0s 45ms/step - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.0037 - val_accuracy: 1.0000\n",
      "Epoch 83/100\n",
      "3/3 [==============================] - 0s 48ms/step - loss: 0.0018 - accuracy: 1.0000 - val_loss: 0.0015 - val_accuracy: 1.0000\n",
      "Epoch 84/100\n",
      "3/3 [==============================] - 0s 50ms/step - loss: 9.6018e-04 - accuracy: 1.0000 - val_loss: 9.4505e-04 - val_accuracy: 1.0000\n",
      "Epoch 85/100\n",
      "3/3 [==============================] - 0s 48ms/step - loss: 0.0015 - accuracy: 1.0000 - val_loss: 0.0012 - val_accuracy: 1.0000\n",
      "Epoch 86/100\n",
      "3/3 [==============================] - 0s 50ms/step - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.0041 - val_accuracy: 1.0000\n",
      "Epoch 87/100\n",
      "3/3 [==============================] - 0s 56ms/step - loss: 0.0025 - accuracy: 1.0000 - val_loss: 0.0046 - val_accuracy: 1.0000\n",
      "Epoch 88/100\n",
      "3/3 [==============================] - 0s 49ms/step - loss: 0.0010 - accuracy: 1.0000 - val_loss: 0.0015 - val_accuracy: 1.0000\n",
      "Epoch 89/100\n",
      "3/3 [==============================] - 0s 49ms/step - loss: 0.0010 - accuracy: 1.0000 - val_loss: 0.0013 - val_accuracy: 1.0000\n",
      "Epoch 90/100\n",
      "3/3 [==============================] - 0s 41ms/step - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.0040 - val_accuracy: 1.0000\n",
      "Epoch 91/100\n",
      "3/3 [==============================] - 0s 47ms/step - loss: 8.8639e-04 - accuracy: 1.0000 - val_loss: 0.0088 - val_accuracy: 1.0000\n",
      "Epoch 92/100\n",
      "3/3 [==============================] - 0s 42ms/step - loss: 0.0018 - accuracy: 1.0000 - val_loss: 0.0060 - val_accuracy: 1.0000\n",
      "Epoch 93/100\n",
      "3/3 [==============================] - 0s 48ms/step - loss: 6.9986e-04 - accuracy: 1.0000 - val_loss: 0.0028 - val_accuracy: 1.0000\n",
      "Epoch 94/100\n",
      "3/3 [==============================] - 0s 46ms/step - loss: 3.9569e-04 - accuracy: 1.0000 - val_loss: 0.0015 - val_accuracy: 1.0000\n",
      "Epoch 95/100\n",
      "3/3 [==============================] - 0s 50ms/step - loss: 4.1034e-04 - accuracy: 1.0000 - val_loss: 0.0011 - val_accuracy: 1.0000\n",
      "Epoch 96/100\n",
      "3/3 [==============================] - 0s 45ms/step - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.0019 - val_accuracy: 1.0000\n",
      "Epoch 97/100\n",
      "3/3 [==============================] - 0s 49ms/step - loss: 5.8818e-04 - accuracy: 1.0000 - val_loss: 0.0034 - val_accuracy: 1.0000\n",
      "Epoch 98/100\n",
      "3/3 [==============================] - 0s 37ms/step - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.0045 - val_accuracy: 1.0000\n",
      "Epoch 99/100\n",
      "3/3 [==============================] - 0s 43ms/step - loss: 5.7168e-04 - accuracy: 1.0000 - val_loss: 0.0037 - val_accuracy: 1.0000\n",
      "Epoch 100/100\n",
      "3/3 [==============================] - 0s 46ms/step - loss: 8.3124e-04 - accuracy: 1.0000 - val_loss: 0.0024 - val_accuracy: 1.0000\n"
     ]
    }
   ],
   "source": [
    "H = model.fit(x_train, y_train, validation_data=(x_test, y_test), \n",
    "          epochs=max_epochs, batch_size=32)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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00+TnJzM73//e/Lz83n22Weprq7mkUceIScnh0mTJvH6669TWlrKokWLyM3N5fDhw5hMptBxt23bxuOPP057ezv3338/hYWFTJ06tdP5fT4f//mf/4mmabjdbu699178fj+qqnLLLbdw9dVXR+WUFSL8pKCLiPTb3/62Q2v3xhtvDLW009LS+Na3voWiKMyYMYM1a9awbds2zj77bPbu3csDDzxAXFwco0aNoqioiPfee49Jkyaxfv16brzxRoYNGwbAqFGjOpxz7ty5JCUlkZSUxDnnnMOhQ4e6LOhGo5EXXniB9evXU1VVxS233MKjjz7K9773PcaOHTtoPxMhpKCLiHTvvfd224duNps7dIVkZmbidDppaGggOTmZhISE0HtWq5UDBw4AwelKs7Ozuz1nenp66LHJZMLtdne53VNPPcX27dvxeDwYjUY2bNiA2+2moqKC3NxcHn/88b6kKkSvSUEXUcfpdKJpWqio19fXU1hYSEZGBq2trbS3t4eKen19fWjOaYvFwvHjxxkxYsRpnf+uu+5CVVV++MMf8txzz7F161Y+/PBD7rzzztNLTIgeyDh0EXWamppYu3Ytfr+fDz/8kGPHjnHeeedhtVo566yz+Nvf/obX6+Xw4cNs2LCBSy+9FICioiJWrVqF3W5H0zQOHz5MS0tLv2I4duwY2dnZ6HQ6Kisr+zwNqhD9IS10EZF+/etfdxiHPnnyZO69914Axo0bh91uZ/78+aSnp3PPPfeQkpICwMKFC3n++ee57bbbSE5O5tprrw113Vx11VX4fD4effRRWlpaGD58OD/72c/6Fd/BgwcZPXp06PE111xzOukK0SsyH7qIKieHLT7yyCPhDkWIISddLkIIESWkoAshRJSQLhchhIgS0kIXQogoIQVdCCGihBR0IYSIElLQhRAiSkhBF0KIKPH/AaRgPaF1idCRAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "N = np.arange(0, max_epochs)\n",
    "plt.style.use(\"ggplot\")\n",
    "plt.figure()\n",
    "plt.plot(N, H.history[\"loss\"], label=\"train_loss\")\n",
    "plt.plot(N, H.history[\"val_loss\"], label=\"val_loss\")\n",
    "#plt.plot(N, H.history[\"accuracy\"], label=\"train_acc\")\n",
    "#plt.plot(N, H.history[\"val_accuracy\"], label=\"val_acc\")\n",
    "plt.xlabel(\"Epoch #\")\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>7. Evaluate the Model</h3>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "       Sirih       1.00      1.00      1.00         8\n",
      " Jeruk Nipis       1.00      1.00      1.00        12\n",
      "\n",
      "    accuracy                           1.00        20\n",
      "   macro avg       1.00      1.00      1.00        20\n",
      "weighted avg       1.00      1.00      1.00        20\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Asus\\AppData\\Local\\Temp/ipykernel_12184/954476436.py:3: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.\n",
      "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
      "  target = (predictions > 0.5).astype(np.int)\n"
     ]
    }
   ],
   "source": [
    "# menghitung nilai akurasi model terhadap data test\n",
    "predictions = model.predict(x_test, batch_size=32)\n",
    "target = (predictions > 0.5).astype(np.int)\n",
    "print(classification_report(y_test, target, target_names=label_list))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.00049722]]\n"
     ]
    }
   ],
   "source": [
    "# uji model menggunakan image lain\n",
    "queryPath = imagePaths+'001.jpg'\n",
    "query = cv2.imread(queryPath)\n",
    "output = cv2.resize(query, (500, 400) )\n",
    "query = cv2.resize(query, (32, 32))\n",
    "q = []\n",
    "q.append(query)\n",
    "q = np.array(q, dtype='float') / 255.0\n",
    "\n",
    "q_pred = model.predict(q)\n",
    "print(q_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "if q_pred<=0.5 :\n",
    "    target = \"Jeruk Nipis\"\n",
    "else :\n",
    "    target = \"Sirih\"\n",
    "text = \"{}\".format(target)\n",
    "cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)\n",
    " \n",
    "# menampilkan output image\n",
    "cv2.imshow('Output', output)\n",
    "cv2.waitKey() # image tidak akan diclose,sebelum user menekan sembarang tombol\n",
    "cv2.destroyWindow('Output') # image akan diclose"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "interpreter": {
   "hash": "380030d1298d5a27518acca789ff38fe82bbf2e68b73263de6a6bf23efb7704c"
  },
  "kernelspec": {
   "display_name": "Python 3.8.8 64-bit ('base': conda)",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
