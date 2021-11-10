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
    "from tensorflow.keras.layers import Flatten, Dense\n",
    "from tensorflow.keras.optimizers import SGD\n",
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
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "imagePaths = 'dataset\\\\daun\\\\'\n",
    "label_list = ['Sirih', 'Jeruk Nipis']\n",
    "data = []\n",
    "labels = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
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
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(200, 32, 32, 3)"
      ]
     },
     "execution_count": 4,
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
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Sirih' 'Sirih' 'Sirih' 'Sirih' 'Sirih' 'Sirih'\n",
      " 'Sirih' 'Sirih' 'Sirih' 'Sirih' 'Sirih' 'Sirih'\n",
      " 'Sirih' 'Sirih' 'Sirih' 'Sirih' 'Sirih' 'Sirih'\n",
      " 'Sirih' 'Sirih' 'Sirih' 'Sirih' 'Sirih' 'Sirih'\n",
      " 'Sirih' 'Sirih' 'Sirih' 'Sirih' 'Sirih' 'Sirih'\n",
      " 'Sirih' 'Sirih' 'Sirih' 'Sirih' 'Sirih' 'Sirih'\n",
      " 'Sirih' 'Sirih' 'Sirih' 'Sirih' 'Sirih' 'Sirih'\n",
      " 'Sirih' 'Sirih' 'Sirih' 'Sirih' 'Sirih' 'Sirih'\n",
      " 'Sirih' 'Sirih' 'Jeruk Nipis' 'Jeruk Nipis'\n",
      " 'Jeruk Nipis' 'Jeruk Nipis' 'Jeruk Nipis'\n",
      " 'Jeruk Nipis' 'Jeruk Nipis' 'Jeruk Nipis'\n",
      " 'Jeruk Nipis' 'Jeruk Nipis' 'Jeruk Nipis'\n",
      " 'Jeruk Nipis' 'Jeruk Nipis' 'Jeruk Nipis'\n",
      " 'Jeruk Nipis' 'Jeruk Nipis' 'Jeruk Nipis'\n",
      " 'Jeruk Nipis' 'Jeruk Nipis' 'Jeruk Nipis'\n",
      " 'Jeruk Nipis' 'Jeruk Nipis' 'Jeruk Nipis'\n",
      " 'Jeruk Nipis' 'Jeruk Nipis' 'Jeruk Nipis'\n",
      " 'Jeruk Nipis' 'Jeruk Nipis' 'Jeruk Nipis'\n",
      " 'Jeruk Nipis' 'Jeruk Nipis' 'Jeruk Nipis'\n",
      " 'Jeruk Nipis' 'Jeruk Nipis' 'Jeruk Nipis'\n",
      " 'Jeruk Nipis' 'Jeruk Nipis' 'Jeruk Nipis'\n",
      " 'Jeruk Nipis' 'Jeruk Nipis' 'Jeruk Nipis'\n",
      " 'Jeruk Nipis' 'Jeruk Nipis' 'Jeruk Nipis'\n",
      " 'Jeruk Nipis' 'Jeruk Nipis' 'Jeruk Nipis'\n",
      " 'Jeruk Nipis' 'Jeruk Nipis' 'Jeruk Nipis'\n",
     ]
    }
   ],
   "source": [
    "print(labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
      " 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n",
      " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]\n",
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
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
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
    "<h3>5. Build ANN Architecture</h3>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "model.add(Flatten(input_shape=(32,32,3)))\n",
    "model.add(Dense(1024, activation=\"relu\"))\n",
    "model.add(Dense(1024, activation=\"relu\"))\n",
    "model.add(Dense(1, activation=\"sigmoid\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "flatten_1 (Flatten)          (None, 3072)              0         \n",
      "_________________________________________________________________\n",
      "dense_3 (Dense)              (None, 1024)              3146752   \n",
      "_________________________________________________________________\n",
      "dense_4 (Dense)              (None, 1024)              1049600   \n",
      "_________________________________________________________________\n",
      "dense_5 (Dense)              (None, 1)                 1025      \n",
      "=================================================================\n",
      "Total params: 4,197,377\n",
      "Trainable params: 4,197,377\n",
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
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# tentukan hyperparameter\n",
    "lr = 0.01\n",
    "max_epochs = 100\n",
    "opt_funct = SGD(learning_rate=lr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
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
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100\n",
      "5/5 [==============================] - 1s 108ms/step - loss: 0.8005 - accuracy: 0.4462 - val_loss: 0.6603 - val_accuracy: 0.6750\n",
      "Epoch 2/100\n",
      "5/5 [==============================] - 0s 58ms/step - loss: 0.6565 - accuracy: 0.7293 - val_loss: 0.6114 - val_accuracy: 0.7750\n",
      "Epoch 3/100\n",
      "5/5 [==============================] - 0s 61ms/step - loss: 0.6046 - accuracy: 0.7095 - val_loss: 0.5829 - val_accuracy: 0.7750\n",
      "Epoch 4/100\n",
      "5/5 [==============================] - 0s 56ms/step - loss: 0.5781 - accuracy: 0.7148 - val_loss: 0.5709 - val_accuracy: 0.7750\n",
      "Epoch 5/100\n",
      "5/5 [==============================] - 0s 70ms/step - loss: 0.5571 - accuracy: 0.7809 - val_loss: 0.5882 - val_accuracy: 0.5500\n",
      "Epoch 6/100\n",
      "5/5 [==============================] - 0s 57ms/step - loss: 0.6185 - accuracy: 0.6077 - val_loss: 0.5712 - val_accuracy: 0.7250\n",
      "Epoch 7/100\n",
      "5/5 [==============================] - 0s 58ms/step - loss: 0.5120 - accuracy: 0.8251 - val_loss: 0.5709 - val_accuracy: 0.7250\n",
      "Epoch 8/100\n",
      "5/5 [==============================] - 0s 54ms/step - loss: 0.4939 - accuracy: 0.8462 - val_loss: 0.5196 - val_accuracy: 0.8000\n",
      "Epoch 9/100\n",
      "5/5 [==============================] - 0s 53ms/step - loss: 0.4818 - accuracy: 0.8235 - val_loss: 0.5082 - val_accuracy: 0.7250\n",
      "Epoch 10/100\n",
      "5/5 [==============================] - 0s 53ms/step - loss: 0.4584 - accuracy: 0.8216 - val_loss: 0.5390 - val_accuracy: 0.7250\n",
      "Epoch 11/100\n",
      "5/5 [==============================] - 0s 55ms/step - loss: 0.4944 - accuracy: 0.8282 - val_loss: 0.5938 - val_accuracy: 0.6750\n",
      "Epoch 12/100\n",
      "5/5 [==============================] - 0s 67ms/step - loss: 0.4642 - accuracy: 0.8188 - val_loss: 0.4983 - val_accuracy: 0.8250\n",
      "Epoch 13/100\n",
      "5/5 [==============================] - 0s 55ms/step - loss: 0.5671 - accuracy: 0.7432 - val_loss: 0.5057 - val_accuracy: 0.7500\n",
      "Epoch 14/100\n",
      "5/5 [==============================] - 0s 52ms/step - loss: 0.4368 - accuracy: 0.8662 - val_loss: 0.5204 - val_accuracy: 0.7750\n",
      "Epoch 15/100\n",
      "5/5 [==============================] - 0s 62ms/step - loss: 0.4317 - accuracy: 0.8503 - val_loss: 0.4810 - val_accuracy: 0.8250\n",
      "Epoch 16/100\n",
      "5/5 [==============================] - 0s 51ms/step - loss: 0.4635 - accuracy: 0.7636 - val_loss: 0.4647 - val_accuracy: 0.7750\n",
      "Epoch 17/100\n",
      "5/5 [==============================] - 0s 55ms/step - loss: 0.4661 - accuracy: 0.8260 - val_loss: 0.6674 - val_accuracy: 0.5750\n",
      "Epoch 18/100\n",
      "5/5 [==============================] - 0s 54ms/step - loss: 0.5277 - accuracy: 0.7720 - val_loss: 0.4784 - val_accuracy: 0.8000\n",
      "Epoch 19/100\n",
      "5/5 [==============================] - 0s 54ms/step - loss: 0.3805 - accuracy: 0.8886 - val_loss: 0.4592 - val_accuracy: 0.8500\n",
      "Epoch 20/100\n",
      "5/5 [==============================] - 0s 57ms/step - loss: 0.3506 - accuracy: 0.9219 - val_loss: 0.4561 - val_accuracy: 0.8000\n",
      "Epoch 21/100\n",
      "5/5 [==============================] - 0s 54ms/step - loss: 0.4402 - accuracy: 0.7830 - val_loss: 0.4485 - val_accuracy: 0.8500\n",
      "Epoch 22/100\n",
      "5/5 [==============================] - 0s 52ms/step - loss: 0.3912 - accuracy: 0.8571 - val_loss: 0.4742 - val_accuracy: 0.8250\n",
      "Epoch 23/100\n",
      "5/5 [==============================] - 0s 55ms/step - loss: 0.5822 - accuracy: 0.6628 - val_loss: 0.4413 - val_accuracy: 0.8000\n",
      "Epoch 24/100\n",
      "5/5 [==============================] - 0s 56ms/step - loss: 0.3480 - accuracy: 0.8984 - val_loss: 0.4588 - val_accuracy: 0.7250\n",
      "Epoch 25/100\n",
      "5/5 [==============================] - 0s 56ms/step - loss: 0.3398 - accuracy: 0.8937 - val_loss: 0.4934 - val_accuracy: 0.7250\n",
      "Epoch 26/100\n",
      "5/5 [==============================] - 0s 57ms/step - loss: 0.3303 - accuracy: 0.9049 - val_loss: 0.4459 - val_accuracy: 0.7750\n",
      "Epoch 27/100\n",
      "5/5 [==============================] - 0s 55ms/step - loss: 0.3569 - accuracy: 0.8784 - val_loss: 0.4561 - val_accuracy: 0.7500\n",
      "Epoch 28/100\n",
      "5/5 [==============================] - 0s 57ms/step - loss: 0.3120 - accuracy: 0.9139 - val_loss: 0.4174 - val_accuracy: 0.8000\n",
      "Epoch 29/100\n",
      "5/5 [==============================] - 0s 55ms/step - loss: 0.3281 - accuracy: 0.9190 - val_loss: 0.4178 - val_accuracy: 0.7750\n",
      "Epoch 30/100\n",
      "5/5 [==============================] - 0s 65ms/step - loss: 0.4134 - accuracy: 0.8119 - val_loss: 0.5184 - val_accuracy: 0.7250\n",
      "Epoch 31/100\n",
      "5/5 [==============================] - 0s 54ms/step - loss: 0.2901 - accuracy: 0.9233 - val_loss: 0.4078 - val_accuracy: 0.8250\n",
      "Epoch 32/100\n",
      "5/5 [==============================] - 0s 62ms/step - loss: 0.3304 - accuracy: 0.8885 - val_loss: 0.4058 - val_accuracy: 0.8000\n",
      "Epoch 33/100\n",
      "5/5 [==============================] - 0s 61ms/step - loss: 0.2783 - accuracy: 0.9243 - val_loss: 0.5113 - val_accuracy: 0.7250\n",
      "Epoch 34/100\n",
      "5/5 [==============================] - 0s 54ms/step - loss: 0.4657 - accuracy: 0.7517 - val_loss: 0.4150 - val_accuracy: 0.8000\n",
      "Epoch 35/100\n",
      "5/5 [==============================] - 0s 53ms/step - loss: 0.3001 - accuracy: 0.9264 - val_loss: 0.3985 - val_accuracy: 0.8250\n",
      "Epoch 36/100\n",
      "5/5 [==============================] - 0s 54ms/step - loss: 0.2536 - accuracy: 0.9470 - val_loss: 0.3964 - val_accuracy: 0.8000\n",
      "Epoch 37/100\n",
      "5/5 [==============================] - 0s 65ms/step - loss: 0.2998 - accuracy: 0.8828 - val_loss: 0.3929 - val_accuracy: 0.8250\n",
      "Epoch 38/100\n",
      "5/5 [==============================] - 0s 52ms/step - loss: 0.2416 - accuracy: 0.9517 - val_loss: 0.4066 - val_accuracy: 0.7500\n",
      "Epoch 39/100\n",
      "5/5 [==============================] - 0s 54ms/step - loss: 0.2636 - accuracy: 0.9289 - val_loss: 0.5148 - val_accuracy: 0.8000\n",
      "Epoch 40/100\n",
      "5/5 [==============================] - 0s 52ms/step - loss: 0.4228 - accuracy: 0.7707 - val_loss: 0.4469 - val_accuracy: 0.8250\n",
      "Epoch 41/100\n",
      "5/5 [==============================] - 0s 51ms/step - loss: 0.2575 - accuracy: 0.9135 - val_loss: 0.5229 - val_accuracy: 0.7000\n",
      "Epoch 42/100\n",
      "5/5 [==============================] - 0s 53ms/step - loss: 0.2886 - accuracy: 0.8911 - val_loss: 0.4168 - val_accuracy: 0.7750\n",
      "Epoch 43/100\n",
      "5/5 [==============================] - 0s 53ms/step - loss: 0.3182 - accuracy: 0.8497 - val_loss: 0.3800 - val_accuracy: 0.8250\n",
      "Epoch 44/100\n",
      "5/5 [==============================] - 0s 52ms/step - loss: 0.2281 - accuracy: 0.9685 - val_loss: 0.3741 - val_accuracy: 0.8250\n",
      "Epoch 45/100\n",
      "5/5 [==============================] - 0s 52ms/step - loss: 0.2118 - accuracy: 0.9728 - val_loss: 0.3921 - val_accuracy: 0.7750\n",
      "Epoch 46/100\n",
      "5/5 [==============================] - 0s 55ms/step - loss: 0.2344 - accuracy: 0.9425 - val_loss: 0.3744 - val_accuracy: 0.8250\n",
      "Epoch 47/100\n",
      "5/5 [==============================] - 0s 55ms/step - loss: 0.2956 - accuracy: 0.8766 - val_loss: 0.4200 - val_accuracy: 0.7750\n",
      "Epoch 48/100\n",
      "5/5 [==============================] - 0s 53ms/step - loss: 0.2274 - accuracy: 0.9446 - val_loss: 0.4510 - val_accuracy: 0.7750\n",
      "Epoch 49/100\n",
      "5/5 [==============================] - 0s 53ms/step - loss: 0.3004 - accuracy: 0.8735 - val_loss: 0.4545 - val_accuracy: 0.8250\n",
      "Epoch 50/100\n",
      "5/5 [==============================] - 0s 54ms/step - loss: 0.3218 - accuracy: 0.8524 - val_loss: 0.3591 - val_accuracy: 0.8500\n",
      "Epoch 51/100\n",
      "5/5 [==============================] - 0s 55ms/step - loss: 0.2057 - accuracy: 0.9620 - val_loss: 0.3728 - val_accuracy: 0.8250\n",
      "Epoch 52/100\n",
      "5/5 [==============================] - 0s 63ms/step - loss: 0.3139 - accuracy: 0.8314 - val_loss: 0.4309 - val_accuracy: 0.7750\n",
      "Epoch 53/100\n",
      "5/5 [==============================] - 0s 52ms/step - loss: 0.2187 - accuracy: 0.9503 - val_loss: 0.3585 - val_accuracy: 0.8250\n",
      "Epoch 54/100\n",
      "5/5 [==============================] - 0s 48ms/step - loss: 0.2407 - accuracy: 0.9135 - val_loss: 0.3661 - val_accuracy: 0.8250\n",
      "Epoch 55/100\n",
      "5/5 [==============================] - 0s 48ms/step - loss: 0.1857 - accuracy: 0.9569 - val_loss: 0.3480 - val_accuracy: 0.8500\n",
      "Epoch 56/100\n",
      "5/5 [==============================] - 0s 50ms/step - loss: 0.1778 - accuracy: 0.9671 - val_loss: 0.4048 - val_accuracy: 0.7750\n",
      "Epoch 57/100\n",
      "5/5 [==============================] - 0s 49ms/step - loss: 0.1870 - accuracy: 0.9455 - val_loss: 0.3601 - val_accuracy: 0.8500\n",
      "Epoch 58/100\n",
      "5/5 [==============================] - 0s 48ms/step - loss: 0.1913 - accuracy: 0.9357 - val_loss: 0.3833 - val_accuracy: 0.7750\n",
      "Epoch 59/100\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5/5 [==============================] - 0s 52ms/step - loss: 0.1993 - accuracy: 0.9579 - val_loss: 0.3528 - val_accuracy: 0.8250\n",
      "Epoch 60/100\n",
      "5/5 [==============================] - 0s 50ms/step - loss: 0.1816 - accuracy: 0.9589 - val_loss: 0.3553 - val_accuracy: 0.8500\n",
      "Epoch 61/100\n",
      "5/5 [==============================] - 0s 49ms/step - loss: 0.1734 - accuracy: 0.9404 - val_loss: 0.3350 - val_accuracy: 0.8250\n",
      "Epoch 62/100\n",
      "5/5 [==============================] - 0s 50ms/step - loss: 0.1625 - accuracy: 0.9655 - val_loss: 0.3400 - val_accuracy: 0.8750\n",
      "Epoch 63/100\n",
      "5/5 [==============================] - 0s 49ms/step - loss: 0.1874 - accuracy: 0.9291 - val_loss: 0.4120 - val_accuracy: 0.8250\n",
      "Epoch 64/100\n",
      "5/5 [==============================] - 0s 51ms/step - loss: 0.2791 - accuracy: 0.8347 - val_loss: 0.3781 - val_accuracy: 0.7750\n",
      "Epoch 65/100\n",
      "5/5 [==============================] - 0s 51ms/step - loss: 0.2499 - accuracy: 0.8979 - val_loss: 0.7387 - val_accuracy: 0.6750\n",
      "Epoch 66/100\n",
      "5/5 [==============================] - 0s 52ms/step - loss: 0.7732 - accuracy: 0.6115 - val_loss: 0.5938 - val_accuracy: 0.7250\n",
      "Epoch 67/100\n",
      "5/5 [==============================] - 0s 51ms/step - loss: 0.2445 - accuracy: 0.8940 - val_loss: 0.3384 - val_accuracy: 0.8500\n",
      "Epoch 68/100\n",
      "5/5 [==============================] - 0s 51ms/step - loss: 0.1699 - accuracy: 0.9610 - val_loss: 0.3397 - val_accuracy: 0.8250\n",
      "Epoch 69/100\n",
      "5/5 [==============================] - 0s 49ms/step - loss: 0.1472 - accuracy: 0.9691 - val_loss: 0.3515 - val_accuracy: 0.8250\n",
      "Epoch 70/100\n",
      "5/5 [==============================] - 0s 57ms/step - loss: 0.1417 - accuracy: 0.9754 - val_loss: 0.3351 - val_accuracy: 0.8250\n",
      "Epoch 71/100\n",
      "5/5 [==============================] - 0s 63ms/step - loss: 0.1274 - accuracy: 0.9688 - val_loss: 0.3259 - val_accuracy: 0.8750\n",
      "Epoch 72/100\n",
      "5/5 [==============================] - 0s 50ms/step - loss: 0.1819 - accuracy: 0.9424 - val_loss: 0.5597 - val_accuracy: 0.8000\n",
      "Epoch 73/100\n",
      "5/5 [==============================] - 0s 50ms/step - loss: 0.2094 - accuracy: 0.9222 - val_loss: 0.3448 - val_accuracy: 0.8750\n",
      "Epoch 74/100\n",
      "5/5 [==============================] - 0s 61ms/step - loss: 0.1348 - accuracy: 0.9588 - val_loss: 0.3876 - val_accuracy: 0.7750\n",
      "Epoch 75/100\n",
      "5/5 [==============================] - 0s 50ms/step - loss: 0.1601 - accuracy: 0.9645 - val_loss: 0.5978 - val_accuracy: 0.7500\n",
      "Epoch 76/100\n",
      "5/5 [==============================] - 0s 50ms/step - loss: 0.2449 - accuracy: 0.9080 - val_loss: 0.3194 - val_accuracy: 0.8750\n",
      "Epoch 77/100\n",
      "5/5 [==============================] - 0s 63ms/step - loss: 0.1308 - accuracy: 0.9742 - val_loss: 0.3635 - val_accuracy: 0.8000\n",
      "Epoch 78/100\n",
      "5/5 [==============================] - 0s 52ms/step - loss: 0.1000 - accuracy: 0.9894 - val_loss: 0.3976 - val_accuracy: 0.8000\n",
      "Epoch 79/100\n",
      "5/5 [==============================] - 0s 55ms/step - loss: 0.1510 - accuracy: 0.9613 - val_loss: 0.3382 - val_accuracy: 0.8250\n",
      "Epoch 80/100\n",
      "5/5 [==============================] - 0s 51ms/step - loss: 0.1089 - accuracy: 0.9742 - val_loss: 0.3815 - val_accuracy: 0.7750\n",
      "Epoch 81/100\n",
      "5/5 [==============================] - 0s 69ms/step - loss: 0.1443 - accuracy: 0.9641 - val_loss: 0.3100 - val_accuracy: 0.8500\n",
      "Epoch 82/100\n",
      "5/5 [==============================] - 0s 66ms/step - loss: 0.1163 - accuracy: 1.0000 - val_loss: 0.3226 - val_accuracy: 0.8250\n",
      "Epoch 83/100\n",
      "5/5 [==============================] - 0s 53ms/step - loss: 0.1109 - accuracy: 0.9689 - val_loss: 0.3515 - val_accuracy: 0.8750\n",
      "Epoch 84/100\n",
      "5/5 [==============================] - 0s 60ms/step - loss: 0.1372 - accuracy: 0.9630 - val_loss: 0.3541 - val_accuracy: 0.8000\n",
      "Epoch 85/100\n",
      "5/5 [==============================] - 0s 56ms/step - loss: 0.1087 - accuracy: 0.9843 - val_loss: 0.3037 - val_accuracy: 0.8750\n",
      "Epoch 86/100\n",
      "5/5 [==============================] - 0s 49ms/step - loss: 0.1170 - accuracy: 0.9793 - val_loss: 0.3054 - val_accuracy: 0.9000\n",
      "Epoch 87/100\n",
      "5/5 [==============================] - 0s 48ms/step - loss: 0.1230 - accuracy: 0.9610 - val_loss: 0.4628 - val_accuracy: 0.7750\n",
      "Epoch 88/100\n",
      "5/5 [==============================] - 0s 47ms/step - loss: 0.1985 - accuracy: 0.9279 - val_loss: 0.5949 - val_accuracy: 0.7750\n",
      "Epoch 89/100\n",
      "5/5 [==============================] - 0s 48ms/step - loss: 0.2251 - accuracy: 0.8583 - val_loss: 0.3032 - val_accuracy: 0.8500\n",
      "Epoch 90/100\n",
      "5/5 [==============================] - 0s 51ms/step - loss: 0.0894 - accuracy: 0.9928 - val_loss: 0.3346 - val_accuracy: 0.8250\n",
      "Epoch 91/100\n",
      "5/5 [==============================] - 0s 50ms/step - loss: 0.0993 - accuracy: 0.9932 - val_loss: 0.3207 - val_accuracy: 0.8250\n",
      "Epoch 92/100\n",
      "5/5 [==============================] - 0s 48ms/step - loss: 0.0837 - accuracy: 0.9966 - val_loss: 0.3335 - val_accuracy: 0.8250\n",
      "Epoch 93/100\n",
      "5/5 [==============================] - 0s 47ms/step - loss: 0.0837 - accuracy: 0.9845 - val_loss: 0.3014 - val_accuracy: 0.8750\n",
      "Epoch 94/100\n",
      "5/5 [==============================] - 0s 48ms/step - loss: 0.0858 - accuracy: 0.9923 - val_loss: 0.3094 - val_accuracy: 0.8500\n",
      "Epoch 95/100\n",
      "5/5 [==============================] - 0s 48ms/step - loss: 0.0861 - accuracy: 1.0000 - val_loss: 0.4095 - val_accuracy: 0.8000\n",
      "Epoch 96/100\n",
      "5/5 [==============================] - 0s 52ms/step - loss: 0.0792 - accuracy: 1.0000 - val_loss: 0.3258 - val_accuracy: 0.8250\n",
      "Epoch 97/100\n",
      "5/5 [==============================] - 0s 59ms/step - loss: 0.0914 - accuracy: 0.9898 - val_loss: 0.3275 - val_accuracy: 0.8250\n",
      "Epoch 98/100\n",
      "5/5 [==============================] - 0s 63ms/step - loss: 0.0725 - accuracy: 0.9979 - val_loss: 0.2973 - val_accuracy: 0.8750\n",
      "Epoch 99/100\n",
      "5/5 [==============================] - 0s 55ms/step - loss: 0.0777 - accuracy: 1.0000 - val_loss: 0.3468 - val_accuracy: 0.8250\n",
      "Epoch 100/100\n",
      "5/5 [==============================] - 0s 70ms/step - loss: 0.0826 - accuracy: 0.9949 - val_loss: 0.3122 - val_accuracy: 0.8250\n"
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
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXQAAAEJCAYAAACE39xMAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAB3BklEQVR4nO2dd3yb1b3/30db8rbkPTKckIQQCMYQCDMkpGxoS1va0lvKuKXjsn7lFnoptGXethRKL3RSaOleUEZp00BYSSAhIawkJM5wHG9L3tp6zu+PR5IlW952vM779cortvTo0Tmy9Hm++pzv+X6FlFKiUCgUimmPYbIHoFAoFIrxQQm6QqFQzBCUoCsUCsUMQQm6QqFQzBCUoCsUCsUMQQm6QqFQzBBMk/nk9fX1o3qcy+WitbV1nEcz9ZmN856Nc4bZOe/ZOGcY+byLi4sHvE9F6AqFQjFDUIKuUCgUMwQl6AqFQjFDmFQPXaFQzCyklPj9fjRNQwgxosc2NTURCAQmaGRTl1TzllJiMBiw2Wwjeh2VoCsUinHD7/djNpsxmUYuLSaTCaPROAGjmtoMNO9wOIzf78dutw/7XMpyUSgU44amaaMSc0V/TCYTmqaN6DFK0BUKxbgxUptFMTgjfT2nnaDLuhq6f/tTZHfnZA9FoVAophTTTtBpqqfnL78CT8tkj0ShUCimFNNP0NMz9f9VhK5QKPrQ0dHBE088MeLHfe5zn6Ojo2PEj7vxxht57rnnRvy4iWJYqxc7duzg8ccfR9M0Vq9ezaWXXpp0v9fr5eGHH8btdhOJRLjoootYtWrVRIwXMnRBl91dKLdOoVAk0tnZya9//WuuvPLKpNsjkcigGTRPPvnkBI/syDCkoGuaxmOPPcbtt9+O0+nktttuo6qqitLS0vgx//znPyktLeXWW2+ls7OTG264gdNPP31iVrtVhK5QTAu0P/wcWXtg+McLwVAdMUXZPAyXXzvg/ffeey81NTWcc845mM1mHA4HBQUFfPDBB7z88stcddVV1NfXEwgEuPrqq7niiisAWLFiBS+88AI9PT1cccUVnHTSSbz11lsUFhbyy1/+clipg6+99hp33XUXkUiE4447jvvuuw+r1cq9997LunXrMJlMnHHGGdxxxx08++yzPPjggxgMBrKysvjrX/867NdpMIZU3OrqagoLCykoKABg5cqVbN26NUnQhRD4/f74poL09HQMhglycxzpIIQSdIVC0Y9vfOMbfPjhh/z73/9m06ZN/Md//AcvvfQS5eXlADzwwAPk5OTg8/m44IILOP/888nNzU06x4EDB3jkkUf43ve+xxe/+EX+8Y9/8PGPf3zQ5/X7/dx000388Y9/pKKiguuvv55f//rXXHbZZbzwwgu8+uqrCCHits5DDz3Eb3/7W4qKiujp6Rm3+Q8p6B6PB6fTGf/d6XSyd+/epGPOPfdcvvvd7/LFL34Rn8/HTTfdlFLQ169fz/r16wG4//77cblcoxp0c3oGtnCIzFE+frpiMplG/ZpNV2bjnGH6zrupqan3m/kVXzrizx+zVWKbdY4//njmz58fv/+JJ57gH//4B6BXez106BD5+fkIITAajRiNRsrLy1m+fDkAy5cvp66ubkC3wWAwYDQaqampYc6cOSxatAiAyy+/nMcff5xrr70Wm83GLbfcwjnnnMM555yDyWTipJNO4uabb+biiy/mggsuGPD8Vqt1RO+DIQU91VegvrmR77zzDnPmzOGOO+6gqamJu+66i8WLF+NwOJKOW7NmDWvWrIn/PtpSmYaMbPytzQRnWanN2VhedDbOGabvvAOBwKh3e5pMJsLh8JiePxKJAPouy0gkgt1uj59z06ZNvPLKKzzzzDPY7XYuu+wyvF4v4XAYKSWRSIRIJILFYok/RghBKBQacFyaphGJRAiFQkgp48dFIpG4dj733HO8/vrr/P3vf+cXv/gFf/7zn7nvvvvYvn07L774IqtXr+Zf//pXv28KoL+efd8HYyqf63Q6cbvd8d/dbjc5OTlJx2zYsIEVK1YghKCwsJD8/PxR1zofDoaMTJWHrlAo+pGWlkZ3d3fK+7q6usjKysJut1NdXc327dvH7XkXLFhAbW0tBw7oawZ//etfOfnkk+np6aGrq4vVq1fz7W9/m507dwJw8OBBKisrueWWW8jNzR03vRwyQq+oqKChoYHm5mZyc3PZtGkT119/fdIxLpeL9957jyVLltDe3k59fT35+fnjMsBUiMxsaKibsPMrFIrpSW5uLieeeCJnn302Npstya4466yzePLJJ1mzZg3z58+nsrJy3J7XZrPxgx/8gC9+8YvxRdHPfe5ztLe3c9VVVxEIBJBScueddwJw9913c+DAAaSUnH766SxdunRcxiHkUMvK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\n",
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
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "       Sirih       1.00      0.75      0.86        12\n",
      "Jeruk Nipis        0.73      1.00      0.84         8\n",
      "\n",
      "    accuracy                           0.85        20\n",
      "   macro avg       0.86      0.88      0.85        20\n",
      "weighted avg       0.89      0.85      0.85        20\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Asus\\AppData\\Local\\Temp/ipykernel_9720/954476436.py:3: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.\n",
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
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.12693542]]\n"
     ]
    }
   ],
   "source": [
    "# uji model menggunakan image lain\n",
    "queryPath = imagePaths+'Sirih_001.jpg'\n",
    "query = cv2.imread(queryPath)\n",
    "output = query.copy(query,(500,400))\n",
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
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "if q_pred<=0.5 :\n",
    "    target = \"Sirih\"\n",
    "else :\n",
    "    target = \"Jeruk Nipis\"\n",
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
  "kernelspec": {
   "display_name": "Python 3.9.6 64-bit",
   "language": "python",
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
