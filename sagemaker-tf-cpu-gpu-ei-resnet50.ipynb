{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Amazon SageMaker inference deployment to CPUs, GPUs, and EI\n",
    "This example demonstrates Amazon SageMaker inference deployment using SageMaker SDK\n",
    "\n",
    "This example was tested on Amazon SageMaker Studio Notebook\n",
    "Run this notebook using the following Amazon SageMaker Studio conda environment:\n",
    "`TensorFlow 2 CPU Optimized`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# !pip install --upgrade pip -q\n",
    "# !pip install --upgrade sagemaker -q"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sagemaker version: 2.15.1\n",
      "tensorflow version: 2.1.0\n"
     ]
    }
   ],
   "source": [
    "import tarfile\n",
    "import sagemaker\n",
    "import tensorflow as tf\n",
    "import tensorflow.keras as keras\n",
    "import shutil\n",
    "import os\n",
    "import time\n",
    "from tensorflow.keras.applications.resnet50 import ResNet50\n",
    "\n",
    "role = sagemaker.get_execution_role()\n",
    "sess = sagemaker.Session()\n",
    "region = sess.boto_region_name\n",
    "bucket = sess.default_bucket()\n",
    "print('sagemaker version: '+sagemaker.__version__)\n",
    "print('tensorflow version: '+tf.__version__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_save_resnet50_model(model_path):\n",
    "    model = ResNet50(weights='imagenet')\n",
    "    shutil.rmtree(model_path, ignore_errors=True)\n",
    "    model.save(model_path, include_optimizer=False, save_format='tf')\n",
    "\n",
    "saved_model_dir = 'resnet50_saved_model' \n",
    "model_ver = '1'\n",
    "model_path = os.path.join(saved_model_dir, model_ver)\n",
    "\n",
    "# load_save_resnet50_model(model_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "./\n",
      "./1/\n",
      "./1/variables/\n",
      "./1/variables/variables.data-00000-of-00001\n",
      "./1/variables/variables.index\n",
      "./1/saved_model.pb\n",
      "./1/assets/\n"
     ]
    }
   ],
   "source": [
    "shutil.rmtree('model.tar.gz', ignore_errors=True)\n",
    "!tar cvfz model.tar.gz -C resnet50_saved_model ."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sagemaker.tensorflow.model import TensorFlowModel, TensorFlowPredictor\n",
    "\n",
    "prefix = 'keras_models'\n",
    "s3_model_path = sess.upload_data(path='model.tar.gz', key_prefix=prefix)\n",
    "\n",
    "model = TensorFlowModel(model_data=s3_model_path, \n",
    "                        framework_version='1.15',\n",
    "                        role=role,\n",
    "                        predictor_cls = TensorFlowPredictor,\n",
    "                        sagemaker_session=sess)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Deploy to CPU instance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "update_endpoint is a no-op in sagemaker>=2.\n",
      "See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-------------!"
     ]
    }
   ],
   "source": [
    "predictor_cpu = model.deploy(initial_instance_count=1, \n",
    "                             instance_type='ml.c5.xlarge')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Deploy using EI"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "update_endpoint is a no-op in sagemaker>=2.\n",
      "See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-------------!"
     ]
    }
   ],
   "source": [
    "predictor_ei = model.deploy(initial_instance_count=1, \n",
    "                            instance_type='ml.c5.xlarge',\n",
    "                            accelerator_type='ml.eia2.large')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Deploy to GPU instance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "update_endpoint is a no-op in sagemaker>=2.\n",
      "See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-------------!"
     ]
    }
   ],
   "source": [
    "predictor_gpu = model.deploy(initial_instance_count=1, \n",
    "                         instance_type='ml.g4dn.xlarge')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Test endpoint"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "## If you have an existing endpoint, create a predictor using the endpoint name\n",
    "\n",
    "# from sagemaker.tensorflow.model import TensorFlowPredictor\n",
    "# predictor = TensorFlowPredictor('ENDPOINT_NAME',\n",
    "#                                sagemaker_session=sess)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def image_preprocess(img, reps=1):\n",
    "    img = np.asarray(img.resize((224, 224)))\n",
    "    img = np.stack([img]*reps)\n",
    "    img = tf.keras.applications.resnet50.preprocess_input(img)\n",
    "    return img"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "from PIL import Image \n",
    "import numpy as np\n",
    "import json\n",
    "\n",
    "img= Image.open('kitten.jpg')\n",
    "img = image_preprocess(img, 5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Invoke CPU endpoint"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[[('n02123159', 'tiger_cat', 0.495739877),\n",
       "  ('n02123045', 'tabby', 0.434538245),\n",
       "  ('n02124075', 'Egyptian_cat', 0.0492461845),\n",
       "  ('n02127052', 'lynx', 0.0143557377),\n",
       "  ('n02128385', 'leopard', 0.00133766234)]]"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "response = predictor_cpu.predict(data=img)\n",
    "probs = np.array(response['predictions'][0])\n",
    "tf.keras.applications.resnet.decode_predictions(np.expand_dims(probs, axis=0), top=5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Invoke CPU Instance + EI endpoint"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[[('n02123159', 'tiger_cat', 0.495739),\n",
       "  ('n02123045', 'tabby', 0.434539199),\n",
       "  ('n02124075', 'Egyptian_cat', 0.0492460541),\n",
       "  ('n02127052', 'lynx', 0.0143557545),\n",
       "  ('n02128385', 'leopard', 0.00133766781)]]"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "response = predictor_ei.predict(data=img)\n",
    "probs = np.array(response['predictions'][0])\n",
    "tf.keras.applications.resnet.decode_predictions(np.expand_dims(probs, axis=0), top=5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Invoke G4 GPU Instance with NVIDIA T4 endpoint"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[[('n02123159', 'tiger_cat', 0.495739311),\n",
       "  ('n02123045', 'tabby', 0.434538603),\n",
       "  ('n02124075', 'Egyptian_cat', 0.0492461771),\n",
       "  ('n02127052', 'lynx', 0.0143557768),\n",
       "  ('n02128385', 'leopard', 0.00133766851)]]"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "response = predictor_gpu.predict(data=img)\n",
    "probs = np.array(response['predictions'][0])\n",
    "tf.keras.applications.resnet.decode_predictions(np.expand_dims(probs, axis=0), top=5)"
   ]
  }
 ],
 "metadata": {
  "instance_type": "ml.t3.medium",
  "kernelspec": {
   "display_name": "Python 3 (TensorFlow 2 CPU Optimized)",
   "language": "python",
   "name": "python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-west-2:236514542706:image/tensorflow-2.1-cpu-py36"
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
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
