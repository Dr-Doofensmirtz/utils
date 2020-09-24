
import numpy as np
import matplotlib.pyplot as pyplot

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam

device = tf.device("/GPU:0")
def select_gpu():
    if tf.device("/GPU:0"):
        device = tf.device("/GPU:0")
    
def gan(gen, dis):
  with device:
    dis.trainable = False
    
    model = Sequential()
    model.add(gen)
    model.add(dis)

    model.compile(loss= 'binary_crossentropy', optimizer='adam')
  return model

def load_real_data():
  (X,_), (_,_) = tf.keras.datasets.mnist.load_data()
  X = np.expand_dims(X, axis=-1)
  X = X.astype('float32')
  X = X/255.0

  return X  

def generate_real_sample(n_samples):
  dataset = load_real_data()
  ix = np.random.randint(0, dataset.shape[0], n_samples)
  X = dataset[ix]
  y = np.ones((n_samples, 1))
  return X,y

def get_latent_points(latent_dim, n_samples):
  x = np.random.randn(latent_dim*n_samples)
  x = x.reshape(n_samples, latent_dim)
  return x

def generate_fake_sample(g_model, latent_dim, n_samples):
  x = get_latent_points(latent_dim, n_samples)
  output = g_model.predict(x)
  y = np.zeros((n_samples, 1))
  return output, y

def save_plot(examples, epoch, n=10):
  for i in range(n * n):
  # define subplot
    pyplot.subplot(n, n, 1 + i)
    # turn off axis
    pyplot.axis('off')
    # plot raw pixel data
    pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
  # save plot to file
  filename = 'generated_plot_e%03d.png' % (epoch+1)
  pyplot.savefig(filename)
  pyplot.close()

def summarize_performance(epoch, g_mod, d_mod, dataset, latent_dim, n_samples=100):
  x_real, y_real = generate_real_sample(n_samples)
  x_fake, y_fake = generate_fake_sample(g_mod, latent_dim, n_samples)

  _, real_acc = d_mod.evaluate(x_real, y_real)
  _, fake_acc = d_mod.evaluate(x_fake, y_fake)

  print(f"Accuracy: real: {real_acc*100}, fake: {fake_acc*100}")
  save_plot(x_fake, epoch)
  save_plot(x_fake, epoch)
  filename = 'generator_model_%03d.h5' % (epoch + 1)
  g_mod.save(filename)
        
def train(g_model, d_model, gan, dataset, latent_dim, epochs=100, n_batch=256):
  with device:    
    batch_per_epoch = int(dataset.shape[0]/ n_batch)
    half_batch = int(n_batch/2)

    for epoch in range(epochs):
      for batch in range(batch_per_epoch):
        x_real, y_real = generate_real_sample(half_batch)
        x_fake, y_fake = generate_fake_sample(g_model, latent_dim, n_samples=half_batch)

        x, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
        d_loss ,_ = d_model.train_on_batch(x,y)

        x_gan = get_latent_points(100, n_batch)
        y_gan = np.ones((n_batch,1))
        gan_loss= gan.train_on_batch(x_gan,y_gan)

        print('>%d, %d/%d, d=%.3f, g=%.3f' % (epoch+1, batch+1, batch_per_epoch, d_loss, gan_loss))

      if (epoch+1)%10 == 0:
        summarize_performance(epoch, g_model, d_model, dataset, latent_dim)
        

