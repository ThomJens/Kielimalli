import os
import shutil
import sys
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt
#import optimization  # to create AdamW optimizer

# Source
# https://www.tensorflow.org/text/tutorials/classify_text_with_bert



tf.get_logger().setLevel('ERROR')

train_dir = './nnData/train/'
test_dir = './nnData/test/'

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 128
seed = 42


raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
#    label_mode = 'categorical',
    subset='training',
    validation_split = 0.2,
    seed = seed)


class_names = raw_train_ds.class_names

train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
#    label_mode = 'categorical',
    subset='validation',
    validation_split = 0.2,
    seed = seed)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


test_ds = tf.keras.utils.text_dataset_from_directory(
    test_dir,
#    label_mode = 'categorical',
    batch_size=batch_size)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    
        
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
#tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4'

bert_model = hub.KerasLayer(tfhub_handle_encoder)
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)



def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(17, activation='softmax', name='classifier')(net)
    return tf.keras.Model(text_input, net)


classifier_model = build_classifier_model()


loss = tf.keras.losses.SparseCategoricalCrossentropy()#from_logits=True)
metrics = tf.metrics.CategoricalAccuracy()

epochs = 200
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)


#init_lr = 3e-5
#optimizer = optimization.create_optimizer(init_lr=init_lr,
#                                          num_train_steps=num_train_steps,
#                                          num_warmup_steps=num_warmup_steps,
#                                          optimizer_type='adamw')

optimizer =  tf.keras.optimizers.AdamW(learning_rate = 1e-6, clipnorm = 1.0)

classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

with open('train_log.out', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.

    history = classifier_model.fit(x=train_ds,
                                   validation_data=val_ds,
                                   epochs=epochs)


    loss, accuracy = classifier_model.evaluate(test_ds)
    print()
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')


dataset_name = 'sairaudet'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

classifier_model.save(saved_model_path, include_optimizer=False)


